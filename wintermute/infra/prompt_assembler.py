"""
Assembles the complete system prompt from individual file components.

Order:
  1. BASE_PROMPT.txt   – immutable core
  2. Current datetime   – local time + timezone
  3. MEMORIES.txt      – long-term user facts
  4. Tasks (from DB)    – active goals, reminders, scheduled actions
  5. skills/*.md       – capability documentation
"""

import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from wintermute.infra import database
from wintermute.infra import data_versioning
from wintermute.infra import prompt_loader

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MEMORIES_FILE     = DATA_DIR / "MEMORIES.txt"
SKILLS_DIR        = DATA_DIR / "skills"

# Size thresholds (characters) that trigger AI summarisation.
# Defaults — overridden at startup via set_component_limits() from config.yaml.
MEMORIES_LIMIT = 10_000
TASKS_LIMIT    = 5_000
SKILLS_LIMIT   = 2_000  # TOC-only; individual skills loaded on demand

# Configured timezone — set by main.py at startup via set_timezone().
_timezone: str = "UTC"

# Lock guarding read-modify-write operations on MEMORIES.txt.
_memories_lock = threading.Lock()

# Configured tool profiles — set by main.py at startup via set_tool_profiles().
_tool_profiles: dict[str, dict] = {}

# Cached parsed BASE_PROMPT sections — populated on first call to _get_sections().
_cached_sections: list[tuple[str, set[str], str]] | None = None
_cached_sections_mtime: float = 0.0
_sections_lock = threading.Lock()

# Self-model profiler — set by main.py at startup via set_self_model().
_self_model_profiler = None


def set_self_model(profiler) -> None:
    """Set the SelfModelProfiler for system prompt injection."""
    global _self_model_profiler
    _self_model_profiler = profiler
    logger.info("Self-model profiler registered with prompt assembler")


def set_component_limits(memories: int = 10_000, tasks: int = 5_000,
                         skills: int = 2_000, **_compat) -> None:
    """Override component size limits from config.yaml."""
    global MEMORIES_LIMIT, TASKS_LIMIT, SKILLS_LIMIT
    MEMORIES_LIMIT = memories
    TASKS_LIMIT = tasks
    SKILLS_LIMIT = skills
    logger.info("Component size limits: memories=%d, tasks=%d, skills=%d",
                memories, tasks, skills)


def set_timezone(tz: str) -> None:
    """Set the timezone used for datetime injection into the system prompt."""
    global _timezone
    _timezone = tz
    logger.info("Prompt assembler timezone set to %s", tz)


def set_tool_profiles(profiles: dict[str, dict]) -> None:
    """Set tool profiles from config (called once at startup by main.py)."""
    global _tool_profiles
    _tool_profiles = dict(profiles) if profiles else {}
    if _tool_profiles:
        logger.info("Tool profiles loaded: %s", ", ".join(_tool_profiles))


def get_tool_profiles() -> dict[str, dict]:
    """Return the configured tool profiles."""
    return _tool_profiles


# ---------------------------------------------------------------------------
# Sectioned BASE_PROMPT parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r'<!--\s*section:\s*(\S+)\s+requires:\s*(\S+)\s*-->'
)


def _parse_sections(raw: str) -> list[tuple[str, set[str], str]]:
    """Parse BASE_PROMPT into (name, required_tools, content) tuples.

    Section markers have the format:
        <!-- section: <name> requires: <tool1>,<tool2> -->
    or:
        <!-- section: <name> requires: always -->

    Sections with ``always`` as the requirement are always included.
    """
    sections: list[tuple[str, set[str], str]] = []
    # Split on section markers, keeping the marker groups.
    parts = _SECTION_RE.split(raw)
    # parts[0] is text before the first marker (if any).
    # Then groups of (name, requires, text) follow.
    preamble = parts[0].strip()
    if preamble:
        # Text before any marker → always-included section.
        sections.append(("preamble", set(), preamble))

    idx = 1
    while idx + 2 < len(parts):
        name = parts[idx].strip()
        requires_str = parts[idx + 1].strip()
        content = parts[idx + 2].strip()
        idx += 3

        if requires_str == "always":
            required_tools: set[str] = set()
        else:
            required_tools = {t.strip() for t in requires_str.split(",") if t.strip()}

        if content:
            sections.append((name, required_tools, content))

    return sections


def _get_sections() -> list[tuple[str, set[str], str]]:
    """Return cached parsed BASE_PROMPT sections.

    Re-parses automatically when the underlying file's mtime changes.
    Thread-safe via ``_sections_lock``.
    """
    global _cached_sections, _cached_sections_mtime
    base_prompt_path = prompt_loader.PROMPTS_DIR / "BASE_PROMPT.txt"
    try:
        current_mtime = base_prompt_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    with _sections_lock:
        if _cached_sections is None or current_mtime != _cached_sections_mtime:
            raw = prompt_loader.load("BASE_PROMPT.txt")
            _cached_sections = _parse_sections(raw)
            _cached_sections_mtime = current_mtime
        return _cached_sections


def _assemble_base(available_tools: set[str] | None = None) -> str:
    """Assemble the BASE_PROMPT, optionally filtering sections by available tools.

    When *available_tools* is None, all sections are included (backward
    compatible — main session behavior).

    When *available_tools* is provided, only sections where at least one
    required tool is in the set (or the section has no requirements, i.e.
    ``always``) are included.

    The ``delegation`` section gets dynamic profile names appended when
    tool profiles are configured and ``spawn_sub_session`` is available.
    """
    sections = _get_sections()
    parts: list[str] = []

    for name, required_tools, content in sections:
        if available_tools is not None and required_tools:
            # Section requires specific tools — include only if at least one is available.
            if not required_tools & available_tools:
                continue
        text = content
        # Inject tool profile names into the delegation section.
        if name == "delegation" and _tool_profiles:
            if available_tools is None or "spawn_sub_session" in available_tools:
                profile_names = ", ".join(sorted(_tool_profiles))
                text += f"\n\nAvailable tool profiles: {profile_names}"
        parts.append(text)

    return "\n\n".join(parts)


def _get_reflection_observations() -> str:
    """Return recent reflection findings (last 24h) for the main thread prompt."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    lines: list[str] = []
    for action in ("reflection_rule", "reflection_analysis"):
        try:
            entries = database.get_interaction_log(limit=10, action_filter=action)
        except Exception:
            continue
        for e in entries:
            ts_str = e.get("timestamp", "")
            try:
                ts = datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                continue
            if action == "reflection_rule":
                rule = e.get("input", "unknown")
                detail = e.get("output", "")
                try:
                    parsed = json.loads(detail)
                    detail = parsed.get("summary", detail)
                except (json.JSONDecodeError, AttributeError):
                    pass
                lines.append(f"- Rule '{rule}': {detail[:120]}")
            else:
                text = e.get("output", "")[:200]
                lines.append(f"- Analysis: {text}")
    if not lines:
        return ""
    combined = "\n".join(lines[:8])
    return combined[:500]


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return default


def _read_skills_toc() -> str:
    """Build a TOC of skills with first-line summaries and exact file paths.

    Always returns a non-empty string (includes instructions even when no
    skills exist) so the LLM knows the skill system is available.
    """
    header = 'Load a skill with read_file when relevant to the current task.'
    entries: list[str] = []
    if SKILLS_DIR.exists():
        for md_file in sorted(SKILLS_DIR.glob("*.md")):
            content = _read(md_file)
            if content:
                summary = content.split("\n", 1)[0].strip()
                rel_path = f"data/skills/{md_file.name}"
                entries.append(f"- {rel_path} — {summary}")
    if entries:
        return header + "\n" + "\n".join(entries)
    return header


def assemble(extra_summary: Optional[str] = None, thread_id: Optional[str] = None,
             available_tools: Optional[set[str]] = None,
             query: Optional[str] = None,
             memory_results: Optional[list[dict]] = None,
             prompt_mode: str = "full") -> str:
    """
    Build and return the full system prompt string.

    ``extra_summary`` is an optional compaction summary injected between
    TASKS and SKILLS when context has been compacted.

    ``available_tools``, when provided, filters BASE_PROMPT sections to only
    include those relevant to the given tool set.  When None (default), all
    sections are included (backward compatible).

    ``query``, when provided alongside a vector memory backend, triggers
    relevance-ranked memory retrieval instead of loading the full file.

    ``memory_results``, when provided, uses pre-fetched memory search results
    instead of calling memory_store.search() synchronously. Callers in async
    contexts should fetch memories via ``asyncio.to_thread`` and pass them here
    to avoid blocking the event loop.

    ``prompt_mode`` controls how much context is injected:
      - ``"full"`` (default): all sections (memories, tasks, skills, etc.)
      - ``"minimal"``: only Core Instructions + Current Time + Conversation Summary
    """
    from wintermute.infra import memory_store

    minimal = prompt_mode == "minimal"

    sections: list[str] = []

    base = _assemble_base(available_tools)
    sections.append(f"# Core Instructions\n\n{base}")

    # Inject current local datetime so the LLM has accurate time awareness.
    try:
        tz = ZoneInfo(_timezone)
        now = datetime.now(tz)
        time_str = now.strftime("%A, %Y-%m-%d %H:%M %Z")
        sections.append(f"# Current Time\n\n{time_str}")
    except Exception as exc:
        logger.warning("Could not determine local time: %s", exc)

    if not minimal:
        if memory_results is not None:
            if memory_results:
                memories_text = "\n".join(r["text"] for r in memory_results)
                sections.append(f"# User Memories (relevance-ranked)\n\n{memories_text}")
        elif memory_store.is_vector_enabled() and query:
            try:
                results = memory_store.search(query)
            except Exception as exc:
                logger.warning("Memory search failed in prompt assembly, continuing without: %s", exc)
                results = []
            if results:
                memories_text = "\n".join(r["text"] for r in results)
                sections.append(f"# User Memories (relevance-ranked)\n\n{memories_text}")
        else:
            memories = _read(MEMORIES_FILE)
            if memories:
                sections.append(f"# User Memories\n\n{memories}")

        tasks_text = database.get_active_tasks_text(thread_id=thread_id)
        if tasks_text:
            sections.append(f"# Active Tasks\n\n{tasks_text}")

        # Reflection observations — main thread only
        if available_tools is None:
            reflection = _get_reflection_observations()
            if reflection:
                sections.append(f"# System Observations\n\n{reflection}")

            if _self_model_profiler:
                sm = _self_model_profiler.get_summary()
                if sm:
                    sections.append(f"# Self-Assessment\n\n{sm}")

    if extra_summary:
        sections.append(f"# Conversation Summary\n\n{extra_summary}")

    if not minimal:
        skills_toc = _read_skills_toc()
        sections.append(f"# Skills\n\n{skills_toc}")

    return "\n\n---\n\n".join(sections)


def check_component_sizes() -> dict[str, bool]:
    """
    Return a dict indicating which components exceed their size thresholds.
    Keys: 'memories', 'tasks', 'skills'

    When vector memory is enabled, the full MEMORIES.txt is never injected
    into the system prompt (only relevance-ranked results are), so the
    memories size check is skipped.
    """
    from wintermute.infra import memory_store

    if memory_store.is_vector_enabled():
        memories_oversized = False
    else:
        memories_oversized = len(_read(MEMORIES_FILE)) > MEMORIES_LIMIT
    tasks_len     = len(database.get_active_tasks_text())
    skills_toc_len = len(_read_skills_toc())
    return {
        "memories": memories_oversized,
        "tasks":    tasks_len    > TASKS_LIMIT,
        "skills":   skills_toc_len > SKILLS_LIMIT,
    }


def update_memories(content: str) -> None:
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        MEMORIES_FILE.write_text(content, encoding="utf-8")
    logger.info("MEMORIES.txt updated (%d chars)", len(content))
    if memory_store.is_vector_enabled():
        try:
            entries = [l.strip() for l in content.strip().splitlines() if l.strip()]
            memory_store.replace_all(entries)
        except Exception as exc:
            logger.error("Failed to sync vector store on update_memories: %s", exc)
    threading.Thread(
        target=data_versioning.auto_commit, args=("memory: consolidation",),
        daemon=True,
    ).start()


def append_memory(entry: str, source: str = "unknown") -> int:
    """Append a memory entry to MEMORIES.txt. Returns the new total length."""
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        existing = _read(MEMORIES_FILE)
        if existing:
            new_content = existing + "\n" + entry.strip()
        else:
            new_content = entry.strip()
        MEMORIES_FILE.write_text(new_content, encoding="utf-8")
    logger.info("MEMORIES.txt appended (%d chars total)", len(new_content))
    if memory_store.is_vector_enabled():
        try:
            memory_store.add(entry.strip(), source=source)
        except Exception as exc:
            logger.error("Failed to add memory to vector store: %s", exc)
    threading.Thread(
        target=data_versioning.auto_commit, args=("memory: append",),
        daemon=True,
    ).start()
    return len(new_content)


def add_skill(skill_name: str, documentation: str,
              summary: Optional[str] = None) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    is_update = skill_file.exists()
    if summary:
        content = f"{summary.strip()}\n\n{documentation.strip()}"
    else:
        # Fallback: use first line of documentation as summary.
        content = documentation.strip()
    # Append changelog entry when overwriting an existing skill, preserving
    # the existing changelog from the old file content.
    if is_update:
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # Read old file to preserve its changelog section.
        try:
            existing = skill_file.read_text(encoding="utf-8")
        except OSError:
            existing = ""
        existing_changelog = ""
        if "## Changelog" in existing:
            idx = existing.index("## Changelog")
            existing_changelog = existing[idx:].rstrip()
        # Strip any changelog the LLM may have included in the new content.
        if "## Changelog" in content:
            content = content[: content.index("## Changelog")].rstrip()
        # Build the updated changelog section.
        if existing_changelog:
            changelog_section = f"{existing_changelog}\n- {date_str}: updated"
        else:
            changelog_section = f"## Changelog\n- {date_str}: updated"
        content = f"{content}\n\n{changelog_section}"
    skill_file.write_text(content, encoding="utf-8")
    logger.info("Skill '%s' written to %s", skill_name, skill_file)
    threading.Thread(
        target=data_versioning.auto_commit, args=(f"skill: {skill_name}",),
        daemon=True,
    ).start()


def merge_consolidated_memories(snapshot: str, consolidated: str) -> None:
    """Atomically write *consolidated* memories while preserving any lines
    that were appended to MEMORIES.txt after *snapshot* was taken.

    This solves the race condition where ``append_memory()`` is called while
    the dreaming consolidation LLM call is in flight: the snapshot (taken
    before the LLM call) is diffed against the current file content, and any
    newly appended lines are tacked onto the consolidated result before
    writing.

    The entire read-diff-write cycle runs under ``_memories_lock`` so no
    appends can slip through.
    """
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_lines = set(snapshot.strip().splitlines())
    with _memories_lock:
        current = _read(MEMORIES_FILE)
        new_lines = [
            line for line in current.strip().splitlines()
            if line not in snapshot_lines
        ]
        merged = consolidated.strip()
        if new_lines:
            merged = merged + "\n" + "\n".join(new_lines)
            logger.info(
                "merge_consolidated_memories: preserved %d appended line(s)",
                len(new_lines),
            )
        MEMORIES_FILE.write_text(merged, encoding="utf-8")
    logger.info("MEMORIES.txt merged-write (%d chars)", len(merged))
    if memory_store.is_vector_enabled():
        try:
            entries = [l.strip() for l in merged.splitlines() if l.strip()]
            memory_store.replace_all(entries)
        except Exception as exc:
            logger.error("Failed to sync vector store on merge_consolidated: %s", exc)
