"""
Assembles the complete system prompt from individual file components.

Order:
  1. BASE_PROMPT.txt   – immutable core
  2. Current datetime   – local time + timezone
  3. MEMORIES.txt      – long-term user facts
  4. Agenda (from DB)   – active goals / working memory
  5. skills/*.md       – capability documentation
"""

import logging
import re
import threading
from datetime import datetime
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
AGENDA_LIMIT   = 5_000
SKILLS_LIMIT   = 2_000  # TOC-only; individual skills loaded on demand

# Configured timezone — set by main.py at startup via set_timezone().
_timezone: str = "UTC"

# Lock guarding read-modify-write operations on MEMORIES.txt.
_memories_lock = threading.Lock()

# Configured tool profiles — set by main.py at startup via set_tool_profiles().
_tool_profiles: dict[str, dict] = {}

# Cached parsed BASE_PROMPT sections — populated on first call to _get_sections().
_cached_sections: list[tuple[str, set[str], str]] | None = None


def set_component_limits(memories: int = 10_000, agenda: int = 5_000,
                         skills: int = 2_000) -> None:
    """Override component size limits from config.yaml."""
    global MEMORIES_LIMIT, AGENDA_LIMIT, SKILLS_LIMIT
    MEMORIES_LIMIT = memories
    AGENDA_LIMIT = agenda
    SKILLS_LIMIT = skills
    logger.info("Component size limits: memories=%d, agenda=%d, skills=%d",
                memories, agenda, skills)


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
    """Return cached parsed BASE_PROMPT sections (parses once on first call)."""
    global _cached_sections
    if _cached_sections is None:
        raw = prompt_loader.load("BASE_PROMPT.txt")
        _cached_sections = _parse_sections(raw)
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
             available_tools: Optional[set[str]] = None) -> str:
    """
    Build and return the full system prompt string.

    ``extra_summary`` is an optional compaction summary injected between
    Agenda and SKILLS when context has been compacted.

    ``available_tools``, when provided, filters BASE_PROMPT sections to only
    include those relevant to the given tool set.  When None (default), all
    sections are included (backward compatible).
    """
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

    memories = _read(MEMORIES_FILE)
    if memories:
        sections.append(f"# User Memories\n\n{memories}")

    agenda = database.get_active_agenda_text(thread_id=thread_id)
    if agenda:
        sections.append(f"# Active Agenda\n\n{agenda}")

    if extra_summary:
        sections.append(f"# Conversation Summary\n\n{extra_summary}")

    skills_toc = _read_skills_toc()
    sections.append(f"# Skills\n\n{skills_toc}")

    return "\n\n---\n\n".join(sections)


def check_component_sizes() -> dict[str, bool]:
    """
    Return a dict indicating which components exceed their size thresholds.
    Keys: 'memories', 'agenda', 'skills'
    """
    memories_len = len(_read(MEMORIES_FILE))
    agenda_len    = len(database.get_active_agenda_text())
    skills_toc_len = len(_read_skills_toc())
    return {
        "memories": memories_len > MEMORIES_LIMIT,
        "agenda":    agenda_len    > AGENDA_LIMIT,
        "skills":   skills_toc_len > SKILLS_LIMIT,
    }


def update_memories(content: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        MEMORIES_FILE.write_text(content, encoding="utf-8")
    logger.info("MEMORIES.txt updated (%d chars)", len(content))
    threading.Thread(
        target=data_versioning.auto_commit, args=("memory: consolidation",),
        daemon=True,
    ).start()


def append_memory(entry: str) -> int:
    """Append a memory entry to MEMORIES.txt. Returns the new total length."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        existing = _read(MEMORIES_FILE)
        if existing:
            new_content = existing + "\n" + entry.strip()
        else:
            new_content = entry.strip()
        MEMORIES_FILE.write_text(new_content, encoding="utf-8")
    logger.info("MEMORIES.txt appended (%d chars total)", len(new_content))
    threading.Thread(
        target=data_versioning.auto_commit, args=("memory: append",),
        daemon=True,
    ).start()
    return len(new_content)


def add_skill(skill_name: str, documentation: str,
              summary: Optional[str] = None) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    if summary:
        content = f"{summary.strip()}\n\n{documentation.strip()}"
    else:
        # Fallback: use first line of documentation as summary.
        content = documentation.strip()
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
