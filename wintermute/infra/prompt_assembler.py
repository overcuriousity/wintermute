"""
Assembles the complete system prompt from individual file components.

Order:
  1. BASE_PROMPT.txt          – immutable core
  2. Current datetime          – local time + timezone
  3. Memories (vector store)   – long-term user facts
  4. Tasks (from DB)           – active goals, reminders, scheduled actions
  5. System Observations      – runtime diagnostics (main thread only)
  6. Predictions & Patterns   – dreaming predictions + promoted schemas (main thread only)
  7. Conversation Summary     – compaction summary (when context has been compacted)
  8. Skills TOC               – query-ranked when vector backend is active
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
from wintermute.infra import prompt_loader

logger = logging.getLogger(__name__)

# Size thresholds (characters) that trigger AI summarisation.
# Defaults — overridden at startup via set_component_limits() from config.yaml.
TASKS_LIMIT    = 5_000
SKILLS_LIMIT   = 2_000  # TOC-only; individual skills loaded on demand

# Configured timezone — set by main.py at startup via set_timezone().
_timezone: str = "UTC"

# Cached parsed BASE_PROMPT sections — populated on first call to _get_sections().
_cached_sections: list[tuple[str, set[str], str]] | None = None
_cached_sections_mtime: float = 0.0
_sections_lock = threading.Lock()


def set_component_limits(memories: int = 10_000, tasks: int = 5_000,
                         skills: int = 2_000, **_compat) -> None:
    """Override component size limits from config.yaml.

    The ``memories`` parameter is accepted for backward compatibility but
    ignored — memories are always injected via ranked retrieval.
    """
    global TASKS_LIMIT, SKILLS_LIMIT
    TASKS_LIMIT = tasks
    SKILLS_LIMIT = skills
    logger.info("Component size limits: tasks=%d, skills=%d", tasks, skills)


def set_timezone(tz: str) -> None:
    """Set the timezone used for datetime injection into the system prompt."""
    global _timezone
    _timezone = tz
    logger.info("Prompt assembler timezone set to %s", tz)


def get_timezone() -> str:
    """Return the configured timezone string."""
    return _timezone


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


def _assemble_base(available_tools: set[str] | None = None,
                   nl_tools: set[str] | None = None) -> str:
    """Assemble the BASE_PROMPT, optionally filtering sections by available tools.

    When *available_tools* is None, all sections are included (backward
    compatible — main session behavior).

    When *available_tools* is provided, only sections where at least one
    required tool is in the set (or the section has no requirements, i.e.
    ``always``) are included.

    When *nl_tools* is provided, sections that have an ``_nl`` variant are
    swapped: if the section's required tools overlap with *nl_tools*, the
    ``_nl`` variant is used instead of the base version.  For example,
    ``delegation_nl`` replaces ``delegation`` when ``worker_delegation``
    is NL-translated.
    """
    sections = _get_sections()
    parts: list[str] = []

    # Collect section names to detect _nl variant pairs.
    section_names = {s[0] for s in sections}

    for name, required_tools, content in sections:
        if available_tools is not None and required_tools:
            # Section requires specific tools — include only if at least one is available.
            if not required_tools & available_tools:
                continue

        # NL variant selection: if both "X" and "X_nl" exist, pick based
        # on whether the section's required tools are NL-translated.
        if name.endswith("_nl"):
            # Include the _nl variant only when its tools are NL-enabled.
            if not nl_tools or not (required_tools & nl_tools):
                continue
        elif f"{name}_nl" in section_names:
            # Base variant has an _nl counterpart — skip if the NL one applies.
            if nl_tools and required_tools and (required_tools & nl_tools):
                continue

        parts.append(content)

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
    return combined[:300]


_self_model_cache: tuple[float, str] = (0.0, "")  # (mtime, summary)
_self_model_path: Optional[Path] = None  # Set at startup via set_self_model_path()
_self_model_cache_lock = threading.Lock()


def set_self_model_path(path: "Path | str") -> None:
    """Configure the self-model YAML path (called at startup)."""
    global _self_model_path, _self_model_cache
    with _self_model_cache_lock:
        _self_model_path = Path(path)
        _self_model_cache = (0.0, "")  # Invalidate cache on path change.


def _get_self_model_summary() -> str:
    """Read the self-model prose summary, cached by file mtime."""
    global _self_model_cache
    with _self_model_cache_lock:
        if _self_model_path is None:
            return ""
        path = _self_model_path
    try:
        if not path.exists():
            return ""
        mtime = path.stat().st_mtime
        with _self_model_cache_lock:
            if mtime == _self_model_cache[0]:
                return _self_model_cache[1]
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        summary = (data.get("summary") or "").strip()[:300]
        with _self_model_cache_lock:
            _self_model_cache = (mtime, summary)
        return summary
    except Exception:
        logger.debug("Failed to read self-model summary", exc_info=True)
        return ""


# Hard cap for prediction text injected into the system prompt.
_PREDICTIONS_CAP = 800


def fetch_predictions() -> list[str]:
    """Fetch prediction lines from memory store (blocking I/O).

    Intended to be called via ``asyncio.to_thread`` from async callers,
    then passed to ``assemble(prediction_results=...)``.
    """
    from wintermute.infra import memory_store

    dreaming_cfg = memory_store.get_dreaming_config()
    if not dreaming_cfg.get("prediction_inject_prompt", True):
        return []

    lines: list[str] = []
    try:
        for source in ("dreaming_prediction", "dreaming_schema"):
            entries = memory_store.get_by_source(source, limit=20)
            for e in entries:
                text = e.get("text", "").strip()
                if text:
                    lines.append(f"- {text}")
    except Exception as exc:
        logger.debug("Prediction retrieval failed: %s", exc)
    return lines



def _read_skills_toc(query: Optional[str] = None) -> str:
    """Build a TOC of skills from the skill store.

    When *query* is provided and the skill backend supports vector search,
    skills are ranked by relevance (mirroring the memory query-ranking
    pattern).  Otherwise falls back to listing all skills alphabetically.

    Always returns a non-empty string so the LLM knows the skill system
    is available.
    """
    from wintermute.infra import skill_store

    header = 'Use the skill tool (action "read" or "search") to retrieve details.'
    entries: list[str] = []
    ranked = False
    try:
        if query and skill_store.is_vector_enabled():
            results = skill_store.search(query)
            ranked = True
        else:
            results = skill_store.get_all()
            ranked = False
        for rec in results:
            name = rec.get("name", "unknown")
            summary = rec.get("summary", "").strip()
            score = rec.get("score", 0)
            if ranked and summary:
                entries.append(f"- {name} ({score:.2f}) — {summary}")
            elif summary:
                entries.append(f"- {name} — {summary}")
            else:
                entries.append(f"- {name}")
    except Exception as exc:
        logger.warning("skill_store skills TOC failed: %s", exc)
    if entries:
        label = "relevance-ranked" if ranked else "all"
        return f"{header} ({label})\n" + "\n".join(entries)
    return header


def assemble(extra_summary: Optional[str] = None, thread_id: Optional[str] = None,
             available_tools: Optional[set[str]] = None,
             query: Optional[str] = None,
             memory_results: Optional[list[dict]] = None,
             prompt_mode: str = "full",
             tool_profiles: Optional[dict[str, dict]] = None,  # deprecated — profiles now in tool schemas
             nl_tools: Optional[set[str]] = None,
             prediction_results: Optional[list[str]] = None) -> str:
    """
    Build and return the full system prompt string.

    ``extra_summary`` is an optional compaction summary injected between
    TASKS and SKILLS when context has been compacted.

    ``available_tools``, when provided, filters BASE_PROMPT sections to only
    include those relevant to the given tool set.  When None (default), all
    sections are included (backward compatible).

    ``query``, when provided, triggers relevance-ranked memory retrieval
    (works with any backend) instead of loading the full file.

    ``memory_results``, when provided, uses pre-fetched memory search results
    instead of calling memory_store.search() synchronously. Callers in async
    contexts should fetch memories via ``asyncio.to_thread`` and pass them here
    to avoid blocking the event loop.

    ``prediction_results``, when provided, is a pre-fetched list of prediction
    entries to inject into the Predictions & Patterns section.  Callers should
    fetch this off-thread (e.g. via ``asyncio.to_thread``) to avoid blocking
    the event loop.  When None, predictions are not included.

    ``prompt_mode`` controls how much context is injected:
      - ``"full"`` (default): all sections (memories, tasks, skills, etc.)
      - ``"minimal"``: only Core Instructions + Current Time + Conversation Summary
    """
    from wintermute.infra import memory_store

    minimal = prompt_mode == "minimal"

    sections: list[str] = []

    base = _assemble_base(available_tools, nl_tools=nl_tools)
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
            # Treat pre-fetched results as authoritative ([] means search
            # returned nothing or failed — do not retry).
            if memory_results:
                memories_text = "\n".join(r["text"] for r in memory_results)
                sections.append(f"# User Memories (relevance-ranked)\n\n{memories_text}")
        elif query:
            try:
                results = memory_store.search(query)
            except Exception as exc:
                logger.warning("Memory search failed in prompt assembly, continuing without: %s", exc)
                results = []
            if results:
                memories_text = "\n".join(r["text"] for r in results)
                sections.append(f"# User Memories (relevance-ranked)\n\n{memories_text}")

        tasks_text = database.get_active_tasks_text(thread_id=thread_id)
        if tasks_text:
            sections.append(f"# Active Tasks\n\n{tasks_text}")

        # Reflection observations — main thread only
        if available_tools is None:
            reflection = _get_reflection_observations()
            if reflection:
                sections.append(f"# System Observations\n\n{reflection}")

        # Operational self-model — main thread only (skip when path explicitly unset)
        if available_tools is None and _self_model_path is not None:
            sm_summary = _get_self_model_summary()
            if sm_summary:
                sections.append(f"# Operational Self-Model\n\n{sm_summary}")

        # Predictions & Patterns — main thread only
        if available_tools is None and prediction_results:
            predictions_text = "\n".join(prediction_results)[:_PREDICTIONS_CAP]
            sections.append(f"# Predictions & Patterns\n\n{predictions_text}")

    if extra_summary:
        sections.append(f"# Conversation Summary\n\n{extra_summary}")

    if not minimal:
        skills_toc = _read_skills_toc(query=query)
        sections.append(f"# Skills\n\n{skills_toc}")

    return "\n\n---\n\n".join(sections)


def check_component_sizes() -> dict[str, bool]:
    """
    Return a dict indicating which components exceed their size thresholds.
    Keys: 'memories', 'tasks', 'skills'

    Memories are always injected via ranked retrieval from the memory backend,
    so the memories component is never oversized.
    """
    memories_oversized = False
    tasks_len     = len(database.get_active_tasks_text())
    skills_toc_len = len(_read_skills_toc())
    return {
        "memories": memories_oversized,
        "tasks":    tasks_len    > TASKS_LIMIT,
        "skills":   skills_toc_len > SKILLS_LIMIT,
    }
