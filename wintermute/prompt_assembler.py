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
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from wintermute import database
from wintermute import prompt_loader

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
MEMORIES_FILE     = DATA_DIR / "MEMORIES.txt"
SKILLS_DIR        = DATA_DIR / "skills"

# Size thresholds (characters) that trigger AI summarisation
MEMORIES_LIMIT = 10_000
AGENDA_LIMIT    = 5_000
SKILLS_LIMIT   = 20_000

# Configured timezone — set by main.py at startup via set_timezone().
_timezone: str = "UTC"

# Lock guarding read-modify-write operations on MEMORIES.txt.
_memories_lock = threading.Lock()


def set_timezone(tz: str) -> None:
    """Set the timezone used for datetime injection into the system prompt."""
    global _timezone
    _timezone = tz
    logger.info("Prompt assembler timezone set to %s", tz)


def _read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return default


def _read_skills() -> str:
    if not SKILLS_DIR.exists():
        return ""
    parts = []
    for md_file in sorted(SKILLS_DIR.glob("*.md")):
        content = _read(md_file)
        if content:
            parts.append(f"### Skill: {md_file.stem}\n\n{content}")
    return "\n\n---\n\n".join(parts)


def assemble(extra_summary: Optional[str] = None, thread_id: Optional[str] = None) -> str:
    """
    Build and return the full system prompt string.

    ``extra_summary`` is an optional compaction summary injected between
    Agenda and SKILLS when context has been compacted.
    """
    sections: list[str] = []

    base = prompt_loader.load("BASE_PROMPT.txt")
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

    skills = _read_skills()
    if skills:
        sections.append(f"# Skills\n\n{skills}")

    return "\n\n---\n\n".join(sections)


def check_component_sizes() -> dict[str, bool]:
    """
    Return a dict indicating which components exceed their size thresholds.
    Keys: 'memories', 'agenda', 'skills'
    """
    memories_len = len(_read(MEMORIES_FILE))
    agenda_len    = len(database.get_active_agenda_text())
    skills_len   = len(_read_skills())

    return {
        "memories": memories_len > MEMORIES_LIMIT,
        "agenda":    agenda_len    > AGENDA_LIMIT,
        "skills":   skills_len   > SKILLS_LIMIT,
    }


def update_memories(content: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        MEMORIES_FILE.write_text(content, encoding="utf-8")
    logger.info("MEMORIES.txt updated (%d chars)", len(content))


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
    return len(new_content)


def add_skill(skill_name: str, documentation: str) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    skill_file.write_text(documentation, encoding="utf-8")
    logger.info("Skill '%s' written to %s", skill_name, skill_file)
