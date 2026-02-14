"""
Assembles the complete system prompt from individual file components.

Order:
  1. BASE_PROMPT.txt   – immutable core
  2. Current datetime   – local time + timezone
  3. MEMORIES.txt      – long-term user facts
  4. PULSE.txt         – active goals / working memory
  5. skills/*.md       – capability documentation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
BASE_PROMPT_FILE  = DATA_DIR / "BASE_PROMPT.txt"
MEMORIES_FILE     = DATA_DIR / "MEMORIES.txt"
PULSE_FILE        = DATA_DIR / "PULSE.txt"
SKILLS_DIR        = DATA_DIR / "skills"

# Size thresholds (characters) that trigger AI summarisation
MEMORIES_LIMIT = 10_000
PULSE_LIMIT    = 5_000
SKILLS_LIMIT   = 20_000

# Configured timezone — set by main.py at startup via set_timezone().
_timezone: str = "UTC"


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


def assemble(extra_summary: Optional[str] = None) -> str:
    """
    Build and return the full system prompt string.

    ``extra_summary`` is an optional compaction summary injected between
    Pulse and SKILLS when context has been compacted.
    """
    sections: list[str] = []

    base = _read(BASE_PROMPT_FILE)
    if not base:
        logger.warning("BASE_PROMPT.txt is empty or missing – using fallback")
        base = "You are a helpful personal AI assistant."
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

    pulse = _read(PULSE_FILE)
    if pulse:
        sections.append(f"# Active Pulse\n\n{pulse}")

    if extra_summary:
        sections.append(f"# Conversation Summary\n\n{extra_summary}")

    skills = _read_skills()
    if skills:
        sections.append(f"# Skills\n\n{skills}")

    return "\n\n---\n\n".join(sections)


def check_component_sizes() -> dict[str, bool]:
    """
    Return a dict indicating which components exceed their size thresholds.
    Keys: 'memories', 'pulse', 'skills'
    """
    memories_len = len(_read(MEMORIES_FILE))
    pulse_len    = len(_read(PULSE_FILE))
    skills_len   = len(_read_skills())

    return {
        "memories": memories_len > MEMORIES_LIMIT,
        "pulse":    pulse_len    > PULSE_LIMIT,
        "skills":   skills_len   > SKILLS_LIMIT,
    }


def update_memories(content: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_FILE.write_text(content, encoding="utf-8")
    logger.info("MEMORIES.txt updated (%d chars)", len(content))


def append_memory(entry: str) -> int:
    """Append a memory entry to MEMORIES.txt. Returns the new total length."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read(MEMORIES_FILE)
    if existing:
        new_content = existing + "\n" + entry.strip()
    else:
        new_content = entry.strip()
    MEMORIES_FILE.write_text(new_content, encoding="utf-8")
    logger.info("MEMORIES.txt appended (%d chars total)", len(new_content))
    return len(new_content)


def update_pulse(content: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PULSE_FILE.write_text(content, encoding="utf-8")
    logger.info("PULSE.txt updated (%d chars)", len(content))


def add_skill(skill_name: str, documentation: str) -> None:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    skill_file.write_text(documentation, encoding="utf-8")
    logger.info("Skill '%s' written to %s", skill_name, skill_file)
