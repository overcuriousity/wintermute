"""
Central prompt file loader.

All prompt templates live in ``data/prompts/`` as plain-text files.
No hardcoded fallbacks — a missing file is a startup error.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("data/prompts")

REQUIRED_FILES = [
    "BASE_PROMPT.txt",
    "COMPACTION_PROMPT.txt",
    "DREAM_MEMORIES_PROMPT.txt",
    "DREAM_AGENDA_PROMPT.txt",
    "DREAM_SKILLS_DEDUP_PROMPT.txt",
    "DREAM_SKILLS_CONDENSATION_PROMPT.txt",
    "WORKER_MINIMAL.txt",
    "WORKER_OBJECTIVE.txt",
    "WORKER_CONTINUATION.txt",
    "TURING_STAGE1.txt",
    "TURING_OBJECTIVE_COMPLETION.txt",
    "COMPONENT_OVERSIZE.txt",
    "MEMORY_HARVEST_PROMPT.txt",
]


NL_TRANSLATION_FILES = [
    "NL_TRANSLATOR_SET_ROUTINE.txt",
    "NL_TRANSLATOR_SPAWN_SUB_SESSION.txt",
    "NL_TRANSLATOR_ADD_SKILL.txt",
    "NL_TRANSLATOR_AGENDA.txt",
]


def validate_nl_translation() -> None:
    """Check that NL translation prompt files exist.

    Called from main.py only when ``nl_translation.enabled: true``.
    """
    missing = [f for f in NL_TRANSLATION_FILES if not (PROMPTS_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing NL translation prompt files in {PROMPTS_DIR}/: "
            + ", ".join(missing)
        )
    logger.info("NL translation prompt files validated (%d files)", len(NL_TRANSLATION_FILES))


def load_seed(language: str = "en") -> str:
    """Load the seed prompt for the given language code (e.g. 'en', 'de').

    Falls back to English if the requested language file does not exist.
    """
    name = f"SEED_{language}.txt"
    path = PROMPTS_DIR / name
    if not path.is_file():
        logger.warning("Seed prompt %s not found, falling back to SEED_en.txt", name)
        path = PROMPTS_DIR / "SEED_en.txt"
    return path.read_text(encoding="utf-8").strip()


def load(name: str, **kwargs: object) -> str:
    """Read a prompt template from *data/prompts/{name}*.

    If **kwargs** are provided the template is formatted via ``str.format()``.
    Raises ``FileNotFoundError`` if the file is missing.
    """
    path = PROMPTS_DIR / name
    text = path.read_text(encoding="utf-8").strip()
    if kwargs:
        text = text.format(**kwargs)
    return text


def validate_all() -> None:
    """Check that every required prompt file exists.  Call at startup."""
    missing = [f for f in REQUIRED_FILES if not (PROMPTS_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required prompt files in {PROMPTS_DIR}/: "
            + ", ".join(missing)
        )
    logger.info("All %d prompt files validated in %s", len(REQUIRED_FILES), PROMPTS_DIR)
