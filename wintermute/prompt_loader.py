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
    "DREAM_PULSE_PROMPT.txt",
    "WORKER_MINIMAL.txt",
    "WORKER_OBJECTIVE.txt",
    "WORKER_CONTINUATION.txt",
    "TURING_STAGE1.txt",
    "TURING_OBJECTIVE_COMPLETION.txt",
    "COMPONENT_OVERSIZE.txt",
]


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
