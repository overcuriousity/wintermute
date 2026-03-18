"""
Read / write operations for skills via ``skill_store``.

Thin wrapper providing input validation around ``skill_store`` operations.
Replaces the former file-based ``add_skill`` logic (#81).
"""

import logging
import re
from typing import Optional

from wintermute.infra import data_versioning
from wintermute.infra import skill_store

logger = logging.getLogger(__name__)

# Only allow safe characters in skill names: alphanumerics, hyphens, underscores.
_SAFE_SKILL_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')


def _validate_skill_name(skill_name: str) -> str:
    """Validate and return a safe skill name, or raise ValueError."""
    if not skill_name or not _SAFE_SKILL_NAME_RE.match(skill_name):
        raise ValueError(
            f"Invalid skill name {skill_name!r}: must be alphanumeric "
            f"(hyphens and underscores allowed, no path separators or dots)"
        )
    return skill_name


def add_skill(skill_name: str, documentation: str,
              summary: Optional[str] = None) -> None:
    """Create or update a skill in the active skill store."""
    skill_name = _validate_skill_name(skill_name)
    # Derive summary from documentation first line when not provided,
    # preserving the TOC/search UX that depends on non-empty summaries.
    if summary is not None and summary.strip():
        final_summary = summary.strip()
    else:
        doc_stripped = documentation.strip()
        if doc_stripped:
            final_summary = doc_stripped.splitlines()[0]
        else:
            final_summary = ""
    skill_store.add(skill_name, final_summary, documentation)
    logger.info("Skill '%s' written via skill_store", skill_name)
    data_versioning.commit_async(f"skill: {skill_name}")


def read_skill(skill_name: str) -> Optional[dict]:
    """Read a skill by name.  Returns dict or None."""
    skill_name = _validate_skill_name(skill_name)
    return skill_store.get(skill_name)


def search_skills(query: str, top_k: int = 5) -> list[dict]:
    """Search skills by relevance query."""
    return skill_store.search(query, top_k)


