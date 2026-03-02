"""
Read / write operations for skill files (``data/skills/*.md``).

Extracted from ``prompt_assembler`` (#81) so that prompt assembly and
skill persistence live in separate modules.
"""

import logging
import re
from typing import Optional

from wintermute.infra import data_versioning
from wintermute.infra.paths import SKILLS_DIR

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
    """Create or update a skill markdown file under ``data/skills/``."""
    skill_name = _validate_skill_name(skill_name)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    # Ensure resolved path stays within SKILLS_DIR.
    if not skill_file.resolve().parent == SKILLS_DIR.resolve():
        raise ValueError(f"Skill path escapes skills directory: {skill_name!r}")
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
    data_versioning.commit_async(f"skill: {skill_name}")
