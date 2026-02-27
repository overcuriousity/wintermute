"""
Skill Stats — YAML-backed usage tracking for skills.

Tracks read counts, session outcomes, and staleness for each skill file.
Pure bookkeeping — zero LLM calls.  Persisted to data/skill_stats.yaml
alongside the skills themselves (versioned in the data/ git repo).

Public API:
  init()                              — load from YAML on startup
  record_read(skill_name)             — increment read counter + timestamp
  record_skill_written(skill_name)    — increment version counter
  record_session_outcome(names, ok)   — correlate session result with skills
  flush()                             — persist to YAML + auto-commit
  get_all()                           — all stats (for enrichment)
  get_unused_skills(days=90)          — candidates for retirement
  remove_skill(skill_name)            — clean up on archive
"""

from __future__ import annotations

import logging
import threading
import time as _time
from pathlib import Path
from typing import Any

import yaml

from wintermute.infra import data_versioning

logger = logging.getLogger(__name__)

_YAML_PATH = Path("data/skill_stats.yaml")
_lock = threading.Lock()
_skills: dict[str, dict[str, Any]] = {}


def init() -> None:
    """Load skill stats from YAML on startup."""
    global _skills
    with _lock:
        try:
            if _YAML_PATH.exists():
                raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8")) or {}
                _skills = raw.get("skills", {})
            else:
                _skills = {}
        except Exception:
            logger.debug("[skill_stats] Failed to load YAML", exc_info=True)
            _skills = {}
    logger.info("[skill_stats] Loaded stats for %d skills", len(_skills))


def _ensure_entry(name: str) -> dict[str, Any]:
    """Return or create a stats entry for the given skill name."""
    if name not in _skills:
        now = _time.time()
        _skills[name] = {
            "created": now,
            "last_read": now,  # grace period: new skills won't retire for 90 days
            "read_count": 0,
            "sessions_loaded": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_updated": _time.time(),
            "version": 1,
        }
    return _skills[name]


def record_read(skill_name: str) -> None:
    """Increment read counter and update last_read timestamp."""
    with _lock:
        entry = _ensure_entry(skill_name)
        entry["read_count"] += 1
        entry["last_read"] = _time.time()


def record_skill_written(skill_name: str) -> None:
    """Increment version counter on skill creation/update."""
    with _lock:
        entry = _ensure_entry(skill_name)
        entry["version"] = entry.get("version", 0) + 1
        entry["last_updated"] = _time.time()


def record_session_outcome(skill_names: list[str], success: bool) -> None:
    """Correlate a session result with the skills it loaded."""
    with _lock:
        for name in skill_names:
            entry = _ensure_entry(name)
            entry["sessions_loaded"] += 1
            if success:
                entry["success_count"] += 1
            else:
                entry["failure_count"] += 1


def flush() -> None:
    """Persist current stats to YAML and queue a background auto-commit."""
    with _lock:
        snapshot = dict(_skills)
    if not snapshot:
        return
    try:
        _YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
        _YAML_PATH.write_text(
            yaml.dump({"skills": snapshot}, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        data_versioning.commit_async("skill_stats: update")
    except Exception:
        logger.debug("[skill_stats] Failed to flush YAML", exc_info=True)


def get_all() -> dict[str, dict[str, Any]]:
    """Return a copy of all skill stats."""
    with _lock:
        return {k: dict(v) for k, v in _skills.items()}


def get_unused_skills(days: int = 90) -> list[str]:
    """Return skill names not read in the last N days."""
    cutoff = _time.time() - (days * 86400)
    result: list[str] = []
    with _lock:
        for name, entry in _skills.items():
            last_read = entry.get("last_read", 0.0)
            if last_read < cutoff:
                result.append(name)
    return result


def remove_skill(skill_name: str) -> None:
    """Remove a skill's stats entry (called on archive)."""
    with _lock:
        _skills.pop(skill_name, None)
