"""Per-thread configuration overrides.

Provides a two-layer config resolution model:
  per-thread override (SQLite)  →  hardcoded default

Each thread can independently override:
  - ``backend_name``              — pin inference to a named backend
  - ``session_timeout_minutes``   — auto-session-renewal timeout (plumbing for #58)
  - ``sub_sessions_enabled``      — enable/disable sub-session spawning
  - ``system_prompt_mode``        — 'full' or 'minimal' prompt assembly

The ``ThreadConfigManager`` caches configs in memory and persists to SQLite.
All mutations are logged to the interaction_log for auditability.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Optional

from wintermute.infra import database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThreadConfig:
    """Per-thread config overrides.  ``None`` means 'use global default'."""
    backend_name: Optional[str] = None
    session_timeout_minutes: Optional[int] = None
    sub_sessions_enabled: Optional[bool] = None
    system_prompt_mode: Optional[str] = None          # 'full' | 'minimal'

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "ThreadConfig":
        data = json.loads(raw)
        # Only accept known fields.
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ResolvedThreadConfig:
    """Fully resolved config — no ``None`` values remain."""
    backend_name: Optional[str]          # None means "use role-based pool"
    session_timeout_minutes: Optional[int]  # None means "no auto-renewal"
    sub_sessions_enabled: bool
    system_prompt_mode: str              # 'full' | 'minimal'


# Hardcoded fallback defaults (baseline when no per-thread override exists).
_HARDCODED_DEFAULTS = {
    "backend_name": None,
    "session_timeout_minutes": None,
    "sub_sessions_enabled": True,
    "system_prompt_mode": "full",
}

# Valid values for system_prompt_mode.
_VALID_PROMPT_MODES = {"full", "minimal"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: Any) -> bool:
    """Parse a bool from various representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
    raise ValueError(f"Cannot parse {value!r} as bool")


def _parse_optional_int(value: Any) -> Optional[int]:
    """Parse an optional int (None / null / 'null' all mean None)."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in ("null", "none", ""):
            return None
        return int(value)
    return int(value)


# ---------------------------------------------------------------------------
# ThreadConfigManager
# ---------------------------------------------------------------------------

class ThreadConfigManager:
    """In-memory cache + SQLite persistence for per-thread configuration.

    Parameters
    ----------
    available_backends : list[str]
        Names from ``inference_backends`` — used to validate ``backend_name``.
    """

    def __init__(self, available_backends: list[str]) -> None:
        self._available_backends = list(available_backends)
        self._cache: dict[str, ThreadConfig] = {}
        self._load_all()

    # ── Bootstrap ────────────────────────────────────────────────────

    def _load_all(self) -> None:
        """Populate the cache from SQLite on startup."""
        try:
            rows = database.get_all_thread_configs()
            for thread_id, raw_json in rows:
                try:
                    self._cache[thread_id] = ThreadConfig.from_json(raw_json)
                except Exception:
                    logger.warning("Invalid thread_config for %s — skipping", thread_id)
        except Exception:
            logger.debug("Could not load thread_configs (table may not exist yet)")

    # ── Read ─────────────────────────────────────────────────────────

    def get(self, thread_id: str) -> ThreadConfig:
        """Return per-thread overrides (empty ``ThreadConfig`` if none set)."""
        return self._cache.get(thread_id, ThreadConfig())

    def resolve(self, thread_id: str) -> ResolvedThreadConfig:
        """Merge per-thread overrides with hardcoded defaults."""
        tc = self.get(thread_id)

        def _pick(key: str):
            val = getattr(tc, key, None)
            return val if val is not None else _HARDCODED_DEFAULTS[key]

        sub_enabled = _pick("sub_sessions_enabled")
        return ResolvedThreadConfig(
            backend_name=_pick("backend_name"),
            session_timeout_minutes=_pick("session_timeout_minutes"),
            sub_sessions_enabled=bool(sub_enabled) if sub_enabled is not None else True,
            system_prompt_mode=_pick("system_prompt_mode") or "full",
        )

    def resolve_as_dict(self, thread_id: str) -> dict:
        """Return resolved config as a plain dict with source annotations."""
        tc = self.get(thread_id)
        resolved = self.resolve(thread_id)
        result = {}
        for f in fields(ResolvedThreadConfig):
            val = getattr(resolved, f.name)
            override = getattr(tc, f.name, None)
            source = "override" if override is not None else "default"
            result[f.name] = {"value": val, "source": source}
        return result

    def get_available_backends(self) -> list[str]:
        """Return the list of configured backend names."""
        return list(self._available_backends)

    def get_all_overrides(self) -> dict[str, dict]:
        """Return all thread_ids that have custom config (for SSE/debug)."""
        result = {}
        for tid, tc in self._cache.items():
            # Only include threads that actually have at least one override.
            d = asdict(tc)
            has_override = any(v is not None for v in d.values())
            if has_override:
                result[tid] = self.resolve_as_dict(tid)
        return result

    # ── Write ────────────────────────────────────────────────────────

    def set(self, thread_id: str, key: str, value: Any,
            source: str = "api") -> ResolvedThreadConfig:
        """Validate and set a single config key for a thread.

        Returns the updated resolved config.
        Raises ``ValueError`` on invalid key or value.
        """
        tc = self._cache.get(thread_id, ThreadConfig())

        # Validate key.
        if key not in {f.name for f in fields(ThreadConfig)}:
            valid_keys = ", ".join(f.name for f in fields(ThreadConfig))
            raise ValueError(f"Unknown config key {key!r}. Valid keys: {valid_keys}")

        # Validate + coerce value per key.
        if key == "backend_name":
            if value is None or (isinstance(value, str) and value.lower() in ("null", "none", "")):
                value = None
            elif value not in self._available_backends:
                raise ValueError(
                    f"Unknown backend {value!r}. Available: {', '.join(self._available_backends)}"
                )
        elif key == "session_timeout_minutes":
            value = _parse_optional_int(value)
            if value is not None and value < 1:
                raise ValueError("session_timeout_minutes must be >= 1 (or null to disable)")
        elif key == "sub_sessions_enabled":
            value = _parse_bool(value)
        elif key == "system_prompt_mode":
            if value not in _VALID_PROMPT_MODES:
                raise ValueError(
                    f"Invalid system_prompt_mode {value!r}. Valid: {', '.join(sorted(_VALID_PROMPT_MODES))}"
                )

        old_value = getattr(tc, key, None)
        setattr(tc, key, value)
        self._cache[thread_id] = tc
        self._persist(thread_id, tc)

        # Audit log.
        self._log_change(thread_id, key, old_value, value, source)

        return self.resolve(thread_id)

    def reset(self, thread_id: str, source: str = "api") -> None:
        """Remove all per-thread overrides (revert to defaults)."""
        old_tc = self._cache.pop(thread_id, None)
        database.delete_thread_config(thread_id)

        # Audit log.
        old_dict = asdict(old_tc) if old_tc else {}
        try:
            database.save_interaction_log(
                time.time(), "config_reset", thread_id, "",
                json.dumps({"old_config": old_dict, "source": source}),
                json.dumps(self.resolve_as_dict(thread_id)),
                "ok",
            )
        except Exception:
            logger.debug("Failed to log config reset", exc_info=True)

        logger.info("Thread config reset for %s (source=%s)", thread_id, source)

    # ── Persistence ──────────────────────────────────────────────────

    def _persist(self, thread_id: str, tc: ThreadConfig) -> None:
        """Write through to SQLite."""
        try:
            database.set_thread_config(thread_id, tc.to_json())
        except Exception:
            logger.exception("Failed to persist thread config for %s", thread_id)

    def _log_change(self, thread_id: str, key: str, old_value: Any,
                    new_value: Any, source: str) -> None:
        """Record the change in interaction_log for auditability."""
        try:
            database.save_interaction_log(
                time.time(), "config_change", thread_id, "",
                json.dumps({"key": key, "old": old_value, "new": new_value, "source": source}),
                json.dumps(self.resolve_as_dict(thread_id)),
                "ok",
            )
        except Exception:
            logger.debug("Failed to log config change", exc_info=True)

        logger.info(
            "Thread config %s.%s: %r → %r (source=%s)",
            thread_id, key, old_value, new_value, source,
        )
