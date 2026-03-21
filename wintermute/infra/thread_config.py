"""Per-thread configuration overrides.

Provides a three-layer config resolution model:
  per-thread override (SQLite)  →  global config default  →  hardcoded default

Each thread can independently override:
  - ``backend_name``              — pin main inference to a named backend
  - ``backend_overrides``         — per-role backend overrides (dict: role → backend name)
  - ``session_timeout_minutes``   — auto-session-renewal timeout (plumbing for #58)
  - ``sub_sessions_enabled``      — enable/disable sub-session spawning
  - ``system_prompt_mode``        — 'full' or 'minimal' prompt assembly
  - ``seed_language``             — language code for seed prompt on /new
  - ``nl_translation_enabled``    — enable/disable NL translation per-thread
  - ``memory_top_k``              — max memories retrieved per turn
  - ``memory_score_threshold``    — minimum similarity score for memory retrieval
  - ``compaction_keep_recent``    — messages kept untouched during compaction
  - ``max_inline_tool_rounds``    — inline tool call limit before CP enforcement

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
    backend_overrides: Optional[dict[str, str]] = None  # role → backend name
    session_timeout_minutes: Optional[int] = None
    sub_sessions_enabled: Optional[bool] = None
    system_prompt_mode: Optional[str] = None          # 'full' | 'minimal'
    seed_language: Optional[str] = None
    nl_translation_enabled: Optional[bool] = None
    memory_top_k: Optional[int] = None
    memory_score_threshold: Optional[float] = None
    compaction_keep_recent: Optional[int] = None
    max_inline_tool_rounds: Optional[int] = None

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
    backend_overrides: dict[str, str]    # role → backend name (empty = no overrides)
    session_timeout_minutes: Optional[int]  # None means "no auto-renewal"
    sub_sessions_enabled: bool
    system_prompt_mode: str              # 'full' | 'minimal'
    seed_language: str
    nl_translation_enabled: bool
    memory_top_k: int
    memory_score_threshold: float
    compaction_keep_recent: int
    max_inline_tool_rounds: int


# Hardcoded fallback defaults (baseline when no per-thread override exists).
_HARDCODED_DEFAULTS = {
    "backend_name": None,
    "backend_overrides": {},
    "session_timeout_minutes": None,
    "sub_sessions_enabled": True,
    "system_prompt_mode": "full",
    "seed_language": "en",
    "nl_translation_enabled": False,
    "memory_top_k": 10,
    "memory_score_threshold": 0.3,
    "compaction_keep_recent": 10,
    "max_inline_tool_rounds": 3,
}

# Valid values for system_prompt_mode.
_VALID_PROMPT_MODES = {"full", "minimal"}

# Valid role names for backend_overrides (session-scoped roles only).
_VALID_BACKEND_OVERRIDE_ROLES = {
    "main", "compaction", "sub_sessions", "convergence_protocol", "nl_translation",
}

# Valid 2-char language codes for seed_language.
_VALID_SEED_LANGUAGES = {"en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ko"}


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


def _parse_float(value: Any) -> float:
    """Parse a required float."""
    return float(value)


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

    def __init__(self, available_backends: list[str],
                 global_defaults: dict[str, Any] | None = None) -> None:
        self._available_backends = list(available_backends)
        self._global_defaults: dict[str, Any] = dict(global_defaults or {})
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
        """Merge per-thread overrides → global defaults → hardcoded defaults."""
        tc = self.get(thread_id)

        def _pick(key: str):
            val = getattr(tc, key, None)
            if val is not None:
                return val
            global_val = self._global_defaults.get(key)
            if global_val is not None:
                return global_val
            return _HARDCODED_DEFAULTS[key]

        sub_enabled = _pick("sub_sessions_enabled")

        # Normalize seed_language so that both per-thread overrides and
        # global defaults are treated consistently (e.g. "EN" → "en").
        raw_seed_language = _pick("seed_language") or "en"
        if isinstance(raw_seed_language, str):
            normalized_seed_language = raw_seed_language.strip().lower() or "en"
        else:
            normalized_seed_language = "en"

        # Merge backend_overrides: per-thread dict wins over global dict.
        raw_overrides = {}
        global_overrides = self._global_defaults.get("backend_overrides")
        if isinstance(global_overrides, dict):
            raw_overrides.update(global_overrides)
        if isinstance(tc.backend_overrides, dict):
            if tc.backend_overrides:
                raw_overrides.update(tc.backend_overrides)
        elif tc.backend_overrides is not None:
            logger.warning(
                "Ignoring invalid backend_overrides: expected dict, got %s",
                type(tc.backend_overrides).__name__,
            )
        # If backend_overrides["main"] is set, it takes precedence over backend_name.
        resolved_backend_name = raw_overrides.get("main") or _pick("backend_name")

        return ResolvedThreadConfig(
            backend_name=resolved_backend_name,
            backend_overrides=raw_overrides,
            session_timeout_minutes=_pick("session_timeout_minutes"),
            sub_sessions_enabled=bool(sub_enabled) if sub_enabled is not None else True,
            system_prompt_mode=_pick("system_prompt_mode") or "full",
            seed_language=normalized_seed_language,
            nl_translation_enabled=bool(_pick("nl_translation_enabled")),
            memory_top_k=int(_pick("memory_top_k")),
            memory_score_threshold=float(_pick("memory_score_threshold")),
            compaction_keep_recent=int(_pick("compaction_keep_recent")),
            max_inline_tool_rounds=int(_pick("max_inline_tool_rounds")),
        )

    def resolve_as_dict(self, thread_id: str) -> dict:
        """Return resolved config as a plain dict with source annotations."""
        tc = self.get(thread_id)
        resolved = self.resolve(thread_id)
        result = {}
        for f in fields(ResolvedThreadConfig):
            val = getattr(resolved, f.name)
            override = getattr(tc, f.name, None)
            if f.name == "backend_overrides":
                # Emit per-role source annotations for each override role.
                for role in sorted(_VALID_BACKEND_OVERRIDE_ROLES):
                    dotted = f"backend_overrides.{role}"
                    role_val = val.get(role) if isinstance(val, dict) else None
                    tc_overrides = tc.backend_overrides if isinstance(tc.backend_overrides, dict) else {}
                    _raw_global = self._global_defaults.get("backend_overrides")
                    global_overrides = _raw_global if isinstance(_raw_global, dict) else {}
                    if role in tc_overrides:
                        src = "override"
                    elif role in global_overrides:
                        src = "global"
                    else:
                        src = "default"
                    result[dotted] = {"value": role_val, "source": src}
                continue
            if override is not None:
                source = "override"
            elif f.name in self._global_defaults and self._global_defaults[f.name] is not None:
                source = "global"
            else:
                source = "default"
            result[f.name] = {"value": val, "source": source}
        return result

    def _coerce_global_default(self, key: str, value: Any) -> Any:
        """Validate/coerce a single global default value based on hardcoded defaults.

        This keeps _global_defaults consistent with the types expected by resolve().
        """
        # If we don't have a hardcoded baseline, we can't infer a type reliably.
        if key not in _HARDCODED_DEFAULTS:
            return value

        baseline = _HARDCODED_DEFAULTS[key]
        # If the baseline is None, accept the value as-is.
        if baseline is None:
            return value

        target_type = type(baseline)
        try:
            if target_type is bool:
                # Normalize truthy/falsy into a proper bool.
                return bool(value)
            if target_type is int:
                return int(value)
            if target_type is float:
                return float(value)
            if target_type is str:
                return str(value)
            # Fallback: leave value unchanged for unsupported types.
            return value
        except (TypeError, ValueError):
            logger.warning(
                "Invalid global default for %s: %r (expected %s) — ignoring",
                key,
                value,
                target_type.__name__,
            )
            # Preserve any existing value or fall back to the hardcoded baseline.
            return self._global_defaults.get(key, baseline)

    def update_global_defaults(self, defaults: dict[str, Any]) -> None:
        """Merge additional global defaults (e.g. from tuning config).

        Values are validated/coerced to the types implied by _HARDCODED_DEFAULTS
        to avoid type errors later in resolve().
        """
        coerced: dict[str, Any] = {}
        for key, value in defaults.items():
            coerced[key] = self._coerce_global_default(key, value)
        self._global_defaults.update(coerced)

    def get_available_backends(self) -> list[str]:
        """Return the list of configured backend names."""
        return list(self._available_backends)

    def get_all_overrides(self) -> dict[str, dict]:
        """Return all thread_ids that have custom config (for SSE/debug)."""
        result = {}
        for tid, tc in self._cache.items():
            # Only include threads that actually have at least one override.
            d = asdict(tc)
            has_override = any(
                (v is not None and v != {}) if isinstance(v, (dict, type(None))) else v is not None
                for v in d.values()
            )
            if has_override:
                result[tid] = self.resolve_as_dict(tid)
        return result

    # ── Write ────────────────────────────────────────────────────────

    def set(self, thread_id: str, key: str, value: Any,
            source: str = "api") -> ResolvedThreadConfig:
        """Validate and set a single config key for a thread.

        Supports dotted keys for backend_overrides (e.g. ``backend_overrides.compaction``).

        Returns the updated resolved config.
        Raises ``ValueError`` on invalid key or value.
        """
        tc = self._cache.get(thread_id, ThreadConfig())

        # Handle dotted backend_overrides keys (e.g. "backend_overrides.compaction").
        if key.startswith("backend_overrides."):
            role = key.split(".", 1)[1]
            if role not in _VALID_BACKEND_OVERRIDE_ROLES:
                raise ValueError(
                    f"Unknown backend override role {role!r}. "
                    f"Valid: {', '.join(sorted(_VALID_BACKEND_OVERRIDE_ROLES))}"
                )
            overrides = dict(tc.backend_overrides or {})
            is_clear = value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", ""))
            old_value = overrides.get(role)
            if is_clear:
                overrides.pop(role, None)
            else:
                if value not in self._available_backends:
                    raise ValueError(
                        f"Unknown backend {value!r}. Available: {', '.join(self._available_backends)}"
                    )
                overrides[role] = value
            tc.backend_overrides = overrides or None
            self._cache[thread_id] = tc
            self._persist(thread_id, tc)
            self._log_change(thread_id, key, old_value, value if not is_clear else None, source)
            return self.resolve(thread_id)

        # Validate key.
        known = {f.name for f in fields(ThreadConfig)}
        if key not in known:
            # Also accept "backend_overrides" as a whole-dict key.
            valid_keys = ", ".join(sorted(known))
            raise ValueError(f"Unknown config key {key!r}. Valid keys: {valid_keys}")

        # Treat null / "null" / "" as "clear override" for any key.
        if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "")):
            value = None
        elif key == "backend_name":
            if value not in self._available_backends:
                raise ValueError(
                    f"Unknown backend {value!r}. Available: {', '.join(self._available_backends)}"
                )
        elif key == "backend_overrides":
            # Accept a dict or JSON string mapping role → backend name.
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError("backend_overrides must be a JSON dict (role → backend name)")
            if not isinstance(value, dict):
                raise ValueError("backend_overrides must be a dict (role → backend name)")
            for role, bname in value.items():
                if role not in _VALID_BACKEND_OVERRIDE_ROLES:
                    raise ValueError(
                        f"Unknown role {role!r} in backend_overrides. "
                        f"Valid: {', '.join(sorted(_VALID_BACKEND_OVERRIDE_ROLES))}"
                    )
                if bname is not None and bname not in self._available_backends:
                    raise ValueError(
                        f"Unknown backend {bname!r} for role {role!r}. "
                        f"Available: {', '.join(self._available_backends)}"
                    )
            # Strip None values.
            value = {r: b for r, b in value.items() if b is not None} or None
        elif key == "session_timeout_minutes":
            value = int(value)
            if value < 1:
                raise ValueError("session_timeout_minutes must be >= 1 (or null to clear override)")
        elif key == "sub_sessions_enabled":
            value = _parse_bool(value)
        elif key == "system_prompt_mode":
            if value not in _VALID_PROMPT_MODES:
                raise ValueError(
                    f"Invalid system_prompt_mode {value!r}. Valid: {', '.join(sorted(_VALID_PROMPT_MODES))}"
                )
        elif key == "seed_language":
            value = str(value).lower().strip()
            if value not in _VALID_SEED_LANGUAGES:
                allowed = ", ".join(sorted(_VALID_SEED_LANGUAGES))
                raise ValueError(f"Invalid seed_language {value!r}. Allowed: {allowed}")
        elif key == "nl_translation_enabled":
            value = _parse_bool(value)
        elif key == "memory_top_k":
            value = int(value)
            if value < 1:
                raise ValueError("memory_top_k must be >= 1 (or null to use default)")
        elif key == "memory_score_threshold":
            value = _parse_float(value)
            if not (0.0 <= value <= 1.0):
                raise ValueError("memory_score_threshold must be between 0.0 and 1.0")
        elif key == "compaction_keep_recent":
            value = int(value)
            if value < 1:
                raise ValueError("compaction_keep_recent must be >= 1 (or null to use default)")
        elif key == "max_inline_tool_rounds":
            value = int(value)
            if value < 0:
                raise ValueError("max_inline_tool_rounds must be >= 0 (or null to use default)")

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
