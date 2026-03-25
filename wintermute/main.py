"""
Wintermute - Multi-Thread AI Assistant
Entry point and orchestration.

Matrix, Signal, and web are all optional; at least one must be enabled.

Startup sequence:
  1. Load config.yaml
  2. Configure logging
  3. Initialise SQLite databases
  4. Ensure data/ files exist
  5. Restore APScheduler jobs (and execute missed tasks)
  6. Build shared broadcast function (Matrix + Signal + web)
  7. Start LLM inference task
  8. Start web interface task (if enabled)
  9. Start Matrix task (if configured)
  10. Start Signal task (if configured)
  11. Await shutdown signals
"""

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader
from wintermute.infra.paths import DATA_DIR
from wintermute.infra.event_bus import EventBus
from wintermute.infra.thread_config import ThreadConfigManager
from wintermute import tools as tool_module
from wintermute.core.tool_deps import ToolDeps
from wintermute.core.llm_thread import LLMThread
from wintermute.core.types import BackendPool, MultiProviderConfig, ProviderConfig
from wintermute.interfaces.matrix_thread import MatrixConfig, MatrixThread
from wintermute.interfaces.signal_thread import SignalConfig, SignalThread
from wintermute.workers.dreaming import DreamingConfig, DreamingLoop
from wintermute.workers.memory_harvest import MemoryHarvestConfig, MemoryHarvestLoop
from wintermute.workers.reflection import ReflectionConfig, ReflectionLoop
from wintermute.workers.self_model import SelfModelConfig, SelfModelProfiler
from wintermute.workers.scheduler_thread import TaskScheduler, SchedulerConfig
from wintermute.core.sub_session import SubSessionManager
from wintermute.update_checker import UpdateCheckerConfig, UpdateCheckerLoop
from wintermute.interfaces.web_interface import WebInterface

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("config.yaml")
LOG_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_FILE) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found. Copy config.yaml.example and fill in your settings.")
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level_name = cfg.get("logging", {}).get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Main branch uses bracket-based format

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)

    fh = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "wintermute.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Data file bootstrapping
# ---------------------------------------------------------------------------

def ensure_data_dirs() -> None:
    """Ensure required data directories exist.

    Prompt files live in data/prompts/ and are validated separately
    by prompt_loader.validate_all().
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "skills").mkdir(exist_ok=True)
    (DATA_DIR / "scripts").mkdir(exist_ok=True)
    (DATA_DIR / "archive" / "memories").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

# Fallback exit code when os.execv() fails — systemd's Restart=on-failure
# will restart the service on any non-zero exit.
_EXECV_FALLBACK_EXIT_CODE = 42


class ShutdownCoordinator:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._loop = asyncio.get_event_loop()
        self.restart_requested: bool = False

    def request_shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._event.set)

    def request_restart(self) -> None:
        """Request a full process restart after graceful shutdown."""
        self.restart_requested = True
        self._loop.call_soon_threadsafe(self._event.set)

    async def wait(self) -> None:
        await self._event.wait()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_inference_backends(raw_list: list[dict]) -> dict[str, ProviderConfig]:
    """Parse the ``inference_backends`` list into a name→ProviderConfig map."""
    backends: dict[str, ProviderConfig] = {}
    for entry in raw_list:
        name = entry.get("name")
        if not name:
            print("ERROR: each inference_backends entry must have a 'name' field.")
            sys.exit(1)
        if name in backends:
            print(f"ERROR: duplicate inference_backends name: {name!r}")
            sys.exit(1)
        backends[name] = ProviderConfig(
            name=name,
            model=entry["model"],
            context_size=entry.get("context_size", 32768),
            max_tokens=entry.get("max_tokens", 4096),
            max_output_tokens=entry.get("max_output_tokens", 0),
            reasoning=entry.get("reasoning", False),
            provider=entry.get("provider", "openai"),
            api_key=entry.get("api_key", ""),
            base_url=entry.get("base_url", ""),
        )
    return backends


def _resolve_role(role_name: str, names: list[str],
                  backends: dict[str, ProviderConfig],
                  config_path: str = None) -> list[ProviderConfig]:
    """Resolve a list of backend names into ProviderConfig objects."""
    label = config_path or f"llm.{role_name}"
    result: list[ProviderConfig] = []
    for n in names:
        if n not in backends:
            print(f"ERROR: {label} references unknown backend {n!r}. "
                  f"Available: {', '.join(backends)}")
            sys.exit(1)
        result.append(backends[n])
    return result


def _build_multi_provider_config(cfg: dict) -> MultiProviderConfig:
    """Parse ``inference_backends`` + ``llm`` role mapping into a MultiProviderConfig."""
    raw_backends = cfg.get("inference_backends")
    if not raw_backends:
        print("ERROR: 'inference_backends' section is required in config.yaml. "
              "See config.yaml.example for the new format.")
        sys.exit(1)

    backends = _parse_inference_backends(raw_backends)
    llm_raw = cfg.get("llm", {})

    # Resolve each role.  Missing roles default to the first defined backend.
    first_name = next(iter(backends))
    default_list = [first_name]

    def _get_role(name: str, *, allow_empty: bool = False) -> list[ProviderConfig]:
        raw = llm_raw.get(name)
        if raw is None:
            if allow_empty:
                return []  # not configured → disabled
            return _resolve_role(name, default_list, backends)
        if isinstance(raw, list):
            if not raw and allow_empty:
                return []  # explicitly disabled
            return _resolve_role(name, raw, backends)
        print(f"ERROR: llm.{name} must be a list of backend names (got {type(raw).__name__})")
        sys.exit(1)

    def _get_role_with_fallback(name: str, fallback: list[ProviderConfig]) -> list[ProviderConfig]:
        """Resolve a role, falling back to *fallback* when omitted or set to []."""
        raw = llm_raw.get(name)
        if raw is None or (isinstance(raw, list) and not raw):
            return list(fallback)
        if isinstance(raw, list):
            return _resolve_role(name, raw, backends)
        print(f"ERROR: llm.{name} must be a list of backend names (got {type(raw).__name__})")
        sys.exit(1)

    # -- Convergence Protocol backends --
    cp_configs = _get_role("convergence_protocol", allow_empty=True)

    # -- NL Translation backends --
    nl_raw = cfg.get("nl_translation", {}) or {}
    nl_enabled = nl_raw.get("enabled", False)
    if nl_enabled:
        nl_backends_raw = nl_raw.get("backends")
        if nl_backends_raw is None:
            # Default to convergence_protocol backends if omitted.
            nl_configs = list(cp_configs) if cp_configs else _resolve_role(
                "nl_translation", default_list, backends,
                config_path="nl_translation.backends",
            )
        elif isinstance(nl_backends_raw, list) and nl_backends_raw:
            nl_configs = _resolve_role("nl_translation", nl_backends_raw, backends,
                                       config_path="nl_translation.backends")
        else:
            nl_configs = []
    else:
        nl_configs = []

    # -- Memory Harvest backends --
    # Falls back to sub_sessions backends when omitted (not to first backend).
    sub_sessions_configs = _get_role("sub_sessions")
    mh_configs = _get_role_with_fallback("memory_harvest", sub_sessions_configs)

    # -- Reflection backends --
    # Falls back to compaction backends when omitted.
    compaction_configs = _get_role("compaction")
    refl_configs = _get_role_with_fallback("reflection", compaction_configs)

    return MultiProviderConfig(
        main=_get_role("base"),
        compaction=compaction_configs,
        sub_sessions=sub_sessions_configs,
        dreaming=_get_role("dreaming"),
        convergence_protocol=cp_configs,
        memory_harvest=mh_configs,
        nl_translation=nl_configs,
        reflection=refl_configs,
    )


def _make_client_for_config(cfg: ProviderConfig, cache: dict) -> Any:
    """Create or reuse a client for the given ProviderConfig."""
    if cfg.provider == "gemini-cli":
        key = ("gemini-cli",)
        if key not in cache:
            from wintermute.backends import gemini_auth, gemini_client
            creds = gemini_auth.load_credentials()
            if not creds:
                logger.info("No Gemini credentials found — running interactive setup")
                creds = gemini_auth.setup()
            cache[key] = gemini_client.GeminiCloudClient(creds)
        return cache[key]
    elif cfg.provider == "kimi-code":
        key = ("kimi-code",)
        if key not in cache:
            from wintermute.backends import kimi_auth, kimi_client
            creds = kimi_auth.load_credentials()
            # Client is created even without creds — auto-auth runs after
            # interfaces are up, or the user can run /kimi-auth manually.
            cache[key] = kimi_client.KimiCodeClient(creds)
        return cache[key]
    elif cfg.provider == "anthropic":
        key = ("anthropic", cfg.api_key)
        if key not in cache:
            from wintermute.backends.anthropic_client import AnthropicClient
            cache[key] = AnthropicClient(api_key=cfg.api_key)
        return cache[key]
    else:
        key = (cfg.base_url, cfg.api_key)
        if key not in cache:
            from wintermute.backends.openai_wrapper import OpenAIBackend
            cache[key] = OpenAIBackend(api_key=cfg.api_key, base_url=cfg.base_url)
        return cache[key]


def _build_pool(configs: list[ProviderConfig], cache: dict) -> BackendPool:
    """Build a BackendPool from a list of ProviderConfigs, creating clients as needed."""
    backends = []
    for cfg in configs:
        client = _make_client_for_config(cfg, cache)
        backends.append((cfg, client))
    return BackendPool(backends)


class _LazyBackendPoolDict(dict):
    """Dict that builds BackendPools lazily on first access.

    Backends assigned to an llm role are materialised eagerly at startup.
    Other backends (defined in ``inference_backends`` but not mapped to any
    role) are only built when a user selects them via per-thread config —
    avoiding e.g. kimi-code OAuth prompts at startup when the backend is
    merely *defined* but unused.
    """

    def __init__(self, backends_by_name: dict[str, ProviderConfig],
                 client_cache: dict) -> None:
        super().__init__()
        self._available = backends_by_name
        self._client_cache = client_cache

    def __contains__(self, key: object) -> bool:
        # Report *all* defined backends as available so per-thread
        # overrides can reference them.
        return key in self._available or super().__contains__(key)

    def __getitem__(self, key: str) -> BackendPool:
        try:
            return super().__getitem__(key)
        except KeyError:
            if key in self._available:
                pool = _build_pool([self._available[key]], self._client_cache)
                self[key] = pool
                return pool
            raise


async def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    logger = logging.getLogger("main")
    logger.info("Wintermute starting up")

    ensure_data_dirs()
    prompt_loader.validate_all()
    database.init_db()

    # Initialize vector memory store (before pool construction).
    from wintermute.infra import memory_store
    memory_store.init(cfg.get("memory", {}))

    # Initialize skill store (shares embedding + backend config from memory section).
    from wintermute.infra import skill_store
    memory_cfg = cfg.get("memory", {}) or {}
    skills_cfg = (cfg.get("skills", {}) or {}).copy()
    # Inherit memory.backend as skill backend default when not explicitly set.
    if not skills_cfg.get("backend"):
        mem_backend = memory_cfg.get("backend", "local_vector")
        skills_cfg["backend"] = mem_backend
    # Expose memory Qdrant config for inheritance by the skill Qdrant backend.
    if memory_cfg.get("backend") == "qdrant" or memory_cfg.get("qdrant"):
        skills_cfg.setdefault("_memory_qdrant", memory_cfg.get("qdrant", {}))
    skill_store.init(
        skills_cfg,
        memory_cfg.get("embeddings", {}),
    )

    # Set timezone for prompt assembler (used to inject current datetime).
    configured_tz = cfg.get("scheduler", {}).get("timezone", "UTC")
    prompt_assembler.set_timezone(configured_tz)

    # Apply component size limits from config (context.component_size_limits).
    csl = cfg.get("context", {}).get("component_size_limits", {})
    prompt_assembler.set_component_limits(
        memories=csl.get("memories", 10_000),
        tasks=csl.get("tasks", 5_000),
        skills=csl.get("skills_total", 2_000),
    )

    # Tool profiles for sub-session spawning (passed via ToolDeps).
    _tool_profiles = cfg.get("tool_profiles", {}) or {}

    multi_cfg = _build_multi_provider_config(cfg)

    # Build BackendPools (clients are created/shared internally).
    client_cache: dict = {}
    main_pool = _build_pool(multi_cfg.main, client_cache)
    compaction_pool = _build_pool(multi_cfg.compaction, client_cache)
    sub_sessions_pool = _build_pool(multi_cfg.sub_sessions, client_cache)
    dreaming_pool = _build_pool(multi_cfg.dreaming, client_cache)
    convergence_protocol_pool = _build_pool(multi_cfg.convergence_protocol, client_cache)
    memory_harvest_pool = _build_pool(multi_cfg.memory_harvest, client_cache)
    nl_translation_pool = _build_pool(multi_cfg.nl_translation, client_cache)
    reflection_pool = _build_pool(multi_cfg.reflection, client_cache)

    # Build per-backend pools for per-thread overrides.
    # Pools for backends already used in a role are built eagerly (clients
    # already exist in client_cache). Other backends are built lazily on
    # first access so that e.g. kimi-code auth is not triggered at startup
    # when the backend is defined but not assigned to any llm role.
    backends_by_name = _parse_inference_backends(cfg.get("inference_backends", []))
    _used_backend_names: set[str] = set()
    for cfgs in (multi_cfg.main, multi_cfg.compaction, multi_cfg.sub_sessions,
                 multi_cfg.dreaming, multi_cfg.convergence_protocol,
                 multi_cfg.memory_harvest, multi_cfg.nl_translation,
                 multi_cfg.reflection):
        for pc in cfgs:
            _used_backend_names.add(pc.name)
    backend_pools_by_name = _LazyBackendPoolDict(backends_by_name, client_cache)
    for bname in _used_backend_names:
        if bname in backends_by_name:
            # Eagerly materialise pools for role-assigned backends.
            _ = backend_pools_by_name[bname]

    # Per-thread configuration manager.
    global_sub_sessions = cfg.get("sub_sessions_enabled", True)
    # Normalise global seed language to avoid bypassing per-thread validation.
    _raw_seed_language = (cfg.get("seed", {}) or {}).get("language", "en")
    if _raw_seed_language is None:
        _raw_seed_language = "en"
    normalized_seed_language = str(_raw_seed_language).strip().lower()
    if "-" in normalized_seed_language:
        # Strip region tags like "en-US" -> "en"
        normalized_seed_language = normalized_seed_language.split("-", 1)[0].strip()
    if not normalized_seed_language:
        normalized_seed_language = "en"
    thread_config_mgr = ThreadConfigManager(
        available_backends=list(backends_by_name.keys()),
        global_defaults={
            "sub_sessions_enabled": global_sub_sessions,
            "seed_language": normalized_seed_language,
            "nl_translation_enabled": (cfg.get("nl_translation") or {}).get("enabled", False),
            "memory_top_k": (cfg.get("memory") or {}).get("top_k", 10),
            "memory_score_threshold": (cfg.get("memory") or {}).get("score_threshold", 0.3),
        },
    )
    if not global_sub_sessions:
        logger.info("sub_sessions_enabled=false (lite mode) — worker_delegation disabled globally")

    # Parse NL translation config.
    nl_raw = cfg.get("nl_translation", {}) or {}
    nl_translation_config = {
        "enabled": nl_raw.get("enabled", False) and nl_translation_pool.enabled,
        "tools": set(nl_raw.get("tools", ["task", "worker_delegation", "skill"])),
    }
    if nl_translation_config["enabled"]:
        prompt_loader.validate_nl_translation()
        logger.info("NL Translation enabled (tools=%s, model=%s)",
                     nl_translation_config["tools"], nl_translation_pool.primary.model)

    scheduler_raw = cfg.get("scheduler", {}) or {}
    dreaming_raw = cfg.get("dreaming", {})
    dreaming_cfg = DreamingConfig(
        hour=dreaming_raw.get("hour", 1),
        minute=dreaming_raw.get("minute", 0),
        timezone=scheduler_raw.get("timezone", "UTC"),
    )
    memory_dreaming_raw = cfg.get("memory", {}).get("dreaming", {})
    scheduler_cfg = SchedulerConfig(
        timezone=scheduler_raw.get("timezone", "UTC"),
        prediction_proactive_scheduling=memory_dreaming_raw.get("prediction_proactive_scheduling", True),
        prediction_proactive_cooldown_hours=memory_dreaming_raw.get("prediction_proactive_cooldown_hours", 4),
        proactive_target_thread_id=memory_dreaming_raw.get("proactive_target_thread_id", "default"),
        task_completed_retention_days=scheduler_raw.get("task_completed_retention_days", 30),
        task_maintenance_interval_hours=scheduler_raw.get("task_maintenance_interval_hours", 6),
        auto_complete_stale_once_tasks=scheduler_raw.get("auto_complete_stale_once_tasks", True),
    )
    # --- Tuning constants (optional overrides from config) ---
    _tuning_raw = cfg.get("tuning")
    if _tuning_raw is None:
        tuning: dict = {}
    elif not isinstance(_tuning_raw, dict):
        logger.warning("tuning: expected a mapping, got %r; ignoring tuning section", type(_tuning_raw).__name__)
        tuning = {}
    else:
        tuning = _tuning_raw

    def _tuning_int(key: str, default: int, minimum: int = 0) -> int:
        """Validate and coerce a tuning config value to a bounded int."""
        raw = tuning.get(key, default)
        try:
            val = int(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid tuning.%s=%r; using default %d", key, raw, default)
            return default
        if val < minimum:
            logger.warning("tuning.%s=%d below minimum %d; clamping", key, val, minimum)
            return minimum
        return val

    compaction_keep_recent = _tuning_int("compaction_keep_recent", 10, minimum=1)
    max_continuation_depth = _tuning_int("max_continuation_depth", 3, minimum=0)
    max_nesting_depth = _tuning_int("max_nesting_depth", 2, minimum=0)
    max_blob_chars = _tuning_int("max_blob_chars", 60_000, minimum=1)
    max_completed_workflows = _tuning_int("max_completed_workflows", 50, minimum=1)
    max_inline_tool_rounds = _tuning_int("max_inline_tool_rounds", 3, minimum=0)

    # Feed tuning-based defaults into the per-thread config manager.
    thread_config_mgr.update_global_defaults({
        "compaction_keep_recent": compaction_keep_recent,
        "max_inline_tool_rounds": max_inline_tool_rounds,
    })

    # Push the inline tool limit into the Convergence Protocol module.
    from wintermute.core.convergence_protocol import set_max_inline_tool_rounds
    set_max_inline_tool_rounds(max_inline_tool_rounds)

    # Extract config secrets and enable credential redaction.
    from wintermute.core.convergence_protocol import extract_config_secrets, set_redaction_secrets
    _secrets = extract_config_secrets(cfg)
    set_redaction_secrets(_secrets)
    if _secrets:
        logger.info("Credential redaction: %d secret values loaded", len(_secrets))

    harvest_cfg_raw = cfg.get("memory_harvest", {})
    harvest_config = MemoryHarvestConfig(
        enabled=harvest_cfg_raw.get("enabled", True),
        message_threshold=harvest_cfg_raw.get("message_threshold", 20),
        inactivity_timeout_minutes=harvest_cfg_raw.get("inactivity_timeout_minutes", 15),
        max_message_chars=harvest_cfg_raw.get("max_message_chars", 2000),
        max_blob_chars=max_blob_chars,
    )

    # SearXNG URL from config.
    searxng_url = (cfg.get("search") or {}).get("searxng_url", "")

    # --- Optional interfaces ---
    matrix_cfg_raw: Optional[dict] = cfg.get("matrix")
    web_cfg: dict = cfg.get("web", {"enabled": True, "host": "127.0.0.1", "port": 8080})

    matrix_enabled = bool(
        matrix_cfg_raw
        and matrix_cfg_raw.get("homeserver")
        and (matrix_cfg_raw.get("access_token") or matrix_cfg_raw.get("password"))
    )
    signal_cfg_raw: Optional[dict] = cfg.get("signal")
    signal_enabled = bool(
        signal_cfg_raw
        and signal_cfg_raw.get("enabled")
        and signal_cfg_raw.get("phone_number")
    )

    web_enabled = web_cfg.get("enabled", True)

    if not matrix_enabled and not web_enabled and not signal_enabled:
        logger.error("No interface enabled (Matrix, Signal, or web) - nothing to do. Exiting.")
        sys.exit(1)

    shutdown = ShutdownCoordinator()

    # Event bus — shared infrastructure for all components.
    event_bus = EventBus()

    # ------------------------------------------------------------------
    # Component construction — ordered so every dependency exists before
    # its consumer, eliminating post-construction injection.  The one
    # genuine LLMThread ↔ SubSessionManager cycle is broken with a lazy
    # getter instead of a post-construction inject method.
    # ------------------------------------------------------------------

    # 1. Broadcast closure — needs forward references to matrix / web_iface
    #    which are filled in below (the closure is only _called_ after
    #    asyncio tasks start, by which point everything is wired).
    _matrix_ref: list[Optional[MatrixThread]] = [None]
    _web_ref: list[Optional[WebInterface]] = [None]
    _signal_ref: list[Optional[SignalThread]] = [None]

    async def broadcast(text: str, thread_id: Optional[str] = None, *,
                        reasoning: Optional[str] = None,
                        is_proactive: bool = False) -> None:
        mx = _matrix_ref[0]
        wi = _web_ref[0]
        si = _signal_ref[0]
        if si and thread_id and thread_id.startswith("sig_"):
            await si.send_message(text, thread_id)
        elif mx and thread_id and not thread_id.startswith("web_") and not thread_id.startswith("sig_") and thread_id != "default":
            await mx.send_message(text, thread_id)
        if wi and thread_id:
            await wi.broadcast(text, thread_id, reasoning=reasoning)
        # Proactive results: deliver to all rooms that have opted in via /proactive on.
        # This fires only when thread_id == "default" (proactive sessions have no specific
        # target room), ensuring regular messages are never duplicated.
        if is_proactive and thread_id == "default":
            opted_in = thread_config_mgr.get_proactive_thread_ids()
            if opted_in:
                async def _send_to_room(tid: str) -> None:
                    try:
                        if si and tid.startswith("sig_"):
                            await si.send_message(text, tid)
                        elif mx and not tid.startswith("web_") and not tid.startswith("sig_"):
                            await mx.send_message(text, tid)
                    except Exception:  # noqa: BLE001
                        logger.warning("Failed to deliver proactive message to %s", tid)

                await asyncio.gather(*(_send_to_room(tid) for tid in opted_in))
                if wi:
                    await wi.broadcast(text, thread_id, reasoning=reasoning)

    # 2. LLMThread — uses a lazy getter for SubSessionManager (breaks cycle).
    seed_language = cfg.get("seed", {}).get("language", "en") if cfg.get("seed") else "en"
    cp_validators = (cfg.get("convergence_protocol", {}) or {}).get("validators")

    # Lazy holder: SubSessionManager is appended after construction.
    _ssm_holder: list[SubSessionManager] = []

    # ToolDeps is constructed early with config-only fields; typed object
    # references (sub_session_manager, task_scheduler, self_model_profiler)
    # are set once after each owner is built.
    tool_deps = ToolDeps(
        max_nesting_depth=max_nesting_depth,
        searxng_url=searxng_url,
        tool_profiles=_tool_profiles,
        event_bus=event_bus,
        memory_pool=sub_sessions_pool,
        event_loop=asyncio.get_running_loop(),
        shutdown_coordinator=shutdown,
    )

    llm = LLMThread(main_pool=main_pool, compaction_pool=compaction_pool,
                    convergence_protocol_pool=convergence_protocol_pool, broadcast_fn=broadcast,
                    sub_session_getter=lambda: _ssm_holder[0] if _ssm_holder else None,
                    convergence_protocol_validators=cp_validators,
                    nl_translation_pool=nl_translation_pool,
                    nl_translation_config=nl_translation_config,
                    seed_language=seed_language,
                    event_bus=event_bus,
                    thread_config_manager=thread_config_mgr,
                    backend_pools_by_name=backend_pools_by_name,
                    compaction_keep_recent=compaction_keep_recent,
                    tool_deps=tool_deps)

    # 3. SubSessionManager — needs llm.enqueue_system_event (no cycle: LLM
    #    already exists).  Fills the lazy holder so LLM can reach it later.
    sub_sessions = SubSessionManager(
        pool=sub_sessions_pool,
        enqueue_system_event=llm.enqueue_system_event,
        convergence_protocol_pool=convergence_protocol_pool,
        convergence_protocol_validators=cp_validators,
        nl_translation_pool=nl_translation_pool,
        nl_translation_config=nl_translation_config,
        event_bus=event_bus,
        max_continuation_depth=max_continuation_depth,
        max_completed_workflows=max_completed_workflows,
        tool_deps=tool_deps,
    )
    _ssm_holder.append(sub_sessions)
    tool_deps.sub_session_manager = sub_sessions

    # 4. TaskScheduler (start() deferred until broadcast refs are filled — see step 14)
    scheduler = TaskScheduler(
        config=scheduler_cfg,
        broadcast_fn=broadcast,
        llm_enqueue_fn=llm.enqueue_system_event,
        sub_session_manager=sub_sessions,
        event_bus=event_bus,
    )
    tool_deps.task_scheduler = scheduler
    # Wire session manager to scheduler for timeout enforcement.
    scheduler.set_session_manager(llm.session_manager)

    # 5. Memory harvest loop
    harvest_loop: Optional[MemoryHarvestLoop] = None
    if harvest_config.enabled:
        harvest_loop = MemoryHarvestLoop(
            config=harvest_config,
            sub_session_manager=sub_sessions,
            pool=memory_harvest_pool,
            event_bus=event_bus,
        )
    else:
        logger.info("Memory harvest disabled by config")

    # 6. Self-model profiler
    sm_raw = cfg.get("self_model", {}) or {}
    sm_cfg = SelfModelConfig(
        enabled=sm_raw.get("enabled", True),
        yaml_path=sm_raw.get("yaml_path", "data/self_model.yaml"),
        sub_session_timeout_range=tuple(sm_raw.get("sub_session_timeout_range", [120, 900])),
        memory_harvest_threshold_range=tuple(sm_raw.get("memory_harvest_threshold_range", [5, 50])),
        summary_max_chars=sm_raw.get("summary_max_chars", 300),
    )
    self_model = None
    if sm_cfg.enabled:
        try:
            self_model = SelfModelProfiler(
                config=sm_cfg,
                pool=reflection_pool,
                event_bus=event_bus,
                sub_session_manager=sub_sessions,
                memory_harvest_loop=harvest_loop,
            )
            tool_deps.self_model_profiler = self_model
            from wintermute.infra.prompt_assembler import set_self_model_path
            set_self_model_path(sm_cfg.yaml_path)
            logger.info("Self-model profiler enabled")
        except Exception:
            logger.exception("Self-model profiler failed to initialise — disabling")
            self_model = None
    else:
        logger.info("Self-model profiler disabled by config")

    # Restore auto-tuned values from self-model YAML.
    if self_model:
        self_model.restore_tuned_params()

    # 7. Reflection loop (receives self_model via constructor now)
    reflection_raw = cfg.get("reflection", {}) or {}
    reflection_cfg = ReflectionConfig(
        enabled=reflection_raw.get("enabled", True),
        batch_threshold=reflection_raw.get("batch_threshold", 10),
        consecutive_failure_limit=reflection_raw.get("consecutive_failure_limit", 3),
        lookback_seconds=reflection_raw.get("lookback_seconds", 86400),
        min_result_length=reflection_raw.get("min_result_length", 50),
        main_turn_batch_threshold=reflection_raw.get("main_turn_batch_threshold", 15),
        synthesis_min_cluster_size=reflection_raw.get("synthesis_min_cluster_size", 3),
        synthesis_min_outcomes=reflection_raw.get("synthesis_min_outcomes", 20),
    )
    reflection_loop = ReflectionLoop(
        config=reflection_cfg,
        sub_session_manager=sub_sessions,
        pool=reflection_pool,
        event_bus=event_bus,
        self_model=self_model,
    )

    # 8. Dreaming loop
    dreaming_loop = DreamingLoop(
        config=dreaming_cfg,
        pool=dreaming_pool,
        event_bus=event_bus,
    )

    # 9. Update checker
    uc_raw = cfg.get("update_checker", {})
    uc_config = UpdateCheckerConfig(
        enabled=uc_raw.get("enabled", True),
        check_on_startup=uc_raw.get("check_on_startup", True),
        interval_hours=uc_raw.get("interval_hours", 24),
        remote_url=uc_raw.get("remote_url", ""),
    )
    update_checker: Optional[UpdateCheckerLoop] = None
    if uc_config.enabled:
        uc_rooms: list[str] = []
        if matrix_enabled and matrix_cfg_raw:
            uc_rooms = matrix_cfg_raw.get("allowed_rooms", []) or []
        update_checker = UpdateCheckerLoop(
            config=uc_config,
            broadcast_fn=broadcast,
            matrix_rooms=uc_rooms,
        )
    else:
        logger.info("Update checker disabled by config")

    # 10. Slash command handler (needs almost everything above)
    from wintermute.interfaces.slash_commands import SlashCommandHandler
    slash_handler = SlashCommandHandler(
        llm=llm,
        sub_sessions=sub_sessions,
        thread_config_manager=thread_config_mgr,
        dreaming_loop=dreaming_loop,
        memory_harvest=harvest_loop,
        scheduler=scheduler,
        reflection_loop=reflection_loop,
        self_model=self_model,
        update_checker=update_checker,
    )

    # 11. Whisper client (built once, passed to MatrixThread constructor)
    whisper_client = None
    whisper_model = ""
    whisper_language = ""
    whisper_raw = cfg.get("whisper", {}) or {}
    if whisper_raw.get("enabled") and (matrix_enabled or signal_enabled):
        from openai import AsyncOpenAI
        whisper_client = AsyncOpenAI(
            api_key=whisper_raw.get("api_key", ""),
            base_url=whisper_raw.get("base_url", ""),
            timeout=60.0,
        )
        whisper_model = whisper_raw.get("model", "whisper-1")
        whisper_language = whisper_raw.get("language", "") or ""
        logger.info("Whisper transcription enabled (model=%s)", whisper_model)
        import shutil as _shutil
        if not _shutil.which("ffmpeg"):
            logger.warning(
                "Whisper transcription is enabled but ffmpeg is not in PATH. "
                "Voice messages from Matrix (OGG/Opus) cannot be transcribed without ffmpeg. "
                "Install it with: sudo apt install ffmpeg  OR  sudo dnf install ffmpeg"
            )

    # 12. MatrixThread — all deps passed via constructor (no post-injection)
    matrix: Optional[MatrixThread] = None
    if matrix_enabled:
        matrix_cfg = MatrixConfig(
            homeserver=matrix_cfg_raw["homeserver"],
            user_id=matrix_cfg_raw["user_id"],
            access_token=matrix_cfg_raw.get("access_token", ""),
            device_id=matrix_cfg_raw.get("device_id", ""),
            password=matrix_cfg_raw.get("password", ""),
            allowed_users=matrix_cfg_raw.get("allowed_users", []),
            allowed_rooms=matrix_cfg_raw.get("allowed_rooms", []),
            group_mode=matrix_cfg_raw.get("group_mode", False),
        )
        matrix = MatrixThread(
            matrix_cfg, llm,
            kimi_client=client_cache.get(("kimi-code",)),
            whisper_client=whisper_client,
            whisper_model=whisper_model,
            whisper_language=whisper_language,
            slash_handler=slash_handler,
            event_bus=event_bus,
        )
    else:
        logger.info("Matrix not configured - skipping Matrix interface")

    # 12b. SignalThread
    signal_iface: Optional[SignalThread] = None
    if signal_enabled:
        sig_cfg = SignalConfig(
            phone_number=signal_cfg_raw["phone_number"],
            signal_cli_path=signal_cfg_raw.get("signal_cli_path", "signal-cli"),
            allowed_users=signal_cfg_raw.get("allowed_users", []),
            allowed_groups=signal_cfg_raw.get("allowed_groups", []),
            group_mode=signal_cfg_raw.get("group_mode", False),
            trust_new_keys=signal_cfg_raw.get("trust_new_keys", True),
            http_port=signal_cfg_raw.get("http_port", 8190),
        )
        signal_iface = SignalThread(
            sig_cfg, llm,
            whisper_client=whisper_client,
            whisper_model=whisper_model,
            whisper_language=whisper_language,
            slash_handler=slash_handler,
            event_bus=event_bus,
        )
    else:
        logger.info("Signal not configured - skipping Signal interface")

    # 13. WebInterface — all deps passed via constructor (no post-injection)
    web_iface: Optional[WebInterface] = None
    if web_enabled:
        web_iface = WebInterface(
            host=web_cfg.get("host", "127.0.0.1"),
            port=web_cfg.get("port", 8080),
            llm_thread=llm,
            sub_sessions=sub_sessions,
            scheduler=scheduler,
            matrix=matrix,
            main_pool=main_pool,
            multi_cfg=multi_cfg,
            thread_config_manager=thread_config_mgr,
            slash_handler=slash_handler,
        )

    # Fill the broadcast closure's forward references.
    _matrix_ref[0] = matrix
    _web_ref[0] = web_iface
    _signal_ref[0] = signal_iface

    # 14. Start the scheduler now that broadcast refs are filled, so
    #     _recover_missed() reminder broadcasts reach the interfaces.
    scheduler.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    tasks = [
        asyncio.create_task(llm.run(),              name="llm"),
        asyncio.create_task(dreaming_loop.run(),     name="dreaming"),
        asyncio.create_task(reflection_loop.run(),   name="reflection"),
    ]
    if harvest_loop:
        tasks.append(asyncio.create_task(harvest_loop.run(), name="memory_harvest"))
    if update_checker:
        tasks.append(asyncio.create_task(update_checker.run(), name="update_checker"))
    if matrix:
        tasks.append(asyncio.create_task(matrix.run(), name="matrix"))
    if signal_iface:
        tasks.append(asyncio.create_task(signal_iface.run(), name="signal"))
    if web_iface:
        tasks.append(asyncio.create_task(web_iface.run(), name="web"))

    interfaces = []
    if matrix_enabled:
        interfaces.append("Matrix")
    if signal_enabled:
        interfaces.append("Signal")
    if web_enabled:
        interfaces.append(f"web http://{web_cfg.get('host','127.0.0.1')}:{web_cfg.get('port',8080)}")
    # Feature branch: enhanced logging
    logger.info("✨ All components started successfully. Interfaces: %s", ", ".join(interfaces))

    # Auto-trigger Kimi-Code device auth if credentials are missing.
    kimi_client_instance = client_cache.get(("kimi-code",))
    if kimi_client_instance and not kimi_client_instance.authenticated:
        async def _kimi_auto_auth() -> None:
            from wintermute.backends import kimi_auth
            # Wait for Matrix client to be ready (connected + synced).
            if matrix:
                for _ in range(30):
                    if matrix._client is not None:
                        break
                    await asyncio.sleep(1)
                # Extra pause for first sync to complete.
                await asyncio.sleep(3)
            try:
                async def _broadcast_auth(msg: str) -> None:
                    sent = False
                    # Broadcast to allowed Matrix rooms only.
                    if matrix:
                        allowed = set(matrix._cfg.allowed_rooms) if matrix._cfg.allowed_rooms else await matrix.get_joined_rooms()
                        logger.debug("Kimi auto-auth: broadcasting to rooms=%s", allowed)
                        for room_id in allowed:
                            try:
                                await matrix.send_message(msg, room_id)
                                sent = True
                            except Exception as exc:
                                logger.debug("Kimi auth broadcast to %s failed: %s", room_id, exc)
                    # Broadcast to all connected web clients.
                    if web_iface:
                        for tid in list(web_iface.connected_thread_ids):
                            await web_iface.broadcast(msg, tid)
                            sent = True
                    if not sent:
                        logger.warning("Kimi auth (no interface connected): %s", msg)
                creds = await kimi_auth.run_device_flow(_broadcast_auth)
                kimi_client_instance.update_credentials(creds)
                logger.info("Kimi-Code auto-auth completed")
            except Exception:
                logger.exception("Kimi-Code auto-auth failed — use /kimi-auth to retry")
        tasks.append(asyncio.create_task(_kimi_auto_auth(), name="kimi-auth"))

    await shutdown.wait()
    logger.info("Shutdown requested - stopping components gracefully")

    if harvest_loop:
        harvest_loop.stop()
    if update_checker:
        update_checker.stop()
    dreaming_loop.stop()
    reflection_loop.stop()
    if matrix:
        matrix.stop()
    if signal_iface:
        signal_iface.stop()
    llm.stop()
    scheduler.stop()

    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Some tasks did not stop within 10 s — forcing exit")

    # Drain after tasks are cancelled so any commits queued during shutdown
    # cleanup are also flushed before the process exits.  Run in a thread so
    # we don't block the event loop; cap the wait at 30 s.
    from wintermute.infra import data_versioning
    try:
        await asyncio.wait_for(asyncio.to_thread(data_versioning.drain), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("data_versioning: drain timed out after 30 s — commits may be incomplete")

    database.close_all_connections()
    logger.info("Wintermute shutdown complete")

    if shutdown.restart_requested:
        logger.info("Restart requested — re-executing process")
        try:
            os.execv(sys.executable, [sys.executable, "-m", "wintermute.main"] + sys.argv[1:])
        except OSError:
            logger.exception("os.execv failed — exiting with code %d for systemd restart",
                             _EXECV_FALLBACK_EXIT_CODE)
            sys.exit(_EXECV_FALLBACK_EXIT_CODE)


if __name__ == "__main__":
    asyncio.run(main())


def run() -> None:
    """Entry point for the `wintermute` console script."""
    asyncio.run(main())
