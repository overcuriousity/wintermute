"""
Wintermute - Multi-Thread AI Assistant
Entry point and orchestration.

Both Matrix and the web interface are optional; at least one must be enabled.

Startup sequence:
  1. Load config.yaml
  2. Configure logging
  3. Initialise SQLite databases
  4. Ensure data/ files exist
  5. Restore APScheduler jobs (and execute missed reminders)
  6. Build shared broadcast function (Matrix + web)
  7. Start LLM inference task
  8. Start web interface task (if enabled)
  9. Start Matrix task (if configured)
  10. Start agenda review task
  11. Await shutdown signals
"""

import asyncio
import logging
import logging.handlers
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from wintermute import database
from wintermute import prompt_assembler
from wintermute import prompt_loader
from wintermute import tools as tool_module
from wintermute.agenda import AgendaLoop
from openai import AsyncOpenAI

from wintermute.llm_thread import BackendPool, LLMThread, MultiProviderConfig, ProviderConfig
from wintermute.matrix_thread import MatrixConfig, MatrixThread
from wintermute.dreaming import DreamingConfig, DreamingLoop
from wintermute.memory_harvest import MemoryHarvestConfig, MemoryHarvestLoop
from wintermute.scheduler_thread import ReminderScheduler, SchedulerConfig
from wintermute.sub_session import SubSessionManager
from wintermute.web_interface import WebInterface

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
        "%(asctime)s %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

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

DATA_DIR = Path("data")

def bootstrap_data_files() -> None:
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

class ShutdownCoordinator:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def request_shutdown(self) -> None:
        self._event.set()

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
                return _resolve_role(name, default_list, backends)
            return _resolve_role(name, default_list, backends)
        if isinstance(raw, list):
            if not raw and allow_empty:
                return []  # explicitly disabled
            return _resolve_role(name, raw, backends)
        print(f"ERROR: llm.{name} must be a list of backend names (got {type(raw).__name__})")
        sys.exit(1)

    # -- Turing Protocol backends --
    tp_raw = cfg.get("turing_protocol", {}) or {}
    tp_backends_raw = tp_raw.get("backends")
    if tp_backends_raw is None:
        # Omitted → default to base model backends
        tp_configs = _resolve_role("turing_protocol", default_list, backends,
                                   config_path="turing_protocol.backends")
    elif isinstance(tp_backends_raw, list):
        if not tp_backends_raw:
            tp_configs = []  # explicitly disabled
        else:
            tp_configs = _resolve_role("turing_protocol", tp_backends_raw, backends,
                                       config_path="turing_protocol.backends")
    else:
        print(f"ERROR: turing_protocol.backends must be a list (got {type(tp_backends_raw).__name__})")
        sys.exit(1)

    # -- NL Translation backends --
    nl_raw = cfg.get("nl_translation", {}) or {}
    nl_enabled = nl_raw.get("enabled", False)
    if nl_enabled:
        nl_backends_raw = nl_raw.get("backends")
        if nl_backends_raw is None:
            # Default to turing_protocol backends if omitted.
            nl_configs = list(tp_configs) if tp_configs else _resolve_role(
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

    return MultiProviderConfig(
        main=_get_role("base"),
        compaction=_get_role("compaction"),
        sub_sessions=_get_role("sub_sessions"),
        dreaming=_get_role("dreaming"),
        turing_protocol=tp_configs,
        nl_translation=nl_configs,
    )


def _make_client_for_config(cfg: ProviderConfig, cache: dict) -> Any:
    """Create or reuse a client for the given ProviderConfig."""
    if cfg.provider == "gemini-cli":
        key = ("gemini-cli",)
        if key not in cache:
            from wintermute import gemini_auth, gemini_client
            creds = gemini_auth.load_credentials()
            if not creds:
                logger.info("No Gemini credentials found — running interactive setup")
                creds = gemini_auth.setup()
            cache[key] = gemini_client.GeminiCloudClient(creds)
        return cache[key]
    elif cfg.provider == "kimi-code":
        key = ("kimi-code",)
        if key not in cache:
            from wintermute import kimi_auth, kimi_client
            creds = kimi_auth.load_credentials()
            # Client is created even without creds — auto-auth runs after
            # interfaces are up, or the user can run /kimi-auth manually.
            cache[key] = kimi_client.KimiCodeClient(creds)
        return cache[key]
    else:
        key = (cfg.base_url, cfg.api_key)
        if key not in cache:
            cache[key] = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        return cache[key]


def _build_pool(configs: list[ProviderConfig], cache: dict) -> BackendPool:
    """Build a BackendPool from a list of ProviderConfigs, creating clients as needed."""
    backends = []
    for cfg in configs:
        client = _make_client_for_config(cfg, cache)
        backends.append((cfg, client))
    return BackendPool(backends)


async def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    logger = logging.getLogger("main")
    logger.info("Wintermute starting up")

    bootstrap_data_files()
    prompt_loader.validate_all()
    database.init_db()

    # Set timezone for prompt assembler (used to inject current datetime).
    configured_tz = cfg.get("scheduler", {}).get("timezone", "UTC")
    prompt_assembler.set_timezone(configured_tz)

    multi_cfg = _build_multi_provider_config(cfg)

    # Build BackendPools (clients are created/shared internally).
    client_cache: dict = {}
    main_pool = _build_pool(multi_cfg.main, client_cache)
    compaction_pool = _build_pool(multi_cfg.compaction, client_cache)
    sub_sessions_pool = _build_pool(multi_cfg.sub_sessions, client_cache)
    dreaming_pool = _build_pool(multi_cfg.dreaming, client_cache)
    turing_protocol_pool = _build_pool(multi_cfg.turing_protocol, client_cache)
    nl_translation_pool = _build_pool(multi_cfg.nl_translation, client_cache)

    # Parse NL translation config.
    nl_raw = cfg.get("nl_translation", {}) or {}
    nl_translation_config = {
        "enabled": nl_raw.get("enabled", False) and nl_translation_pool.enabled,
        "tools": set(nl_raw.get("tools", ["set_reminder", "spawn_sub_session"])),
    }
    if nl_translation_config["enabled"]:
        prompt_loader.validate_nl_translation()
        logger.info("NL Translation enabled (tools=%s, model=%s)",
                     nl_translation_config["tools"], nl_translation_pool.primary.model)

    dreaming_raw = cfg.get("dreaming", {})
    dreaming_cfg = DreamingConfig(
        hour=dreaming_raw.get("hour", 1),
        minute=dreaming_raw.get("minute", 0),
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
    )
    scheduler_cfg = SchedulerConfig(
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
    )
    agenda_cfg = cfg.get("agenda", {})
    agenda_enabled = agenda_cfg.get("enabled", True)
    agenda_interval = agenda_cfg.get("review_interval_minutes", 60)

    harvest_cfg_raw = cfg.get("memory_harvest", {})
    harvest_config = MemoryHarvestConfig(
        enabled=harvest_cfg_raw.get("enabled", True),
        message_threshold=harvest_cfg_raw.get("message_threshold", 20),
        inactivity_timeout_minutes=harvest_cfg_raw.get("inactivity_timeout_minutes", 15),
        max_message_chars=harvest_cfg_raw.get("max_message_chars", 2000),
    )

    # --- Optional interfaces ---
    matrix_cfg_raw: Optional[dict] = cfg.get("matrix")
    web_cfg: dict = cfg.get("web", {"enabled": True, "host": "127.0.0.1", "port": 8080})

    matrix_enabled = bool(
        matrix_cfg_raw
        and matrix_cfg_raw.get("homeserver")
        and (matrix_cfg_raw.get("access_token") or matrix_cfg_raw.get("password"))
    )
    web_enabled = web_cfg.get("enabled", True)

    if not matrix_enabled and not web_enabled:
        logger.error("Neither Matrix nor web interface is enabled - nothing to do. Exiting.")
        sys.exit(1)

    shutdown = ShutdownCoordinator()

    # Build Matrix (may be a no-op stub if not configured).
    if matrix_enabled:
        matrix_cfg = MatrixConfig(
            homeserver=matrix_cfg_raw["homeserver"],
            user_id=matrix_cfg_raw["user_id"],
            access_token=matrix_cfg_raw.get("access_token", ""),
            device_id=matrix_cfg_raw.get("device_id", ""),
            password=matrix_cfg_raw.get("password", ""),
            allowed_users=matrix_cfg_raw.get("allowed_users", []),
            allowed_rooms=matrix_cfg_raw.get("allowed_rooms", []),
        )
        matrix: Optional[MatrixThread] = MatrixThread(matrix_cfg, llm_thread=None)
    else:
        logger.info("Matrix not configured - skipping Matrix interface")
        matrix = None

    # Build web interface.
    web_iface: Optional[WebInterface] = None
    if web_enabled:
        web_iface = WebInterface(
            host=web_cfg.get("host", "127.0.0.1"),
            port=web_cfg.get("port", 8080),
            llm_thread=None,  # injected below
        )

    # Thread-aware broadcast: routes to the correct Matrix room or web client.
    # Reasoning tokens are only forwarded to the web interface, not Matrix.
    async def broadcast(text: str, thread_id: str = None, *,
                        reasoning: str = None) -> None:
        if matrix and thread_id and not thread_id.startswith("web_") and thread_id != "default":
            # thread_id is a Matrix room_id — no reasoning (too noisy)
            await matrix.send_message(text, thread_id)
        if web_iface and thread_id:
            await web_iface.broadcast(text, thread_id, reasoning=reasoning)

    # Build LLM thread with the shared broadcast function.
    tp_validators = (cfg.get("turing_protocol", {}) or {}).get("validators")
    llm = LLMThread(main_pool=main_pool, compaction_pool=compaction_pool,
                    turing_protocol_pool=turing_protocol_pool, broadcast_fn=broadcast,
                    turing_protocol_validators=tp_validators,
                    nl_translation_pool=nl_translation_pool,
                    nl_translation_config=nl_translation_config)

    # Build SubSessionManager — shares the LLM backend pool, reports back via
    # enqueue_system_event so results enter the parent thread's queue.
    # Turing Protocol pool + validators are forwarded so sub-sessions can run
    # phase-aware validation hooks (objective_completion, pre/post execution).
    sub_sessions = SubSessionManager(
        pool=sub_sessions_pool,
        enqueue_system_event=llm.enqueue_system_event,
        turing_protocol_pool=turing_protocol_pool,
        turing_protocol_validators=tp_validators,
        nl_translation_pool=nl_translation_pool,
        nl_translation_config=nl_translation_config,
    )
    llm.inject_sub_session_manager(sub_sessions)
    tool_module.register_sub_session_manager(sub_sessions.spawn)

    # Inject LLM into interfaces.
    if matrix:
        matrix._llm = llm
        matrix._sub_sessions = sub_sessions
        matrix._kimi_client = client_cache.get(("kimi-code",))
        # Whisper voice transcription.
        whisper_raw = cfg.get("whisper", {}) or {}
        if whisper_raw.get("enabled"):
            matrix._whisper_client = AsyncOpenAI(
                api_key=whisper_raw.get("api_key", ""),
                base_url=whisper_raw.get("base_url", ""),
                timeout=60.0,
            )
            matrix._whisper_model = whisper_raw.get("model", "whisper-1")
            matrix._whisper_language = whisper_raw.get("language", "") or ""
            logger.info("Whisper transcription enabled (model=%s)", matrix._whisper_model)
            import shutil as _shutil
            if not _shutil.which("ffmpeg"):
                logger.warning(
                    "Whisper transcription is enabled but ffmpeg is not in PATH. "
                    "Voice messages from Matrix (OGG/Opus) cannot be transcribed without ffmpeg. "
                    "Install it with: sudo apt install ffmpeg  OR  sudo dnf install ffmpeg"
                )
    if web_iface:
        web_iface._llm = llm
        web_iface._kimi_client = client_cache.get(("kimi-code",))

    scheduler = ReminderScheduler(
        config=scheduler_cfg,
        broadcast_fn=broadcast,
        llm_enqueue_fn=llm.enqueue_system_event,
        sub_session_manager=sub_sessions,
    )

    # Inject debug dependencies into web interface (after scheduler is built).
    if web_iface:
        web_iface._sub_sessions = sub_sessions
        web_iface._scheduler = scheduler
        web_iface._matrix = matrix
        web_iface._main_pool = main_pool
        web_iface._multi_cfg = multi_cfg

    agenda_loop: Optional[AgendaLoop] = None
    if agenda_enabled:
        agenda_loop = AgendaLoop(
            interval_minutes=agenda_interval,
            sub_session_manager=sub_sessions,
        )
    else:
        logger.info("Agenda loop disabled by config")

    harvest_loop: Optional[MemoryHarvestLoop] = None
    if harvest_config.enabled:
        harvest_loop = MemoryHarvestLoop(
            config=harvest_config,
            sub_session_manager=sub_sessions,
        )
    else:
        logger.info("Memory harvest disabled by config")

    dreaming_loop = DreamingLoop(
        config=dreaming_cfg,
        pool=dreaming_pool,
    )

    # Inject remaining references for /status and /dream commands.
    if matrix:
        matrix._scheduler = scheduler
        matrix._agenda_loop = agenda_loop
        matrix._dreaming_loop = dreaming_loop
    if web_iface:
        web_iface._agenda_loop = agenda_loop
        web_iface._dreaming_loop = dreaming_loop

    scheduler.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    tasks = [
        asyncio.create_task(llm.run(),              name="llm"),
        asyncio.create_task(dreaming_loop.run(),     name="dreaming"),
    ]
    if agenda_loop:
        tasks.append(asyncio.create_task(agenda_loop.run(), name="agenda"))
    if harvest_loop:
        tasks.append(asyncio.create_task(harvest_loop.run(), name="memory_harvest"))
    if matrix:
        tasks.append(asyncio.create_task(matrix.run(), name="matrix"))
    if web_iface:
        tasks.append(asyncio.create_task(web_iface.run(), name="web"))

    interfaces = []
    if matrix_enabled:
        interfaces.append("Matrix")
    if web_enabled:
        interfaces.append(f"web http://{web_cfg.get('host','127.0.0.1')}:{web_cfg.get('port',8080)}")
    logger.info("All components started. Interfaces: %s", ", ".join(interfaces))

    # Auto-trigger Kimi-Code device auth if credentials are missing.
    kimi_client_instance = client_cache.get(("kimi-code",))
    if kimi_client_instance and not kimi_client_instance.authenticated:
        async def _kimi_auto_auth() -> None:
            from wintermute import kimi_auth
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

    if agenda_loop:
        agenda_loop.stop()
    if harvest_loop:
        harvest_loop.stop()
    dreaming_loop.stop()
    if matrix:
        matrix.stop()
    llm.stop()
    scheduler.stop()

    for task in tasks:
        if not task.done():
            task.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Some tasks did not stop within 10 s — forcing exit")

    logger.info("Wintermute shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())


def run() -> None:
    """Entry point for the `wintermute` console script."""
    asyncio.run(main())
