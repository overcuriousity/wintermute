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
  10. Start pulse review task
  11. Await shutdown signals
"""

import asyncio
import logging
import logging.handlers
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml

from wintermute import database
from wintermute import prompt_assembler
from wintermute import tools as tool_module
from wintermute.pulse import PulseLoop
from openai import AsyncOpenAI

from wintermute.llm_thread import LLMConfig, LLMThread, MultiProviderConfig, ProviderConfig
from wintermute.matrix_thread import MatrixConfig, MatrixThread
from wintermute.dreaming import DreamingConfig, DreamingLoop
from wintermute.scheduler_thread import ReminderScheduler, SchedulerConfig
from wintermute.sub_session import SubSessionManager
from wintermute.web_interface import WebInterface

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

    All prompt files (BASE_PROMPT.txt, MEMORIES.txt, PULSE.txt,
    DREAM_*_PROMPT.txt, COMPACTION_PROMPT.txt) are shipped in data/ and
    managed as editable configuration — they are NOT auto-generated.
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

def _build_provider(base: dict, overrides: dict) -> ProviderConfig:
    """Merge partial *overrides* onto *base* and return a ProviderConfig."""
    merged = {**base, **{k: v for k, v in overrides.items() if v is not None}}
    return ProviderConfig(
        api_key=merged["api_key"],
        base_url=merged["base_url"],
        model=merged["model"],
        context_size=merged.get("context_size", 32768),
        max_tokens=merged.get("max_tokens", 4096),
        reasoning=merged.get("reasoning", False),
    )


def _build_multi_provider_config(llm_raw: dict) -> MultiProviderConfig:
    """Parse the llm: block (with backward compat) into a MultiProviderConfig."""
    # Base fields (everything except the sub-keys).
    sub_keys = {"compaction", "sub_sessions", "dreaming", "supervisor"}
    base = {k: v for k, v in llm_raw.items()
            if k not in sub_keys and k != "compaction_model"}

    # Backward compat: old flat compaction_model -> compaction.model
    compaction_overrides = dict(llm_raw.get("compaction") or {})
    if "compaction_model" in llm_raw and "model" not in compaction_overrides:
        compaction_overrides["model"] = llm_raw["compaction_model"]

    # Backward compat: old top-level dreaming.model is handled below in main()
    # after reading the dreaming: section.

    main_cfg = _build_provider(base, {})
    compaction_cfg = _build_provider(base, compaction_overrides)
    sub_sessions_cfg = _build_provider(base, dict(llm_raw.get("sub_sessions") or {}))
    dreaming_cfg = _build_provider(base, dict(llm_raw.get("dreaming") or {}))

    # Supervisor: strip the 'enabled' key before passing to _build_provider
    # (it's not a ProviderConfig field).  Default max_tokens to 150 since the
    # supervisor only produces a small JSON response — avoids inheriting the
    # main provider's (much larger) max_tokens.
    supervisor_overrides = dict(llm_raw.get("supervisor") or {})
    supervisor_overrides.pop("enabled", None)
    supervisor_overrides.setdefault("max_tokens", 150)
    supervisor_cfg = _build_provider(base, supervisor_overrides)

    return MultiProviderConfig(
        main=main_cfg,
        compaction=compaction_cfg,
        sub_sessions=sub_sessions_cfg,
        dreaming=dreaming_cfg,
        supervisor=supervisor_cfg,
    )


def _make_clients(cfg: MultiProviderConfig) -> dict[str, AsyncOpenAI]:
    """Create AsyncOpenAI clients, sharing instances when (base_url, api_key) match."""
    cache: dict[tuple[str, str], AsyncOpenAI] = {}
    result: dict[str, AsyncOpenAI] = {}
    for purpose, pcfg in [("main", cfg.main), ("compaction", cfg.compaction),
                          ("sub_sessions", cfg.sub_sessions), ("dreaming", cfg.dreaming),
                          ("supervisor", cfg.supervisor)]:
        key = (pcfg.base_url, pcfg.api_key)
        if key not in cache:
            cache[key] = AsyncOpenAI(api_key=pcfg.api_key, base_url=pcfg.base_url)
        result[purpose] = cache[key]
    return result


async def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    logger = logging.getLogger("main")
    logger.info("Wintermute starting up")

    bootstrap_data_files()
    database.init_db()

    # Set timezone for prompt assembler (used to inject current datetime).
    configured_tz = cfg.get("scheduler", {}).get("timezone", "UTC")
    prompt_assembler.set_timezone(configured_tz)

    multi_cfg = _build_multi_provider_config(cfg["llm"])

    # Backward compat: old top-level dreaming.model -> llm.dreaming.model
    dreaming_raw = cfg.get("dreaming", {})
    if dreaming_raw.get("model") and multi_cfg.dreaming.model == multi_cfg.main.model:
        multi_cfg.dreaming = _build_provider(
            {k: v for k, v in cfg["llm"].items()
             if k not in {"compaction", "sub_sessions", "dreaming", "compaction_model"}},
            {"model": dreaming_raw["model"]},
        )
        # Refresh client cache after dreaming config changed.

    clients = _make_clients(multi_cfg)
    llm_cfg = multi_cfg.main  # backward-compat alias used throughout

    dreaming_cfg = DreamingConfig(
        hour=dreaming_raw.get("hour", 1),
        minute=dreaming_raw.get("minute", 0),
    )
    scheduler_cfg = SchedulerConfig(
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
    )
    pulse_cfg = cfg.get("pulse", {})
    pulse_interval = pulse_cfg.get("review_interval_minutes", 60)

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
    supervisor_enabled = cfg["llm"].get("supervisor", {}).get("enabled", True)
    llm = LLMThread(config=llm_cfg, broadcast_fn=broadcast,
                    multi_cfg=multi_cfg, clients=clients,
                    supervisor_enabled=supervisor_enabled)

    # Build SubSessionManager — shares the LLM client, reports back via
    # enqueue_system_event so results enter the parent thread's queue.
    sub_sessions = SubSessionManager(
        client=clients["sub_sessions"],
        llm_config=multi_cfg.sub_sessions,
        enqueue_system_event=llm.enqueue_system_event,
    )
    llm.inject_sub_session_manager(sub_sessions)
    tool_module.register_sub_session_manager(sub_sessions.spawn)

    # Inject LLM into interfaces.
    if matrix:
        matrix._llm = llm
        matrix._sub_sessions = sub_sessions
    if web_iface:
        web_iface._llm = llm

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
        web_iface._llm_cfg = llm_cfg
        web_iface._multi_cfg = multi_cfg

    pulse_loop = PulseLoop(
        interval_minutes=pulse_interval,
        sub_session_manager=sub_sessions,
    )

    dreaming_loop = DreamingLoop(
        config=dreaming_cfg,
        llm_client=clients["dreaming"],
        llm_model=multi_cfg.dreaming.model,
        reasoning=multi_cfg.dreaming.reasoning,
    )

    # Inject remaining references for /status and /dream commands.
    if matrix:
        matrix._scheduler = scheduler
        matrix._pulse_loop = pulse_loop
        matrix._dreaming_loop = dreaming_loop
    if web_iface:
        web_iface._pulse_loop = pulse_loop
        web_iface._dreaming_loop = dreaming_loop

    scheduler.start()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    tasks = [
        asyncio.create_task(llm.run(),              name="llm"),
        asyncio.create_task(pulse_loop.run(),        name="pulse"),
        asyncio.create_task(dreaming_loop.run(),     name="dreaming"),
    ]
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

    await shutdown.wait()
    logger.info("Shutdown requested - stopping components gracefully")

    pulse_loop.stop()
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
