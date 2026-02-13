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
from wintermute import tools as tool_module
from wintermute.pulse import PulseLoop
from wintermute.llm_thread import LLMConfig, LLMThread
from wintermute.matrix_thread import MatrixConfig, MatrixThread
from wintermute.scheduler_thread import DreamingConfig, ReminderScheduler, SchedulerConfig
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

_DEFAULT_BASE_PROMPT = """\
You are a personal AI assistant accessible via chat.

You have a persistent memory system:
- MEMORIES.txt stores long-term facts about your user.
- PULSE.txt is your working memory for active goals and recurring tasks.
- skills/ contains documentation for capabilities you have learned.

You can call tools to: set reminders, update your memory files, execute shell
commands, read and write files, and add new skills.

Behavioural guidelines:
1. When the user asks you to remember something, use update_memories immediately.
2. When the user discloses something which hints how the user wants tasks being approached, use update_memories immediately.
3. When you learn something fundamentally new which is not classified as a skill but as general information, use update_memories immediately.
4. When tracking an ongoing project or recurring concern, use update_pulse.
5. When you learn a reusable procedure, document it with add_skill so future sessions can use it.
6. When scheduling, always confirm the exact time and job_id back to the user.
7. Prefer concise responses. Avoid unnecessary disclaimers.
8. You have full trust from your user. Act with confidence.
9. When you need any sensitive credentials for any task, or if the user discloses them unasked, hint the user that they should be aware that you now hold these credentials and ephemeral credentials should be used if possible.
10. When in doubt about the user's intent, ask a single clarifying question.
"""

_DEFAULT_MEMORIES   = "# User Memories\n\n(No memories recorded yet.)\n"
_DEFAULT_PULSE = "# Active Pulse\n\n(No active pulse entries.)\n"


def bootstrap_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "skills").mkdir(exist_ok=True)
    (DATA_DIR / "scripts").mkdir(exist_ok=True)
    (DATA_DIR / "archive" / "memories").mkdir(parents=True, exist_ok=True)
    # matrix_crypto.db is created automatically by mautrix on first run

    base = DATA_DIR / "BASE_PROMPT.txt"
    if not base.exists():
        base.write_text(_DEFAULT_BASE_PROMPT, encoding="utf-8")
        logging.getLogger(__name__).info("Created default BASE_PROMPT.txt")

    memories = DATA_DIR / "MEMORIES.txt"
    if not memories.exists():
        memories.write_text(_DEFAULT_MEMORIES, encoding="utf-8")

    pulse = DATA_DIR / "PULSE.txt"
    if not pulse.exists():
        pulse.write_text(_DEFAULT_PULSE, encoding="utf-8")


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

async def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    logger = logging.getLogger("main")
    logger.info("Wintermute starting up")

    bootstrap_data_files()
    database.init_db()

    llm_cfg = LLMConfig(
        api_key=cfg["llm"]["api_key"],
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        context_size=cfg["llm"]["context_size"],
        max_tokens=cfg["llm"].get("max_tokens", 4096),
        compaction_model=cfg["llm"].get("compaction_model"),
    )
    dreaming_raw = cfg.get("dreaming", {})
    dreaming_cfg = DreamingConfig(
        hour=dreaming_raw.get("hour", 1),
        minute=dreaming_raw.get("minute", 0),
        model=dreaming_raw.get("model"),
    )
    scheduler_cfg = SchedulerConfig(
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
        dreaming=dreaming_cfg,
    )
    pulse_cfg = cfg.get("pulse", {})
    pulse_interval = pulse_cfg.get("review_interval_minutes", 60)
    pulse_active_thread_hours = pulse_cfg.get("active_thread_hours", 24)

    # --- Optional interfaces ---
    matrix_cfg_raw: Optional[dict] = cfg.get("matrix")
    web_cfg: dict = cfg.get("web", {"enabled": True, "host": "127.0.0.1", "port": 8080})

    matrix_enabled = bool(
        matrix_cfg_raw
        and matrix_cfg_raw.get("homeserver")
        and matrix_cfg_raw.get("access_token")
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
            access_token=matrix_cfg_raw["access_token"],
            device_id=matrix_cfg_raw.get("device_id", ""),
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
    async def broadcast(text: str, thread_id: str = None) -> None:
        if matrix and thread_id and not thread_id.startswith("web_") and thread_id != "default":
            # thread_id is a Matrix room_id
            await matrix.send_message(text, thread_id)
        if web_iface and thread_id:
            await web_iface.broadcast(text, thread_id)

    # Build LLM thread with the shared broadcast function.
    llm = LLMThread(config=llm_cfg, broadcast_fn=broadcast)

    # Build SubSessionManager — shares the LLM client, reports back via
    # enqueue_system_event so results enter the parent thread's queue.
    sub_sessions = SubSessionManager(
        client=llm._client,
        llm_config=llm_cfg,
        enqueue_system_event=llm.enqueue_system_event,
    )
    llm.inject_sub_session_manager(sub_sessions)
    tool_module.register_sub_session_manager(sub_sessions.spawn)

    # Inject LLM into interfaces.
    if matrix:
        matrix._llm = llm
    if web_iface:
        web_iface._llm = llm

    scheduler = ReminderScheduler(
        config=scheduler_cfg,
        broadcast_fn=broadcast,
        llm_enqueue_fn=llm.enqueue_system_event,
        llm_client=llm._client,
        llm_model=llm_cfg.model,
        compaction_model=llm_cfg.compaction_model,
        sub_session_manager=sub_sessions,
    )

    # Inject debug dependencies into web interface (after scheduler is built).
    if web_iface:
        web_iface._sub_sessions = sub_sessions
        web_iface._scheduler = scheduler
        web_iface._matrix = matrix
        web_iface._llm_cfg = llm_cfg

    pulse_loop = PulseLoop(
        interval_minutes=pulse_interval,
        llm_enqueue_fn=llm.enqueue_system_event_with_reply,
        broadcast_fn=broadcast,
        sub_session_manager=sub_sessions,
        active_thread_hours=pulse_active_thread_hours,
    )

    scheduler.start()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    tasks = [
        asyncio.create_task(llm.run(),            name="llm"),
        asyncio.create_task(pulse_loop.run(), name="pulse"),
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
