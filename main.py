"""
Ganglion – Matrix AI Reminder Assistant
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
  10. Start heartbeat review task
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

import database
from heartbeat import HeartbeatLoop
from llm_thread import LLMConfig, LLMThread
from matrix_thread import MatrixConfig, MatrixThread
from scheduler_thread import ReminderScheduler, SchedulerConfig
from web_interface import WebInterface

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
        LOG_DIR / "ganglion.log",
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
- HEARTBEATS.txt is your working memory for active goals and recurring tasks.
- skills/ contains documentation for capabilities you have learned.

You can call tools to: set reminders, update your memory files, execute shell
commands, read and write files, and add new skills.

When the user asks you to remember something, use update_memories.
When you need to track an ongoing goal, use update_heartbeats.
When you learn a new reusable procedure, document it with add_skill.
When you need to schedule something, use set_reminder.

Be concise, helpful, and proactive. Always confirm when you have taken an
action (set a reminder, updated memories, etc.).
"""

_DEFAULT_MEMORIES   = "# User Memories\n\n(No memories recorded yet.)\n"
_DEFAULT_HEARTBEATS = "# Active Heartbeats\n\n(No active heartbeats.)\n"


def bootstrap_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "skills").mkdir(exist_ok=True)
    (DATA_DIR / "scripts").mkdir(exist_ok=True)

    base = DATA_DIR / "BASE_PROMPT.txt"
    if not base.exists():
        base.write_text(_DEFAULT_BASE_PROMPT, encoding="utf-8")
        logging.getLogger(__name__).info("Created default BASE_PROMPT.txt")

    memories = DATA_DIR / "MEMORIES.txt"
    if not memories.exists():
        memories.write_text(_DEFAULT_MEMORIES, encoding="utf-8")

    heartbeats = DATA_DIR / "HEARTBEATS.txt"
    if not heartbeats.exists():
        heartbeats.write_text(_DEFAULT_HEARTBEATS, encoding="utf-8")


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
    logger.info("Ganglion starting up")

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
    scheduler_cfg = SchedulerConfig(
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
    )
    heartbeat_interval = cfg.get("heartbeat", {}).get("review_interval_minutes", 60)

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
        logger.error("Neither Matrix nor web interface is enabled – nothing to do. Exiting.")
        sys.exit(1)

    shutdown = ShutdownCoordinator()

    # Build Matrix (may be a no-op stub if not configured).
    if matrix_enabled:
        matrix_cfg = MatrixConfig(
            homeserver=matrix_cfg_raw["homeserver"],
            user_id=matrix_cfg_raw["user_id"],
            access_token=matrix_cfg_raw["access_token"],
            room_id=matrix_cfg_raw["room_id"],
        )
        matrix: Optional[MatrixThread] = MatrixThread(matrix_cfg, llm_thread=None)
    else:
        logger.info("Matrix not configured – skipping Matrix interface")
        matrix = None

    # Build web interface.
    web: Optional[WebInterface] = None
    if web_enabled:
        web = WebInterface(
            host=web_cfg.get("host", "127.0.0.1"),
            port=web_cfg.get("port", 8080),
            llm_thread=None,  # injected below
        )

    # Shared broadcast: fan-out to both interfaces.
    async def broadcast(text: str) -> None:
        if matrix:
            await matrix.send_message(text)
        if web:
            await web.broadcast(text)

    # Build LLM thread with the shared broadcast function.
    llm = LLMThread(config=llm_cfg, matrix_send_fn=broadcast)

    # Inject LLM into interfaces.
    if matrix:
        matrix._llm = llm
    if web:
        web._llm = llm

    scheduler = ReminderScheduler(
        config=scheduler_cfg,
        matrix_send_fn=broadcast,
        llm_enqueue_fn=llm.enqueue_system_event,
    )

    heartbeat_loop = HeartbeatLoop(
        interval_minutes=heartbeat_interval,
        llm_enqueue_fn=llm.enqueue_user_message,
        matrix_send_fn=broadcast,
    )

    scheduler.start()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    tasks = [
        asyncio.create_task(llm.run(),            name="llm"),
        asyncio.create_task(heartbeat_loop.run(), name="heartbeat"),
    ]
    if matrix:
        tasks.append(asyncio.create_task(matrix.run(), name="matrix"))
    if web:
        tasks.append(asyncio.create_task(web.run(), name="web"))

    interfaces = []
    if matrix_enabled:
        interfaces.append("Matrix")
    if web_enabled:
        interfaces.append(f"web http://{web_cfg.get('host','127.0.0.1')}:{web_cfg.get('port',8080)}")
    logger.info("All components started. Interfaces: %s", ", ".join(interfaces))

    await shutdown.wait()
    logger.info("Shutdown requested – stopping components gracefully")

    heartbeat_loop.stop()
    if matrix:
        matrix.stop()
    llm.stop()
    scheduler.stop()

    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Ganglion shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
