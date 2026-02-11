"""
Ganglion – Matrix AI Reminder Assistant
Entry point and orchestration.

Startup sequence:
  1. Load config.yaml
  2. Configure logging
  3. Initialise SQLite databases
  4. Ensure data/ files exist
  5. Restore APScheduler jobs (and execute missed reminders)
  6. Connect to Matrix
  7. Start LLM inference task
  8. Start heartbeat review task
  9. Await shutdown signals

Shutdown sequence:
  1. Stop heartbeat loop
  2. Stop Matrix connection
  3. Stop LLM thread
  4. Stop scheduler (job store flushed automatically)
  5. Close any open resources
"""

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path

import yaml

import database
from heartbeat import HeartbeatLoop
from llm_thread import LLMConfig, LLMThread
from matrix_thread import MatrixConfig, MatrixThread
from scheduler_thread import ReminderScheduler, SchedulerConfig

CONFIG_FILE = Path("config.yaml")
LOG_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_FILE) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found. Copy config.yaml.example and fill in your credentials.")
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

    # Console handler.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)

    # Rotating file handler.
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
You are a personal AI assistant operating through Matrix.

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

_DEFAULT_MEMORIES = "# User Memories\n\n(No memories recorded yet.)\n"
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

    # Build component configs.
    matrix_cfg = MatrixConfig(
        homeserver=cfg["matrix"]["homeserver"],
        user_id=cfg["matrix"]["user_id"],
        access_token=cfg["matrix"]["access_token"],
        room_id=cfg["matrix"]["room_id"],
    )
    llm_cfg = LLMConfig(
        api_key=cfg["anthropic"]["api_key"],
        model=cfg["anthropic"].get("model", "claude-opus-4-5-20251101"),
        max_tokens=cfg["anthropic"].get("max_tokens", 4096),
    )
    scheduler_cfg = SchedulerConfig(
        timezone=cfg.get("scheduler", {}).get("timezone", "UTC"),
    )
    heartbeat_interval = cfg.get("heartbeat", {}).get("review_interval_minutes", 60)

    # Shutdown coordinator.
    shutdown = ShutdownCoordinator()

    # Construct components (order matters: Matrix send_fn needed by LLM).
    matrix = MatrixThread(matrix_cfg, llm_thread=None)  # llm injected below

    llm = LLMThread(
        config=llm_cfg,
        matrix_send_fn=matrix.send_message,
    )

    # Inject LLM back into Matrix.
    matrix._llm = llm

    scheduler = ReminderScheduler(
        config=scheduler_cfg,
        matrix_send_fn=matrix.send_message,
        llm_enqueue_fn=llm.enqueue_system_event,
    )

    heartbeat_loop = HeartbeatLoop(
        interval_minutes=heartbeat_interval,
        llm_enqueue_fn=llm.enqueue_user_message,
        matrix_send_fn=matrix.send_message,
    )

    # Start scheduler (synchronous APScheduler start).
    scheduler.start()

    # Register OS signal handlers for graceful shutdown.
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.request_shutdown)

    # Launch async tasks.
    tasks = [
        asyncio.create_task(llm.run(),        name="llm"),
        asyncio.create_task(matrix.run(),     name="matrix"),
        asyncio.create_task(heartbeat_loop.run(), name="heartbeat"),
    ]

    logger.info("All components started. Awaiting shutdown signal.")

    # Wait for shutdown signal.
    await shutdown.wait()
    logger.info("Shutdown requested – stopping components gracefully")

    # Stop in reverse order.
    heartbeat_loop.stop()
    matrix.stop()
    llm.stop()
    scheduler.stop()

    # Cancel remaining tasks.
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Ganglion shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
