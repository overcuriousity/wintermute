"""
Git auto-versioning for the data/ directory.

Initialises a local git repository inside data/ on first use and provides
an ``auto_commit()`` helper that stages all unignored changes and commits
them.  This gives a full change history for MEMORIES.txt, skills, and other
mutable data files so that any change can be manually rolled back.

For fire-and-forget background commits use ``commit_async()``, which queues
the work onto a single non-daemon worker thread.  Call ``drain()`` during
shutdown to flush all pending commits before the process exits::

    data_versioning.commit_async("memory: append")
    # ... at shutdown:
    data_versioning.drain()
"""

import logging
import queue
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

_lock = threading.Lock()

_GITIGNORE_CONTENT = """\
conversation.db
scheduler.db
routine_history.json
task_history.json
matrix_store/
matrix_crypto.db*
matrix_signed.marker
matrix_recovery.key
*credentials*.json
kimi_device_id.txt
voice/
"""


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command inside DATA_DIR."""
    return subprocess.run(
        ["git", *args],
        cwd=DATA_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _ensure_repo() -> None:
    """Initialise the git repo and .gitignore if they don't exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    git_dir = DATA_DIR / ".git"
    if not git_dir.exists():
        result = _run_git("init")
        if result.returncode != 0:
            logger.error("data_versioning: git init failed: %s", result.stderr)
            return
        logger.info("data_versioning: initialised git repo in data/")

    gitignore = DATA_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_GITIGNORE_CONTENT, encoding="utf-8")
        logger.info("data_versioning: created data/.gitignore")


def auto_commit(message: str) -> None:
    """Stage all unignored changes in data/ and commit.

    Silently does nothing if there are no changes to commit.
    Thread-safe — concurrent callers are serialised.
    """
    with _lock:
        try:
            _ensure_repo()
            _run_git("add", "-A")
            # Check if there is anything staged.
            status = _run_git("diff", "--cached", "--quiet")
            if status.returncode == 0:
                # Nothing staged — skip commit.
                return
            result = _run_git("commit", "-m", message)
            if result.returncode != 0:
                logger.warning("data_versioning: commit failed: %s", result.stderr.strip())
            else:
                logger.debug("data_versioning: committed — %s", message)
        except Exception:  # noqa: BLE001
            logger.exception("data_versioning: auto_commit failed")


# ---------------------------------------------------------------------------
# Async commit helpers (single worker thread + queue, shutdown drain)
# ---------------------------------------------------------------------------

_queue: queue.Queue[str | None] = queue.Queue()
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None


def _worker_loop() -> None:
    """Drain the commit queue until the stop sentinel (None) is received."""
    while True:
        message = _queue.get()
        try:
            if message is None:
                return  # Stop sentinel — shut down worker.
            auto_commit(message)
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_worker_loop, name="data-commit-worker", daemon=False,
            )
            _worker.start()


def commit_async(message: str) -> None:
    """Queue a commit to run on the background worker thread.

    Commits are processed in FIFO order by a single non-daemon worker, so
    concurrent callers never pile up blocked threads.
    """
    _ensure_worker()
    _queue.put(message)


def drain(timeout: float = 5.0) -> None:
    """Flush all pending commits and stop the worker.  Called during shutdown.

    Sends a stop sentinel so the worker exits cleanly after processing all
    queued messages, then joins the thread.  Because the worker is non-daemon,
    the interpreter will wait for it regardless; the timeout controls how long
    we log a warning before giving up on a graceful join.
    """
    with _worker_lock:
        w = _worker
    if w is None or not w.is_alive():
        return
    logger.info("data_versioning: draining pending commits…")
    _queue.put(None)  # Stop sentinel — processed after all queued commits.
    w.join(timeout=timeout)
    if w.is_alive():
        logger.warning("data_versioning: worker still alive after %.1fs", timeout)
