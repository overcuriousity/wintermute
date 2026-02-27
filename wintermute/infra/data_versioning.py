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

from wintermute.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

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
_draining = False


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


def _start_worker_locked() -> None:
    """Start the worker thread if not already running.  Caller must hold _worker_lock."""
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(
            target=_worker_loop, name="data-commit-worker", daemon=False,
        )
        _worker.start()


def commit_async(message: str) -> None:
    """Queue a commit to run on the background worker thread.

    Commits are processed in FIFO order by a single non-daemon worker, so
    concurrent callers never pile up blocked threads.  Once ``drain()`` has
    been called, falls back to a synchronous commit so no work is lost.
    """
    with _worker_lock:
        if _draining:
            # Shutdown in progress — run synchronously so the commit isn't lost.
            auto_commit(message)
            return
        _start_worker_locked()
        _queue.put(message)


def drain() -> None:
    """Flush all pending commits and stop the worker.  Called during shutdown.

    Sets a draining flag (under _worker_lock) before inserting the sentinel,
    so no new messages can be enqueued after the sentinel — preventing the
    race where a concurrent commit_async() call posts after the sentinel and
    is left permanently unprocessed.  Blocks until the worker thread exits.

    Because the worker is non-daemon, the caller should run this in a thread
    pool if it must not block an event loop (e.g. via ``asyncio.to_thread``).
    """
    global _draining
    with _worker_lock:
        _draining = True
        w = _worker
        if w is None or not w.is_alive():
            return
        # Insert sentinel inside the lock — no new messages can be queued after this.
        _queue.put(None)
    logger.info("data_versioning: draining pending commits…")
    w.join()
