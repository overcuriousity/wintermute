"""
Git auto-versioning for the data/ directory.

Initialises a local git repository inside data/ on first use and provides
an ``auto_commit()`` helper that stages all unignored changes and commits
them.  This gives a full change history for MEMORIES.txt, skills, and other
mutable data files so that any change can be manually rolled back.

Usage from async code::

    loop.run_in_executor(None, auto_commit, "memory: append")
"""

import logging
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
