"""
Read / write operations for MEMORIES.txt (long-term user facts).

Extracted from ``prompt_assembler`` (#81) so that prompt assembly and
persistent memory I/O live in separate modules.
"""

import logging
import threading
from pathlib import Path

from wintermute.infra import data_versioning
from wintermute.infra.paths import DATA_DIR, MEMORIES_FILE

logger = logging.getLogger(__name__)

# Lock guarding read-modify-write operations on MEMORIES.txt.
_memories_lock = threading.Lock()


def read_text_safe(path: Path, default: str = "") -> str:
    """Read a text file, returning *default* on missing / unreadable files."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return default


def update_memories(content: str) -> None:
    """Overwrite MEMORIES.txt with *content* and sync the vector store."""
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        MEMORIES_FILE.write_text(content, encoding="utf-8")
    logger.info("MEMORIES.txt updated (%d chars)", len(content))
    if memory_store.is_memory_backend_initialized():
        try:
            entries = [l.strip() for l in content.strip().splitlines() if l.strip()]
            memory_store.replace_all(entries)
        except Exception as exc:
            logger.error("Failed to sync memory backend on update_memories: %s", exc)
    data_versioning.commit_async("memory: consolidation")


def append_memory(entry: str, source: str = "unknown") -> int:
    """Append a memory entry to MEMORIES.txt. Returns the new total length."""
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        existing = read_text_safe(MEMORIES_FILE)
        if existing:
            new_content = existing + "\n" + entry.strip()
        else:
            new_content = entry.strip()
        MEMORIES_FILE.write_text(new_content, encoding="utf-8")
    logger.info("MEMORIES.txt appended (%d chars total)", len(new_content))
    if memory_store.is_memory_backend_initialized():
        try:
            memory_store.add(entry.strip(), source=source)
        except Exception as exc:
            logger.error("Failed to add memory to backend: %s", exc)
    data_versioning.commit_async("memory: append")
    return len(new_content)


def merge_consolidated_memories(snapshot: str, consolidated: str) -> None:
    """Atomically write *consolidated* memories while preserving any lines
    that were appended to MEMORIES.txt after *snapshot* was taken.

    This solves the race condition where ``append_memory()`` is called while
    the dreaming consolidation LLM call is in flight: the snapshot (taken
    before the LLM call) is diffed against the current file content, and any
    newly appended lines are tacked onto the consolidated result before
    writing.

    The entire read-diff-write cycle runs under ``_memories_lock`` so no
    appends can slip through.
    """
    from wintermute.infra import memory_store

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_lines = set(snapshot.strip().splitlines())
    with _memories_lock:
        current = read_text_safe(MEMORIES_FILE)
        new_lines = [
            line for line in current.strip().splitlines()
            if line not in snapshot_lines
        ]
        merged = consolidated.strip()
        if new_lines:
            merged = merged + "\n" + "\n".join(new_lines)
            logger.info(
                "merge_consolidated_memories: preserved %d appended line(s)",
                len(new_lines),
            )
        MEMORIES_FILE.write_text(merged, encoding="utf-8")
    logger.info("MEMORIES.txt merged-write (%d chars)", len(merged))
    if memory_store.is_memory_backend_initialized():
        try:
            entries = [l.strip() for l in merged.splitlines() if l.strip()]
            memory_store.replace_all(entries)
        except Exception as exc:
            logger.error("Failed to sync memory backend on merge_consolidated: %s", exc)


def write_memories_raw(content: str) -> None:
    """Write raw content to MEMORIES.txt under the memories lock.

    Unlike ``update_memories``, this does **not** sync the vector store or
    trigger a data-versioning commit — the caller is responsible for those
    side-effects.  Used by the dreaming working-set export phase.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _memories_lock:
        MEMORIES_FILE.write_text(content, encoding="utf-8")
