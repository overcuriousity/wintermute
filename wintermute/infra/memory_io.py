"""
Read / write operations for long-term memory.

Memory entries are stored exclusively in the vector memory store.
The deprecated MEMORIES.txt flat file has been removed.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_text_safe(path: Path, default: str = "") -> str:
    """Read a text file, returning *default* on missing / unreadable files."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default
    except OSError as exc:
        logger.error("Cannot read %s: %s", path, exc)
        return default


def append_memory(
    entry: str, source: str = "unknown", *, pool=None, loop=None
) -> tuple[int, str]:
    """Add a memory entry to the vector store.

    When *pool* and *loop* are provided, uses similarity-based dedup
    (``add_with_dedup``) to merge near-duplicates via LLM.  Otherwise
    falls back to a plain ``add()``.

    Returns ``(total_entry_count, status)`` where *status* is one of
    ``"ok"``, ``"pending"`` (timeout — coroutine still in-flight), or
    ``"fallback"`` (dedup failed, plain add used).
    """
    from wintermute.infra import memory_store

    status = "ok"
    if pool is not None and loop is not None:
        import asyncio
        import concurrent.futures
        try:
            future = asyncio.run_coroutine_threadsafe(
                memory_store.add_with_dedup(entry.strip(), source=source, pool=pool),
                loop,
            )
            future.result(timeout=30)
        except (concurrent.futures.TimeoutError, TimeoutError):
            # The coroutine may still be running (e.g. blocked in the LLM
            # merge call).  Do NOT fall back to a plain add — that risks
            # duplicates if add_with_dedup eventually completes.  The entry
            # will be added when the coroutine finishes, or on the next
            # append if it was lost.
            logger.warning("add_with_dedup timed out; coroutine still in progress, skipping fallback")
            status = "pending"
        except Exception as exc:
            logger.error("add_with_dedup failed, falling back to plain add: %s", exc)
            memory_store.add(entry.strip(), source=source)
            status = "fallback"
    else:
        memory_store.add(entry.strip(), source=source)

    # No data_versioning commit here — memory is stored in SQLite binary
    # files (local_vectors.db / Qdrant) which are gitignored.
    return memory_store.count(), status
