"""
Read / write operations for long-term memory.

Memory entries are stored exclusively in the vector memory store.
The deprecated MEMORIES.txt flat file has been removed.
"""

import logging
from pathlib import Path

from wintermute.infra import data_versioning

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


def append_memory(entry: str, source: str = "unknown", *, pool=None, loop=None) -> int:
    """Add a memory entry to the vector store.

    When *pool* and *loop* are provided, uses similarity-based dedup
    (``add_with_dedup``) to merge near-duplicates via LLM.  Otherwise
    falls back to a plain ``add()``.

    Returns the new total entry count.
    """
    from wintermute.infra import memory_store

    if pool is not None and loop is not None:
        import asyncio
        try:
            future = asyncio.run_coroutine_threadsafe(
                memory_store.add_with_dedup(entry.strip(), source=source, pool=pool),
                loop,
            )
            future.result(timeout=30)
        except Exception as exc:
            logger.error("add_with_dedup failed, falling back to plain add: %s", exc)
            memory_store.add(entry.strip(), source=source)
    else:
        memory_store.add(entry.strip(), source=source)

    data_versioning.commit_async("memory: append")
    return memory_store.count()
