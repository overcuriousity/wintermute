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
        import concurrent.futures
        try:
            future = asyncio.run_coroutine_threadsafe(
                memory_store.add_with_dedup(entry.strip(), source=source, pool=pool),
                loop,
            )
            future.result(timeout=30)
        except (concurrent.futures.TimeoutError, TimeoutError):
            # The coroutine is still running on the event loop — cancel it
            # to avoid a duplicate add if we fall back.
            future.cancel()
            logger.warning("add_with_dedup timed out, attempting cancel + plain add")
            # Give cancellation a moment to propagate; if the coroutine
            # already completed we accept the result and skip fallback.
            try:
                future.result(timeout=2)
                # Completed successfully despite the earlier timeout — no fallback needed.
            except (asyncio.CancelledError, concurrent.futures.TimeoutError, TimeoutError):
                memory_store.add(entry.strip(), source=source)
            except Exception:
                memory_store.add(entry.strip(), source=source)
        except Exception as exc:
            logger.error("add_with_dedup failed, falling back to plain add: %s", exc)
            memory_store.add(entry.strip(), source=source)
    else:
        memory_store.add(entry.strip(), source=source)

    # No data_versioning commit here — memory is stored in SQLite binary
    # files (local_vectors.db / Qdrant) which are gitignored.
    return memory_store.count()
