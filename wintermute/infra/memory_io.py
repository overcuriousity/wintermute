"""
Read / write operations for long-term memory.

Memory entries are stored exclusively in the vector memory store.
The deprecated MEMORIES.txt flat file has been removed.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Phrases that indicate the model is echoing system/infrastructure instructions.
_SYSTEM_ECHO_PHRASES = (
    "you are an ai",
    "your task is",
    "respond in json",
    "convergence protocol",
    "system prompt",
    "sub-session",
    "function calling",
)


def _validate_memory_entry(entry: str) -> tuple[bool, str]:
    """Programmatic quality gate for memory writes.

    Returns ``(True, "")`` when the entry is acceptable, or
    ``(False, reason)`` when it should be rejected.  Fail-open: if
    validation itself throws, the entry is allowed through.
    """
    try:
        text = entry.strip()

        # --- length bounds ---
        if len(text) < 10:
            return False, "too_short"
        if len(text) > 2000:
            return False, "too_long"

        # --- raw JSON / code dump ---
        if text[0] in ("{", "["):
            try:
                json.loads(text)
                return False, "raw_json"
            except (json.JSONDecodeError, ValueError):
                pass

        # --- system prompt echo (3+ phrase matches) ---
        lower = text.lower()
        matches = sum(1 for p in _SYSTEM_ECHO_PHRASES if p in lower)
        if matches >= 3:
            return False, "system_echo"

        # --- repetitive content (single word dominates the text) ---
        # Filter out tokens ≤2 chars to avoid false-positives from stopwords.
        # Require ≥20 filtered tokens before applying, and flag only when a
        # single word both appears >5 times AND makes up ≥20% of the text.
        tokens = re.findall(r"\b\w+\b", lower)
        filtered = [t for t in tokens if len(t) > 2]
        if len(filtered) >= 20:
            from collections import Counter
            top_word, top_count = Counter(filtered).most_common(1)[0]
            if top_count > 5 and top_count / len(filtered) >= 0.2:
                return False, "repetitive"

        # --- encoding artifacts ---
        if "\x00" in text or re.search(r"\\{3,}", text):
            return False, "encoding_artifacts"

    except Exception:
        logger.debug("Memory validation error (fail-open)", exc_info=True)

    return True, ""


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
    entry: str, source: str = "unknown", *, pool=None, loop=None,
    event_bus=None,
) -> tuple[int, str]:
    """Add a memory entry to the vector store.

    When *pool* and *loop* are provided, uses similarity-based dedup
    (``add_with_dedup``) to merge near-duplicates via LLM.  Otherwise
    falls back to a plain ``add()``.

    Event emission is handled here (single source of truth):
    - Immediate ``memory.appended`` on synchronous success or fallback.
    - Deferred ``memory.appended`` via done-callback on timeout.

    Returns ``(total_entry_count, status)`` where *status* is one of
    ``"ok"``, ``"pending"`` (timeout — coroutine still in-flight),
    ``"rejected"`` (entry failed validation gate), or ``"fallback"``
    (dedup failed, plain add used).
    """
    from wintermute.infra import memory_store

    valid, reason = _validate_memory_entry(entry)
    if not valid:
        logger.warning("Memory entry rejected: %s (entry: %.100s)", reason, entry)
        return memory_store.count(), "rejected"

    status = "ok"
    _entry_preview = entry[:200]

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
            # duplicates if add_with_dedup eventually completes.
            logger.warning("add_with_dedup timed out; coroutine still in progress, skipping fallback")
            status = "pending"
            # Attach done-callback so errors are logged and deferred event
            # fires once the coroutine eventually completes.
            _eb = event_bus

            def _on_done(fut):
                try:
                    fut.result()
                    logger.info("add_with_dedup completed after timeout")
                    if _eb:
                        _eb.emit("memory.appended", entry=_entry_preview)
                except Exception as exc:
                    logger.error("add_with_dedup failed after timeout: %s", exc)

            future.add_done_callback(_on_done)
        except Exception as exc:
            logger.error("add_with_dedup failed, falling back to plain add: %s", exc)
            memory_store.add(entry.strip(), source=source)
            status = "fallback"
    else:
        memory_store.add(entry.strip(), source=source)

    # Emit event immediately for non-pending completions.
    if status != "pending" and event_bus:
        event_bus.emit("memory.appended", entry=_entry_preview)

    # No data_versioning commit here — memory is stored in SQLite binary
    # files (local_vectors.db / Qdrant) which are gitignored.
    return memory_store.count(), status
