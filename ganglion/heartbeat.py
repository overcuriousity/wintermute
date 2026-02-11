"""
Heartbeat Review Loop

Periodically invokes the LLM to review HEARTBEATS.txt and take autonomous
actions.  Runs both a global heartbeat and per-thread heartbeats.
"""

import asyncio
import logging
from typing import Callable

from ganglion import database

logger = logging.getLogger(__name__)

HEARTBEAT_REVIEW_PROMPT = (
    "This is an automatic heartbeat review. "
    "Please read your current HEARTBEATS.txt using the read_file tool, "
    "then review each item and take any appropriate actions (set reminders, "
    "update memories, run shell commands, etc.). "
    "If you update heartbeats or take actions, briefly summarise what you did. "
    "If nothing needs attention right now, respond with a short status note."
)


class HeartbeatLoop:
    """Runs as an asyncio task, periodically triggering heartbeat reviews."""

    def __init__(self, interval_minutes: int, llm_enqueue_fn,
                 broadcast_fn) -> None:
        self._interval = interval_minutes * 60  # convert to seconds
        self._llm_enqueue = llm_enqueue_fn   # async callable(text: str, thread_id: str) -> str
        self._broadcast = broadcast_fn       # async callable(text: str, thread_id: str)
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(
            "Heartbeat loop started (interval=%dm)", self._interval // 60
        )
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            await self._review_global()
            await self._review_per_thread()

    def stop(self) -> None:
        self._running = False

    async def _review_global(self) -> None:
        """Global heartbeat review — not tied to any thread."""
        logger.info("Running global heartbeat review")
        try:
            reply = await self._llm_enqueue(HEARTBEAT_REVIEW_PROMPT, "default")
            if reply and not _is_silent(reply):
                logger.info("Global heartbeat reply: %s", reply[:200])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Global heartbeat review failed: %s", exc)

    async def _review_per_thread(self) -> None:
        """Per-thread heartbeat review — one review per active thread."""
        thread_ids = database.get_active_thread_ids()
        for tid in thread_ids:
            if tid == "default":
                continue  # already handled by global
            logger.info("Running heartbeat review for thread %s", tid)
            try:
                reply = await self._llm_enqueue(HEARTBEAT_REVIEW_PROMPT, tid)
                if reply and not _is_silent(reply):
                    await self._broadcast(
                        f"\U0001f493 Heartbeat review:\n\n{reply}", tid
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Heartbeat review failed for thread %s: %s", tid, exc)


def _is_silent(reply: str) -> bool:
    """
    Heuristic: if the reply is very short and contains common 'nothing to do'
    phrases, skip sending it to Matrix to avoid noise.
    """
    short = len(reply) < 200
    indicators = [
        "nothing to do",
        "no actions needed",
        "everything looks good",
        "no immediate actions",
        "no updates needed",
        "nothing requires attention",
        "all looks good",
    ]
    lower = reply.lower()
    return short and any(phrase in lower for phrase in indicators)
