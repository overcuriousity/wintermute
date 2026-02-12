"""
Pulse Review Loop

Periodically invokes the LLM to review PULSE.txt and take autonomous
actions.  Runs both a global heartbeat and per-thread heartbeats.

Global pulse runs as an isolated sub-session (fire-and-forget, no parent
thread) so it never pollutes any user-facing conversation history.

Per-thread pulse reviews run through the normal LLM queue so their results are
delivered to the correct room/tab.
"""

import asyncio
import logging
from typing import Callable, Optional, TYPE_CHECKING

from ganglion import database

if TYPE_CHECKING:
    from ganglion.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

PULSE_REVIEW_PROMPT = (
    "This is an automatic pulse review. "
    "Please read your current PULSE.txt using the read_file tool, "
    "then review each item and take any appropriate actions (set reminders, "
    "update memories, run shell commands, etc.). "
    "If you update your pulse or take actions, briefly summarise what you did. "
    "If nothing needs attention right now, respond with a short status note."
)


class PulseLoop:
    """Runs as an asyncio task, periodically triggering pulse reviews."""

    def __init__(self, interval_minutes: int, llm_enqueue_fn,
                 broadcast_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._interval = interval_minutes * 60  # convert to seconds
        self._llm_enqueue = llm_enqueue_fn   # async callable(text: str, thread_id: str) -> str
        self._broadcast = broadcast_fn       # async callable(text: str, thread_id: str)
        self._sub_sessions = sub_session_manager
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info(
            "Pulse loop started (interval=%dm)", self._interval // 60
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
        """
        Global pulse review — fire-and-forget isolated sub-session.

        Uses 'full' system prompt so the worker has access to MEMORIES,
        HEARTBEATS, and SKILLS — the same context it would have in a normal
        conversation, without polluting any user thread's history.
        """
        logger.info("Running global pulse review")
        if self._sub_sessions is not None:
            self._sub_sessions.spawn(
                objective=PULSE_REVIEW_PROMPT,
                parent_thread_id=None,   # fire-and-forget
                system_prompt_mode="full",
            )
        else:
            # Fallback if sub-session manager not yet wired (should not happen
            # after the full startup sequence, but kept for safety).
            logger.warning("Global pulse: SubSessionManager not available, skipping")

    async def _review_per_thread(self) -> None:
        """Per-thread pulse review — one review per active thread."""
        thread_ids = database.get_active_thread_ids()
        for tid in thread_ids:
            if tid == "default":
                continue  # already handled by global
            logger.info("Running pulse review for thread %s", tid)
            try:
                reply = await self._llm_enqueue(PULSE_REVIEW_PROMPT, tid)
                if reply and not _is_silent(reply):
                    await self._broadcast(
                        f"\U0001f493 Pulse review:\n\n{reply}", tid
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Pulse review failed for thread %s: %s", tid, exc)


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
