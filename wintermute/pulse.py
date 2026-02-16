"""
Pulse Review Loop

Periodically invokes the LLM to review active pulse items and take autonomous
actions.  Runs as an isolated sub-session (fire-and-forget, no parent
thread) so it never pollutes any user-facing conversation history.
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

PULSE_REVIEW_PROMPT = (
    "This is an automatic pulse review. "
    "Use the pulse tool with action 'list' to see current items, "
    "then review each item and take any appropriate actions (set reminders, "
    "update memories, complete items, run shell commands, etc.). "
    "Use pulse(action='complete', item_id=...) for finished items, "
    "and pulse(action='add', content='...') for new items discovered. "
    "If you take actions, briefly summarise what you did. "
    "If nothing needs attention right now, respond with a short status note."
)


class PulseLoop:
    """Runs as an asyncio task, periodically triggering pulse reviews."""

    def __init__(self, interval_minutes: int,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._interval = interval_minutes * 60  # convert to seconds
        self._sub_sessions = sub_session_manager
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("Pulse loop started (interval=%dm)", self._interval // 60)
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            await self._review_global()

    def stop(self) -> None:
        self._running = False

    async def _review_global(self) -> None:
        """
        Global pulse review — fire-and-forget isolated sub-session.

        Uses 'full' system prompt so the worker has access to MEMORIES,
        pulse, and SKILLS — the same context it would have in a normal
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
            logger.warning("Global pulse: SubSessionManager not available, skipping")
