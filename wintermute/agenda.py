"""
Agenda Review Loop

Periodically invokes the LLM to review active agenda items and take autonomous
actions.  Each thread with active agenda items gets its own sub-session, with
results delivered back to the originating room.  Items without a thread_id
are skipped (legacy / unbound items).
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from wintermute import database

if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

PULSE_REVIEW_PROMPT_THREAD = (
    "This is an automatic agenda review for items bound to this thread. "
    "Use the agenda tool with action 'list' to see current items, "
    "then review each item and take any appropriate actions (set reminders, "
    "update memories, run shell commands, etc.). "
    "Use agenda(action='add', content='...') for new items discovered.\n\n"
    "IMPORTANT rules for completing items:\n"
    "- Do NOT complete items unless you have concrete, verifiable evidence "
    "that the task is fully finished (e.g. a command output confirming it, "
    "or a tool result proving completion).\n"
    "- If an item describes ongoing work, a recurring reminder, or a goal "
    "with no clear completion signal — leave it active.\n"
    "- When completing, you MUST provide a 'reason' explaining the evidence.\n"
    "- When in doubt, leave the item active.\n\n"
    "If you take actions, briefly summarise what you did. "
    "If nothing needs attention right now, respond with exactly [NO_ACTION]."
)


class AgendaLoop:
    """Runs as an asyncio task, periodically triggering agenda reviews."""

    def __init__(self, interval_minutes: int,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._interval = interval_minutes * 60  # convert to seconds
        self._sub_sessions = sub_session_manager
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("Agenda loop started (interval=%dm)", self._interval // 60)
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            await self._review_global()

    def stop(self) -> None:
        self._running = False

    async def _review_global(self) -> None:
        """Spawn per-thread agenda review sub-sessions.

        Each thread with active agenda items gets its own sub-session with
        parent_thread_id set, so results are delivered back to that room.
        Items without thread_id are skipped entirely.
        """
        if self._sub_sessions is None:
            logger.warning("Global agenda: SubSessionManager not available, skipping")
            return

        thread_items = database.get_agenda_thread_ids()
        if not thread_items:
            logger.debug("Agenda review: no active thread-bound items, skipping")
            return

        logger.info("Agenda review: %d thread(s) with active items", len(thread_items))
        for thread_id, count in thread_items:
            logger.info("Agenda review: spawning sub-session for thread %s (%d items)", thread_id, count)
            self._sub_sessions.spawn(
                objective=PULSE_REVIEW_PROMPT_THREAD,
                parent_thread_id=thread_id,
                system_prompt_mode="full",
            )
