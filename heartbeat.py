"""
Heartbeat Review Loop

Periodically invokes the LLM to review HEARTBEATS.txt and take autonomous
actions.  The interval is configurable via config.yaml.
"""

import asyncio
import logging

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

    def __init__(self, interval_minutes: int, llm_enqueue_fn, matrix_send_fn) -> None:
        self._interval = interval_minutes * 60  # convert to seconds
        self._llm_enqueue = llm_enqueue_fn   # async callable(text: str) -> str
        self._matrix_send = matrix_send_fn   # async callable(text: str)
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
            await self._review()

    def stop(self) -> None:
        self._running = False

    async def _review(self) -> None:
        logger.info("Running scheduled heartbeat review")
        try:
            reply = await self._llm_enqueue(HEARTBEAT_REVIEW_PROMPT)
            # Only send to Matrix if the AI actually did something.
            if reply and not _is_silent(reply):
                await self._matrix_send(f"💓 Heartbeat review:\n\n{reply}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Heartbeat review failed: %s", exc)


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
