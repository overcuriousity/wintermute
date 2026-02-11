"""
Matrix Interface Thread

Connects to the Matrix homeserver, listens for messages in the configured
room, and routes them to the LLM thread.  All AI replies and reminder
notifications are sent through the ``send_message`` coroutine.

Special commands handled directly (before reaching the LLM):
  /new       – reset the conversation
  /compact   – force context compaction
  /reminders – list active reminders
  /heartbeat – manually trigger heartbeat review
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from nio import (
    AsyncClient,
    AsyncClientConfig,
    LoginResponse,
    MatrixRoom,
    RoomMessageText,
    SyncError,
)

logger = logging.getLogger(__name__)

RECONNECT_DELAY_MIN = 5   # seconds
RECONNECT_DELAY_MAX = 300  # 5 minutes


@dataclass
class MatrixConfig:
    homeserver: str
    user_id: str
    access_token: str
    room_id: str


class MatrixThread:
    """
    Runs as an asyncio task.  After construction, call ``run()``.
    ``send_message`` may be called from any task in the same event loop.
    """

    def __init__(self, config: MatrixConfig, llm_thread) -> None:
        self._cfg = config
        self._llm = llm_thread
        self._client: Optional[AsyncClient] = None
        self._running = False
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send_message(self, text: str) -> None:
        """Send a plain-text message to the configured room."""
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        async with self._send_lock:
            try:
                await self._client.room_send(
                    room_id=self._cfg.room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body":    text,
                        "format":  "org.matrix.custom.html",
                        "formatted_body": _markdown_to_html(text),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to send Matrix message: %s", exc)

    # ------------------------------------------------------------------
    # Main loop with reconnect
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        delay = RECONNECT_DELAY_MIN

        while self._running:
            try:
                await self._connect_and_sync()
                delay = RECONNECT_DELAY_MIN  # reset on clean disconnect
            except Exception as exc:  # noqa: BLE001
                logger.error("Matrix connection error: %s – retrying in %ds", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_DELAY_MAX)

    def stop(self) -> None:
        self._running = False
        if self._client:
            asyncio.create_task(self._client.close())

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_sync(self) -> None:
        cfg = AsyncClientConfig(max_limit_exceeded=0, max_timeouts=0)
        client = AsyncClient(
            homeserver=self._cfg.homeserver,
            user=self._cfg.user_id,
            config=cfg,
        )
        client.access_token = self._cfg.access_token
        client.user_id = self._cfg.user_id
        self._client = client

        client.add_event_callback(self._on_message, RoomMessageText)

        logger.info("Connecting to Matrix homeserver %s", self._cfg.homeserver)

        # Initial sync to get current state and mark old events as seen.
        response = await client.sync(timeout=30_000, full_state=True)
        if isinstance(response, SyncError):
            raise ConnectionError(f"Initial sync failed: {response.message}")

        logger.info("Matrix connected. Listening in room %s", self._cfg.room_id)

        # Long-poll sync loop.
        await client.sync_forever(timeout=30_000, full_state=False)

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        # Only process messages from our configured room.
        if room.room_id != self._cfg.room_id:
            return

        # Ignore our own messages.
        if event.sender == self._cfg.user_id:
            return

        text = event.body.strip()
        if not text:
            return

        logger.info("Received message from %s: %s", event.sender, text[:100])

        # -- Special commands --
        if text == "/new":
            await self._llm.reset_session()
            await self.send_message("Session reset. Starting fresh conversation.")
            return

        if text == "/compact":
            await self._llm.force_compact()
            await self.send_message("Context compaction complete.")
            return

        if text == "/reminders":
            import tools as tool_module
            result = tool_module.execute_tool("list_reminders", {})
            await self.send_message(f"Reminders:\n```json\n{result}\n```")
            return

        if text == "/heartbeat":
            await self._llm.enqueue_system_event(
                "The user manually triggered a heartbeat review. "
                "Review your HEARTBEATS.txt and report what actions, if any, you take."
            )
            await self.send_message("Heartbeat review triggered.")
            return

        # -- Normal message: route to LLM --
        reply = await self._llm.enqueue_user_message(text)
        await self.send_message(reply)


# ---------------------------------------------------------------------------
# Minimal Markdown → HTML conversion for Matrix formatted_body
# ---------------------------------------------------------------------------

def _markdown_to_html(text: str) -> str:
    """Very small Markdown subset: bold, italic, code blocks, inline code."""
    import re
    # Code blocks
    text = re.sub(r"```(\w*)\n(.*?)```", r"<pre><code>\2</code></pre>", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Newlines
    text = text.replace("\n", "<br>")
    return text
