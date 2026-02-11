"""
Matrix Interface Thread

Connects to the Matrix homeserver, listens for messages in all joined/allowed
rooms, and routes them to the LLM thread.  Each room is its own conversation
thread.

Special commands handled directly (before reaching the LLM):
  /new       - reset the conversation for the current room
  /compact   - force context compaction for the current room
  /reminders - list active reminders
  /heartbeat - manually trigger heartbeat review
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
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
    allowed_users: list[str] = field(default_factory=list)
    allowed_rooms: list[str] = field(default_factory=list)


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

    async def send_message(self, text: str, room_id: str = None) -> None:
        """Send a plain-text message to a specific room."""
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        if room_id is None:
            logger.warning("send_message called without room_id, dropping message")
            return
        async with self._send_lock:
            try:
                await self._client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body":    text,
                        "format":  "org.matrix.custom.html",
                        "formatted_body": _markdown_to_html(text),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to send Matrix message to %s: %s", room_id, exc)

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
                logger.error("Matrix connection error: %s - retrying in %ds", exc, delay)
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
        client.add_event_callback(self._on_invite, InviteMemberEvent)

        logger.info("Connecting to Matrix homeserver %s", self._cfg.homeserver)

        # Initial sync to get current state and mark old events as seen.
        response = await client.sync(timeout=30_000, full_state=True)
        if isinstance(response, SyncError):
            raise ConnectionError(f"Initial sync failed: {response.message}")

        logger.info("Matrix connected. Listening in all allowed rooms.")

        # Long-poll sync loop.
        await client.sync_forever(timeout=30_000, full_state=False)

    # ------------------------------------------------------------------
    # Invite handler
    # ------------------------------------------------------------------

    async def _on_invite(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        """Auto-join rooms when invited by an allowed user."""
        if event.membership != "invite":
            return
        if event.state_key != self._cfg.user_id:
            return

        sender = event.sender
        if not self._is_user_allowed(sender):
            logger.warning("Rejecting invite from non-allowed user %s to %s",
                           sender, room.room_id)
            return

        if self._cfg.allowed_rooms and room.room_id not in self._cfg.allowed_rooms:
            logger.warning("Rejecting invite to non-allowed room %s from %s",
                           room.room_id, sender)
            return

        logger.info("Accepting invite from %s to %s", sender, room.room_id)
        try:
            await self._client.join(room.room_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to join room %s: %s", room.room_id, exc)

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        # Ignore our own messages.
        if event.sender == self._cfg.user_id:
            return

        # Check allowed users
        if not self._is_user_allowed(event.sender):
            return

        # Check allowed rooms (if whitelist is set)
        if self._cfg.allowed_rooms and room.room_id not in self._cfg.allowed_rooms:
            return

        text = event.body.strip()
        if not text:
            return

        thread_id = room.room_id
        logger.info("Received message from %s in %s: %s", event.sender, thread_id, text[:100])

        # -- Special commands --
        if text == "/new":
            await self._llm.reset_session(thread_id)
            await self.send_message("Session reset. Starting fresh conversation.", thread_id)
            return

        if text == "/compact":
            await self._llm.force_compact(thread_id)
            await self.send_message("Context compaction complete.", thread_id)
            return

        if text == "/reminders":
            from ganglion import tools as tool_module
            result = tool_module.execute_tool("list_reminders", {})
            await self.send_message(f"Reminders:\n```json\n{result}\n```", thread_id)
            return

        if text == "/heartbeat":
            await self._llm.enqueue_system_event(
                "The user manually triggered a heartbeat review. "
                "Review your HEARTBEATS.txt and report what actions, if any, you take.",
                thread_id,
            )
            await self.send_message("Heartbeat review triggered.", thread_id)
            return

        # -- Normal message: route to LLM --
        reply = await self._llm.enqueue_user_message(text, thread_id)
        await self.send_message(reply, thread_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_user_allowed(self, user_id: str) -> bool:
        """Check if a user is in the allowed_users list (empty = allow all)."""
        if not self._cfg.allowed_users:
            return True
        return user_id in self._cfg.allowed_users


# ---------------------------------------------------------------------------
# Minimal Markdown -> HTML conversion for Matrix formatted_body
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
