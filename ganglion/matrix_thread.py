"""
Matrix Interface Thread

Connects to the Matrix homeserver, listens for messages in all joined/allowed
rooms, and routes them to the LLM thread.  Each room is its own conversation
thread.

End-to-end encryption is enabled automatically when matrix-nio[e2e] is
installed (which it is).  The Olm/Megolm session store is persisted to
data/matrix_store/ so keys survive restarts.

Special commands handled directly (before reaching the LLM):
  /new       - reset the conversation for the current room
  /compact   - force context compaction for the current room
  /reminders - list active reminders
  /heartbeat - manually trigger heartbeat review
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nio import (
    AsyncClient,
    AsyncClientConfig,
    EncryptedToDeviceEvent,
    InviteMemberEvent,
    MatrixRoom,
    MegolmEvent,
    RoomEncryptionEvent,
    RoomMessageText,
    SyncError,
    ToDeviceError,
)

logger = logging.getLogger(__name__)

RECONNECT_DELAY_MIN = 5    # seconds
RECONNECT_DELAY_MAX = 300  # 5 minutes
STORE_DIR = Path("data/matrix_store")


@dataclass
class MatrixConfig:
    homeserver: str
    user_id: str
    access_token: str
    device_id: str = ""
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
        """Send a message to a specific room, encrypted if the room requires it."""
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        if room_id is None:
            logger.warning("send_message called without room_id, dropping message")
            return
        async with self._send_lock:
            try:
                content = {
                    "msgtype": "m.text",
                    "body":    text,
                    "format":  "org.matrix.custom.html",
                    "formatted_body": _markdown_to_html(text),
                }
                await self._client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content=content,
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
        # Client is closed by the finally block in _connect_and_sync.

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_sync(self) -> None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        client = None

        cfg = AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
            store_sync_tokens=True,
            encryption_enabled=True,
        )
        client = AsyncClient(
            homeserver=self._cfg.homeserver,
            user=self._cfg.user_id,
            device_id=self._cfg.device_id or None,
            store_path=str(STORE_DIR),
            config=cfg,
        )
        client.access_token = self._cfg.access_token
        client.user_id = self._cfg.user_id

        # Load (or create) the Olm account and session store from disk.
        # Required when bypassing client.login() and setting access_token directly.
        await client.load_store()

        # Register event callbacks
        client.add_event_callback(self._on_message, RoomMessageText)
        client.add_event_callback(self._on_encrypted_message, MegolmEvent)
        client.add_event_callback(self._on_invite, InviteMemberEvent)
        client.add_event_callback(self._on_room_encryption, RoomEncryptionEvent)
        client.add_to_device_callback(self._on_to_device, EncryptedToDeviceEvent)

        self._client = client

        try:
            logger.info("Connecting to Matrix homeserver %s (E2EE enabled)", self._cfg.homeserver)

            # Load the local key store and upload keys to the server if needed.
            if client.should_upload_keys:
                logger.info("Uploading encryption keys to homeserver")
                await client.keys_upload()

            # Initial sync to get current state and mark old events as seen.
            response = await client.sync(timeout=30_000, full_state=True)
            if isinstance(response, SyncError):
                raise ConnectionError(f"Initial sync failed: {response.message}")

            # Query and claim missing keys for any known devices.
            await client.keys_query()

            logger.info("Matrix connected with E2EE. Listening in all allowed rooms.")

            # Long-poll sync loop.
            await client.sync_forever(
                timeout=30_000,
                full_state=False,
                loop_sleep_time=100,
            )
        finally:
            await client.close()

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
    # Encryption lifecycle callbacks
    # ------------------------------------------------------------------

    async def _on_room_encryption(self, room: MatrixRoom,
                                  event: RoomEncryptionEvent) -> None:
        """Room has enabled encryption — share our keys with all members."""
        logger.info("Room %s enabled encryption, sharing group session keys", room.room_id)
        try:
            await self._client.joined_members(room.room_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch members for key sharing in %s: %s",
                           room.room_id, exc)

    async def _on_to_device(self, event: EncryptedToDeviceEvent) -> None:
        """Handle incoming to-device messages (key exchanges, etc.)."""
        logger.debug("Received to-device event: %s", type(event).__name__)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _on_encrypted_message(self, room: MatrixRoom, event: MegolmEvent) -> None:
        """
        Received an encrypted message that nio could not decrypt.
        This usually means we're missing the session key — request it.
        """
        if event.sender == self._cfg.user_id:
            return

        logger.warning(
            "Could not decrypt message from %s in %s (session %s) — requesting key",
            event.sender, room.room_id, event.session_id,
        )
        try:
            await self._client.request_room_key(event, self._cfg.user_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Key request failed: %s", exc)

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

        await self._dispatch(text, thread_id)

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, text: str, thread_id: str) -> None:
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
