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

SAS key verification:
  Incoming verification requests from allowed_users are auto-accepted and
  auto-confirmed so the bot's device is marked as verified in Element.
  The SAS emojis are logged at INFO level for manual comparison if desired.
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
from nio.events import (
    KeyVerificationCancel,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    UnknownToDeviceEvent,
)
from nio.event_builders import ToDeviceMessage

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
                # Trust any new devices that appeared since last send.
                self._trust_allowed_devices()
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
        if not self._cfg.device_id:
            raise ValueError(
                "matrix.device_id is required. "
                "Obtain it from the login API response or from Element: "
                "Settings → Security & Privacy → Session list → Session ID."
            )
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
        # Note: load_store() is synchronous in matrix-nio.
        client.load_store()

        # Warn if the Olm store's device_id differs from the configured one.
        # This indicates a mismatch: the access_token belongs to a different device
        # than the one whose Olm keys are in the store. E2EE and verification will
        # not work correctly in this state.
        # When device_id is not configured (empty string), nio assigns one
        # automatically — that is fine and requires no warning.
        if self._cfg.device_id and client.device_id != self._cfg.device_id:
            logger.warning(
                "Device ID mismatch: config says '%s' but Olm store identifies as '%s'. "
                "Delete data/matrix_store/ and restart to re-register with the configured device.",
                self._cfg.device_id, client.device_id,
            )
        logger.info("Running as device_id=%s (configured: '%s')",
                    client.device_id, self._cfg.device_id or "<auto>")

        # Register event callbacks
        client.add_event_callback(self._on_message, RoomMessageText)
        client.add_event_callback(self._on_encrypted_message, MegolmEvent)
        client.add_event_callback(self._on_invite, InviteMemberEvent)
        client.add_event_callback(self._on_room_encryption, RoomEncryptionEvent)
        client.add_to_device_callback(self._on_to_device, EncryptedToDeviceEvent)
        client.add_to_device_callback(self._on_unknown_to_device, UnknownToDeviceEvent)
        client.add_to_device_callback(self._on_verification_start, KeyVerificationStart)
        client.add_to_device_callback(self._on_verification_key, KeyVerificationKey)
        client.add_to_device_callback(self._on_verification_mac, KeyVerificationMac)
        client.add_to_device_callback(self._on_verification_cancel, KeyVerificationCancel)

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

            # Query keys for any devices that need it (skip if none pending).
            if client.users_for_key_query:
                await client.keys_query()

            # Proactively fetch device keys for the bot's own account and all
            # allowed users. Without this, nio refuses to create SAS objects for
            # "unknown devices" if a verification request arrives before the
            # normal key-query cycle has run.
            # keys_query() in this version of nio has no parameters — it queries
            # whatever is in client.olm.users_for_key_query. We add our users
            # to that set manually before calling it.
            if client.olm:
                users_to_prime = {self._cfg.user_id} | set(self._cfg.allowed_users)
                client.olm.users_for_key_query.update(users_to_prime)
            if client.users_for_key_query:
                await client.keys_query()

            # Auto-trust all known devices of allowed users so we can send to them.
            self._trust_allowed_devices()

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
    # SAS key verification
    # ------------------------------------------------------------------

    async def _on_unknown_to_device(self, event: UnknownToDeviceEvent) -> None:
        """
        Handle to-device event types that matrix-nio 0.25 doesn't model natively.

        Modern Element uses a request/ready handshake before starting SAS:
          1. Element sends  m.key.verification.request  (→ UnknownToDeviceEvent)
          2. Bot replies    m.key.verification.ready
          3. Element sends  m.key.verification.start    (→ KeyVerificationStart, handled below)
          4. … SAS flow …
          5. Element sends  m.key.verification.done     (→ UnknownToDeviceEvent, just logged)

        Without step 2 the SAS flow never starts and verification silently stalls.

        Verification requests from the bot's own account (self._cfg.user_id) are also
        accepted — this covers cross-device self-verification flows where Element routes
        the request/start via the bot's own Matrix account.
        """
        content = event.source.get("content", {})
        tx_id = content.get("transaction_id")

        if event.type == "m.key.verification.request":
            if not self._is_user_allowed(event.sender) and event.sender != self._cfg.user_id:
                return
            # Fetch device keys for the sender now, before the subsequent
            # m.key.verification.start arrives. nio will refuse to create a SAS
            # object for a device it hasn't seen, so we must ensure the device
            # store is populated before that event is processed.
            if self._client.olm:
                self._client.olm.users_for_key_query.add(event.sender)
            if self._client.users_for_key_query:
                await self._client.keys_query()
            from_device = content.get("from_device", "")
            logger.info(
                "SAS verification request from %s (device=%s tx=%s) — sending ready",
                event.sender, from_device, tx_id,
            )
            await self._send_to_device(
                event_type="m.key.verification.ready",
                recipient=event.sender,
                recipient_device=from_device,
                content={
                    "from_device": self._client.device_id,
                    "methods": ["m.sas.v1"],
                    "transaction_id": tx_id,
                },
            )

        elif event.type == "m.key.verification.done":
            logger.info("SAS done confirmed by %s (tx=%s)", event.sender, tx_id)

    async def _send_to_device(self, event_type: str, recipient: str,
                               recipient_device: str, content: dict) -> None:
        """Send a raw to-device message via the Matrix client."""
        msg = ToDeviceMessage(
            type=event_type,
            recipient=recipient,
            recipient_device=recipient_device,
            content=content,
        )
        try:
            response = await self._client.to_device(msg)
            if isinstance(response, ToDeviceError):
                logger.error("to-device %s failed: %s", event_type, response.message)
        except Exception as exc:  # noqa: BLE001
            logger.error("to-device %s error: %s", event_type, exc)

    async def _on_verification_start(self, event: KeyVerificationStart) -> None:
        """Auto-accept SAS key verification requests from allowed users.

        Also accepts starts from the bot's own account (self._cfg.user_id) to
        support cross-device self-verification flows where Element routes the
        start event via the bot's own Matrix account rather than the initiating
        user's account.
        """
        sender_ok = self._is_user_allowed(event.sender) or event.sender == self._cfg.user_id
        if not sender_ok:
            logger.warning("Ignoring verification request from non-allowed user %s", event.sender)
            await self._client.cancel_key_verification(event.transaction_id)
            return
        logger.info("SAS verification requested by %s (tx=%s) — accepting",
                    event.sender, event.transaction_id)
        try:
            response = await self._client.accept_key_verification(event.transaction_id)
            if isinstance(response, ToDeviceError):
                logger.error("Failed to accept verification (tx=%s): %s",
                             event.transaction_id, response.message)
        except Exception as exc:  # noqa: BLE001
            # Typically "transaction does not exist" — happens when nio didn't
            # recognise the device during the start event (device store stale).
            logger.error("accept_key_verification failed (tx=%s): %s",
                         event.transaction_id, exc)

    async def _on_verification_key(self, event: KeyVerificationKey) -> None:
        """Log the SAS emojis and auto-confirm."""
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None:
            return
        try:
            emojis = sas.get_emoji()
            lines = ["SAS emojis — compare these on both devices:"]
            lines += [f"  {e[0]}  {e[1]}" for e in emojis]
            logger.info("\n".join(lines))
        except Exception:  # noqa: BLE001
            pass
        response = await self._client.confirm_short_auth_string(event.transaction_id)
        if isinstance(response, ToDeviceError):
            logger.error("Failed to confirm SAS (tx=%s): %s",
                         event.transaction_id, response.message)
        else:
            logger.info("SAS confirmation sent (tx=%s)", event.transaction_id)

    async def _on_verification_mac(self, event: KeyVerificationMac) -> None:
        """Check whether verification completed, update local trust, send done."""
        sas = self._client.key_verifications.get(event.transaction_id)
        if sas is None:
            return
        if sas.verified:
            logger.info("Device %s of %s successfully verified via SAS",
                        sas.other_olm_device.id, event.sender)
            self._trust_allowed_devices()
            await self._send_to_device(
                event_type="m.key.verification.done",
                recipient=event.sender,
                recipient_device=sas.other_olm_device.id,
                content={"transaction_id": event.transaction_id},
            )
        elif sas.canceled:
            logger.warning("SAS verification failed (tx=%s): %s",
                           event.transaction_id, sas.cancel_reason)

    async def _on_verification_cancel(self, event: KeyVerificationCancel) -> None:
        """Log verification cancellations."""
        logger.info("SAS verification canceled by %s (tx=%s): [%s] %s",
                    event.sender, event.transaction_id, event.code, event.reason)

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

        await self._set_typing(thread_id, True)
        try:
            reply = await self._llm.enqueue_user_message(text, thread_id)
        finally:
            await self._set_typing(thread_id, False)
        await self.send_message(reply, thread_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _set_typing(self, room_id: str, typing: bool) -> None:
        """Send a typing notification to a room. Silently ignored if client not ready."""
        if self._client is None:
            return
        try:
            await self._client.room_typing(room_id, typing=typing, timeout=30_000)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Typing notification failed for %s: %s", room_id, exc)

    def _is_user_allowed(self, user_id: str) -> bool:
        """Check if a user is in the allowed_users list (empty = allow all)."""
        if not self._cfg.allowed_users:
            return True
        return user_id in self._cfg.allowed_users

    def _trust_allowed_devices(self) -> None:
        """
        Auto-trust all unverified devices belonging to allowed users.

        matrix-nio refuses to encrypt to unverified devices by default.  For a
        personal assistant this is the wrong behaviour — the owner's devices
        should be trusted automatically.  We call this after key queries and
        before every send so that newly registered devices are picked up without
        requiring a restart.
        """
        if self._client is None:
            return
        # Include the bot's own user_id — nio encrypts the Megolm key to every
        # room member, including the sender itself.
        users = set(self._cfg.allowed_users) | {self._cfg.user_id}
        for user_id in users:
            try:
                for device in self._client.device_store.active_user_devices(user_id):
                    if not device.verified and not device.blacklisted:
                        self._client.verify_device(device)
                        logger.info("Auto-trusted device %s for %s", device.id, user_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not trust devices for %s: %s", user_id, exc)


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
