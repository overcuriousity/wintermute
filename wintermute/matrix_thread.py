"""
Matrix Interface Thread

Connects to the Matrix homeserver via mautrix-python, listens for messages
in all joined/allowed rooms, and routes them to the LLM thread.  Each room
is its own conversation thread.

End-to-end encryption is handled by mautrix's OlmMachine backed by a
SQLite crypto store at data/matrix_crypto.db.  Cross-signing is performed
automatically at startup via generate_recovery_key() so the bot's device
is verified without manual SAS.

Special commands handled directly (before reaching the LLM):
  /new       - reset the conversation for the current room
  /compact   - force context compaction for the current room
  /reminders - list active reminders
  /pulse     - manually trigger pulse review
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mautrix.client import Client, InternalEventType
from mautrix.crypto import OlmMachine
from mautrix.crypto.store import PgCryptoStore
from mautrix.types import (
    EventType,
    MessageType,
    RoomID,
    UserID,
)
from mautrix.util.async_db import Database

logger = logging.getLogger(__name__)

CRYPTO_DB_PATH = Path("data/matrix_crypto.db")
CRYPTO_PICKLE_KEY = "wintermute"


# ---------------------------------------------------------------------------
# State store: MemoryStateStore + the find_shared_rooms method that
# mautrix.crypto.store.StateStore requires.
# ---------------------------------------------------------------------------

from mautrix.client.state_store.memory import MemoryStateStore  # noqa: E402


class _CryptoMemoryStateStore(MemoryStateStore):
    """MemoryStateStore extended with the find_shared_rooms method needed
    by OlmMachine for key-sharing decisions."""

    async def find_shared_rooms(self, user_id: UserID) -> list[RoomID]:
        return [
            room_id
            for room_id, members in self.members.items()
            if user_id in members
        ]


# ---------------------------------------------------------------------------
# Config and thread
# ---------------------------------------------------------------------------

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
        self._client: Optional[Client] = None
        self._crypto_db: Optional[Database] = None
        self._running = False
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send_message(self, text: str, room_id: str = None) -> None:
        """Send a message to a room.  Auto-encrypts if room has E2EE."""
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        if room_id is None:
            logger.warning("send_message called without room_id, dropping message")
            return
        async with self._send_lock:
            try:
                await self._client.send_text(
                    room_id=RoomID(room_id),
                    text=text,
                    html=_markdown_to_html(text),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to send Matrix message to %s: %s", room_id, exc)

    @property
    def joined_room_ids(self) -> set[str]:
        """Return room IDs the client has joined.  Used by the debug API."""
        if self._client is None:
            return set()
        try:
            return {str(r) for r in self._client.state_store.members}
        except Exception:  # noqa: BLE001
            return set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        try:
            client = await self._setup_client()
            self._client = client
            await self._setup_crypto(client)
            logger.info("Matrix connected with E2EE. Listening.")

            # Cross-sign after the first sync so device keys are in the server's
            # registry before sign_own_device() queries them.
            _done = asyncio.Event()

            async def _on_first_sync(**_: object) -> None:
                if not _done.is_set():
                    _done.set()
                    asyncio.create_task(
                        self._ensure_cross_signed(client.crypto),
                        name="cross-sign",
                    )

            client.add_event_handler(InternalEventType.SYNC_SUCCESSFUL, _on_first_sync)

            # client.start() creates a sync task with built-in exponential
            # backoff (5 s → 320 s).  Awaiting the returned task blocks
            # until stop() cancels it or a fatal error occurs.
            client.ignore_initial_sync = True
            await client.start(filter_data=None)
        except asyncio.CancelledError:
            logger.info("Matrix task cancelled")
        except Exception as exc:  # noqa: BLE001
            logger.error("Matrix fatal error: %s", exc, exc_info=True)
        finally:
            if self._client is not None:
                self._client.stop()
                try:
                    await self._client.api.session.close()
                except Exception:  # noqa: BLE001
                    pass
            if self._crypto_db is not None:
                await self._crypto_db.stop()

    def stop(self) -> None:
        self._running = False
        if self._client is not None:
            self._client.stop()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    async def _setup_client(self) -> Client:
        if not self._cfg.device_id:
            raise ValueError(
                "matrix.device_id is required.  "
                "Obtain it from the login API response or from Element: "
                "Settings -> Security & Privacy -> Session list -> Session ID."
            )

        state_store = _CryptoMemoryStateStore()

        client = Client(
            mxid=UserID(self._cfg.user_id),
            device_id=self._cfg.device_id,
            base_url=self._cfg.homeserver,
            token=self._cfg.access_token,
            state_store=state_store,
        )

        # Event handlers
        client.add_event_handler(EventType.ROOM_MESSAGE, self._on_message)
        client.add_event_handler(InternalEventType.INVITE, self._on_invite)
        client.add_event_handler(
            InternalEventType.SYNC_ERRORED, self._on_sync_error,
        )

        logger.info(
            "Running as device_id=%s, homeserver=%s",
            self._cfg.device_id, self._cfg.homeserver,
        )
        return client

    async def _setup_crypto(self, client: Client) -> None:
        CRYPTO_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        abs_db_path = CRYPTO_DB_PATH.resolve()
        db = Database.create(
            f"sqlite:///{abs_db_path}",
            upgrade_table=PgCryptoStore.upgrade_table,
        )
        await db.start()
        self._crypto_db = db

        crypto_store = PgCryptoStore(
            account_id=self._cfg.user_id,
            pickle_key=CRYPTO_PICKLE_KEY,
            db=db,
        )

        olm = OlmMachine(
            client=client,
            crypto_store=crypto_store,
            state_store=client.state_store,
        )
        await olm.load()

        # Wire crypto into the client — enables transparent encrypt/decrypt.
        client.crypto = olm
        client.sync_store = crypto_store

        # Upload identity keys + one-time keys if needed.
        logger.info("Sharing encryption keys with homeserver")
        await olm.share_keys()

        logger.info(
            "Crypto ready for device_id=%s (store=%s)",
            self._cfg.device_id, CRYPTO_DB_PATH,
        )

    async def _ensure_cross_signed(self, olm: OlmMachine) -> None:
        """Cross-sign the bot's own device if not already done.

        Called once after the first sync so device keys are queryable.
        Uses get_own_cross_signing_public_keys() to avoid regenerating keys
        (which would invalidate existing trust) on every restart.
        """
        try:
            existing = await olm.get_own_cross_signing_public_keys()
            if existing is not None:
                logger.info("Cross-signing keys already present on server")
                return
            recovery_key = await olm.generate_recovery_key()
            logger.info(
                "Cross-signing complete.  Recovery key (store securely): %s",
                recovery_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-signing failed (non-fatal, retry on next restart): %s", exc)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_message(self, evt) -> None:
        """Handle m.room.message events (auto-decrypted by mautrix)."""
        if str(evt.sender) == self._cfg.user_id:
            return
        if not self._is_user_allowed(str(evt.sender)):
            return
        if self._cfg.allowed_rooms and str(evt.room_id) not in self._cfg.allowed_rooms:
            return

        # Only handle text messages.
        if getattr(evt.content, "msgtype", None) != MessageType.TEXT:
            return

        text = evt.content.body.strip()
        if not text:
            return

        thread_id = str(evt.room_id)
        logger.info("Received message from %s in %s: %s", evt.sender, thread_id, text[:100])
        await self._dispatch(text, thread_id)

    async def _on_invite(self, evt) -> None:
        """Auto-join rooms when invited by an allowed user."""
        sender = str(evt.sender)
        room_id = str(evt.room_id)

        if not self._is_user_allowed(sender):
            logger.warning("Rejecting invite from non-allowed user %s to %s", sender, room_id)
            return
        if self._cfg.allowed_rooms and room_id not in self._cfg.allowed_rooms:
            logger.warning("Rejecting invite to non-allowed room %s from %s", room_id, sender)
            return

        logger.info("Accepting invite from %s to %s", sender, room_id)
        try:
            await self._client.join_room_by_id(RoomID(room_id))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to join room %s: %s", room_id, exc)

    async def _on_sync_error(self, error=None, **_kwargs) -> None:
        """Log sync errors.  mautrix handles retry/backoff internally."""
        logger.warning("Matrix sync error: %s", error)

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
            from wintermute import tools as tool_module
            result = tool_module.execute_tool("list_reminders", {})
            await self.send_message(f"Reminders:\n```json\n{result}\n```", thread_id)
            return

        if text == "/pulse":
            await self._llm.enqueue_system_event(
                "The user manually triggered a pulse review. "
                "Review your PULSE.txt and report what actions, if any, you take.",
                thread_id,
            )
            await self.send_message("Pulse review triggered.", thread_id)
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
        """Send a typing notification.  Silently ignored if client not ready."""
        if self._client is None:
            return
        try:
            timeout = 30_000 if typing else 0
            await self._client.set_typing(RoomID(room_id), timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Typing notification failed for %s: %s", room_id, exc)

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
