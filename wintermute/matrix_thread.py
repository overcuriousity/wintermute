"""
Matrix Interface Thread

Connects to the Matrix homeserver via mautrix-python, listens for messages
in all joined/allowed rooms, and routes them to the LLM thread.  Each room
is its own conversation thread.

End-to-end encryption is handled by mautrix's OlmMachine backed by a
SQLite crypto store at data/matrix_crypto.db.  The bot's cross-signing
identity is established on first start and persisted via a recovery key
(data/matrix_recovery.key) so DB wipes never require UIA approval again.

SAS (emoji) verification is supported: when an allowed user taps
"Verify Session" in Element the bot completes the m.sas.v1 handshake
automatically, skipping the emoji-comparison step.  After completion the
device shows as verified (green shield) in Element.

Special commands handled directly (before reaching the LLM):
  /new         - reset the conversation for the current room
  /compact     - force context compaction for the current room
  /reminders   - list active reminders
  /pulse       - manually trigger pulse review
  /fingerprint - print the Ed25519 device fingerprint
"""

import asyncio
import base64 as _base64
import hashlib as _hashlib
import json as _json
import logging
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import olm as _olm
    _HAS_OLM = True
except ImportError:
    _olm = None  # type: ignore[assignment]
    _HAS_OLM = False

from mautrix.client import Client, InternalEventType
from mautrix.client.dispatcher import MembershipEventDispatcher
from mautrix.crypto import OlmMachine
from mautrix.crypto.attachments import decrypt_attachment
from mautrix.crypto.store import PgCryptoStore
from mautrix.errors import GroupSessionWithheldError, MUnknownToken
from mautrix.types import (
    DeviceID,
    EventType,
    MessageType,
    RoomID,
    SessionID,
    UserID,
)
from mautrix.util.async_db import Database

logger = logging.getLogger(__name__)

VOICE_DIR = Path("data/voice")
CRYPTO_DB_PATH = Path("data/matrix_crypto.db")
CRYPTO_MARKER_PATH = Path("data/matrix_signed.marker")
CRYPTO_RECOVERY_KEY_PATH = Path("data/matrix_recovery.key")
CRYPTO_PICKLE_KEY = "wintermute"
CONFIG_PATH = Path("config.yaml")


def _update_config_yaml(access_token: str, device_id: str) -> None:
    """Write new access_token and device_id back into config.yaml (in-place).

    Replaces just those two lines, preserving all other content and formatting.
    """
    if not CONFIG_PATH.exists():
        return
    text = CONFIG_PATH.read_text()
    text = _re.sub(
        r"(^\s*access_token:\s*).*$",
        rf'\g<1>"{access_token}"',
        text, count=1, flags=_re.MULTILINE,
    )
    text = _re.sub(
        r"(^\s*device_id:\s*).*$",
        rf'\g<1>"{device_id}"',
        text, count=1, flags=_re.MULTILINE,
    )
    CONFIG_PATH.write_text(text)


# SAS (m.sas.v1) to-device event types
_VERIFY_REQUEST = EventType.find("m.key.verification.request", EventType.Class.TO_DEVICE)
_VERIFY_READY   = EventType.find("m.key.verification.ready",   EventType.Class.TO_DEVICE)
_VERIFY_START   = EventType.find("m.key.verification.start",   EventType.Class.TO_DEVICE)
_VERIFY_ACCEPT  = EventType.find("m.key.verification.accept",  EventType.Class.TO_DEVICE)
_VERIFY_KEY     = EventType.find("m.key.verification.key",     EventType.Class.TO_DEVICE)
_VERIFY_MAC     = EventType.find("m.key.verification.mac",     EventType.Class.TO_DEVICE)
_VERIFY_DONE    = EventType.find("m.key.verification.done",    EventType.Class.TO_DEVICE)
_VERIFY_CANCEL  = EventType.find("m.key.verification.cancel",  EventType.Class.TO_DEVICE)


def _canonical_json(data: dict) -> str:
    return _json.dumps(data, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _v_field(content, key: str, default=""):
    """Get a field from event content regardless of whether it is a dict or typed object."""
    if isinstance(content, dict):
        return content.get(key, default)
    v = getattr(content, key, None)
    return v if v is not None else default


@dataclass
class _SasState:
    """Per-transaction SAS verification state."""
    sas: object                          # _olm.Sas instance (set on start)
    their_user_id: str
    their_device_id: str
    txn_id: str
    start_content: dict = field(default_factory=dict)
    is_initiator: bool = False           # True when wintermute sent the request
    key_sent: bool = False               # True once we have sent our pubkey


def _wipe_crypto_db() -> None:
    """Delete the SQLite crypto store, its WAL/SHM files, and the cross-sign marker."""
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(CRYPTO_DB_PATH) + suffix)
        if p.exists():
            p.unlink()
            logger.info("Removed stale crypto file: %s", p)
    if CRYPTO_MARKER_PATH.exists():
        CRYPTO_MARKER_PATH.unlink()
        logger.info("Removed cross-sign marker")
    # Intentionally NOT deleting CRYPTO_RECOVERY_KEY_PATH — the recovery key
    # is reusable across DB wipes to restore the same cross-signing identity
    # without requiring UIA approval again.


def _extract_uia_url(exc: Exception) -> str | None:
    """Return the interactive-auth approval URL embedded in a mautrix exception, if any.

    When a homeserver requires UIA for cross-signing reset it returns a response
    like: {"params": {"org.matrix.cross_signing_reset": {"url": "https://..."}}}
    mautrix stores the parsed body in exc.data (or exc.body on some versions).
    """
    for attr in ("data", "body"):
        data = getattr(exc, attr, None)
        if isinstance(data, dict):
            for stage_params in data.get("params", {}).values():
                if isinstance(stage_params, dict) and "url" in stage_params:
                    return stage_params["url"]
    return None


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
    access_token: str = ""
    device_id: str = ""
    password: str = ""
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
        self._verifications: dict[str, _SasState] = {}
        self._requested_sessions: set[str] = set()  # UTD: session IDs we've already requested

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send_message(self, text: str, room_id: str = None,
                           _retries: int = 3, _delay: float = 2.0) -> None:
        """Send a message to a room.  Auto-encrypts if room has E2EE.

        Retries up to *_retries* times on transient failures so that
        system-event broadcasts (sub-session results) are not silently lost.
        """
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        if room_id is None:
            logger.warning("send_message called without room_id, dropping message")
            return
        last_exc: Optional[Exception] = None
        for attempt in range(1, _retries + 1):
            async with self._send_lock:
                try:
                    await self._client.send_text(
                        room_id=RoomID(room_id),
                        text=text,
                        html=_markdown_to_html(text),
                    )
                    return  # success
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "Matrix send to %s failed (attempt %d/%d): %s",
                        room_id, attempt, _retries, exc,
                    )
            if attempt < _retries:
                await asyncio.sleep(_delay * attempt)
        logger.error("Matrix send to %s failed after %d attempts: %s",
                     room_id, _retries, last_exc)

    @property
    def joined_room_ids(self) -> set[str]:
        """Return room IDs the client has joined.  Used by the debug API."""
        if self._client is None:
            return set()
        try:
            return {str(r) for r in self._client.state_store.members}
        except Exception:  # noqa: BLE001
            return set()

    async def get_joined_rooms(self) -> set[str]:
        """Fetch joined rooms via the Matrix API (reliable even with ignore_initial_sync)."""
        if self._client is None:
            return set()
        try:
            resp = await self._client.get_joined_rooms()
            return {str(r) for r in resp}
        except Exception:  # noqa: BLE001
            # Fallback to state store
            return self.joined_room_ids

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True

        # Auto-login if no token but password is configured.
        if not self._cfg.access_token and self._cfg.password:
            await self._auto_login()
        elif not self._cfg.access_token:
            logger.error(
                "Matrix: no access_token and no password configured. "
                "Set at least one in config.yaml.",
            )
            return

        # If we have credentials for an existing device but the crypto DB is gone,
        # the server still has OTKs registered for that device from a previous run.
        # A fresh Olm account would generate OTKs with the same IDs → server rejects them.
        # Force a new login so we get a fresh device_id with no server-side OTK conflicts.
        if (self._cfg.access_token and self._cfg.device_id
                and not CRYPTO_DB_PATH.exists() and self._cfg.password):
            logger.info(
                "Crypto DB missing for existing device %s — forcing fresh login "
                "to avoid OTK conflicts with server.",
                self._cfg.device_id,
            )
            _wipe_crypto_db()
            await self._auto_login()

        try:
            await self._connect_and_serve()
        except asyncio.CancelledError:
            logger.info("Matrix task cancelled")
        except MUnknownToken:
            if self._cfg.password:
                logger.info("Token expired — re-authenticating with stored password...")
                await self._cleanup()
                _wipe_crypto_db()
                try:
                    await self._auto_login()
                    await self._connect_and_serve()
                except MUnknownToken:
                    logger.error(
                        "Re-login succeeded but the new token was also rejected. "
                        "Check your homeserver or account status.",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Re-login or reconnect failed: %s", exc, exc_info=True)
            else:
                logger.error(
                    "Matrix access token is invalid or expired.\n"
                    "  Option A: add 'password' to the matrix section of config.yaml\n"
                    "            for automatic re-login.\n"
                    "  Option B: get a new token manually:\n"
                    "    curl -s -X POST '%s/_matrix/client/v3/login' \\\n"
                    "      -H 'Content-Type: application/json' \\\n"
                    "      -d '{\"type\":\"m.login.password\","
                    "\"identifier\":{\"type\":\"m.id.user\",\"user\":\"BOT_USER\"},"
                    "\"password\":\"BOT_PASSWORD\","
                    "\"initial_device_display_name\":\"Wintermute\"}' | python3 -m json.tool\n"
                    "  Then update access_token and device_id in config.yaml and restart.",
                    self._cfg.homeserver,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Matrix fatal error: %s", exc, exc_info=True)
        finally:
            await self._cleanup()

    def stop(self) -> None:
        self._running = False
        if self._client is not None:
            self._client.stop()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_serve(self) -> None:
        """Set up client + crypto, start sync loop.  Blocks until stop().

        Raises MUnknownToken if the access token is invalid.
        """
        client = await self._setup_client()
        self._client = client

        try:
            await self._setup_crypto(client)
        except MUnknownToken:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Crypto setup failed (%s). "
                "Wiping stale crypto store and retrying once...", exc,
            )
            _wipe_crypto_db()
            await self._setup_crypto(client)

        logger.info("Matrix connected with E2EE. Listening.")

        # Cross-sign after the first sync so device keys are in the server's
        # registry before sign_own_device() queries them.
        _done = asyncio.Event()

        async def _on_first_sync(*_args: object, **_kw: object) -> None:
            if not _done.is_set():
                _done.set()
                asyncio.create_task(
                    self._ensure_cross_signed(client.crypto),
                    name="cross-sign",
                )
                asyncio.create_task(
                    self._accept_pending_invites(),
                    name="accept-pending-invites",
                )

        client.add_event_handler(InternalEventType.SYNC_SUCCESSFUL, _on_first_sync)

        client.ignore_initial_sync = True
        await client.start(filter_data=None)

    async def _cleanup(self) -> None:
        """Tear down client and crypto DB so a fresh connect can follow."""
        if self._client is not None:
            self._client.stop()
            try:
                await self._client.api.session.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        if self._crypto_db is not None:
            await self._crypto_db.stop()
            self._crypto_db = None

    async def _auto_login(self) -> None:
        """Login via Matrix password API, update config in memory and on disk."""
        import aiohttp

        hs = self._cfg.homeserver.rstrip("/")
        url = f"{hs}/_matrix/client/v3/login"
        payload = {
            "type": "m.login.password",
            "identifier": {"type": "m.id.user", "user": self._cfg.user_id},
            "password": self._cfg.password,
            "initial_device_display_name": "Wintermute",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()

        if "access_token" not in data:
            raise RuntimeError(
                f"Matrix login failed: {data.get('error', 'unknown error')} "
                f"({data.get('errcode', '')})"
            )

        new_token = data["access_token"]
        new_device = data["device_id"]
        self._cfg.access_token = new_token
        self._cfg.device_id = new_device
        _update_config_yaml(new_token, new_device)
        logger.info(
            "Auto-login successful. device_id=%s, credentials written to config.yaml",
            new_device,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    async def _setup_client(self) -> Client:
        if not self._cfg.device_id:
            raise ValueError(
                "matrix.device_id is required.  "
                "Set 'password' in config.yaml to let Wintermute login automatically, "
                "or obtain it from the login API response."
            )

        state_store = _CryptoMemoryStateStore()

        client = Client(
            mxid=UserID(self._cfg.user_id),
            device_id=self._cfg.device_id,
            base_url=self._cfg.homeserver,
            token=self._cfg.access_token,
            state_store=state_store,
        )

        # Translate m.room.member events into InternalEventType.* (JOIN, INVITE, LEAVE, …)
        client.add_dispatcher(MembershipEventDispatcher)

        # Event handlers
        client.add_event_handler(EventType.ROOM_MESSAGE, self._on_message)
        client.add_event_handler(InternalEventType.INVITE, self._on_invite)
        client.add_event_handler(
            InternalEventType.SYNC_ERRORED, self._on_sync_error,
        )

        # SAS verification (m.sas.v1) — to-device events
        if _HAS_OLM:
            client.add_event_handler(_VERIFY_REQUEST, self._on_verify_request)
            client.add_event_handler(_VERIFY_READY,   self._on_verify_ready)
            client.add_event_handler(_VERIFY_START,   self._on_verify_start)
            client.add_event_handler(_VERIFY_ACCEPT,  self._on_verify_accept)
            client.add_event_handler(_VERIFY_KEY,     self._on_verify_key)
            client.add_event_handler(_VERIFY_MAC,     self._on_verify_mac)
            client.add_event_handler(_VERIFY_CANCEL,  self._on_verify_cancel)
            # UTD (Unable To Decrypt) recovery — request missing Megolm session keys
            client.add_event_handler(EventType.ROOM_ENCRYPTED, self._on_utd_event)

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
        """Cross-sign the bot's own device, keeping the same cross-signing identity
        across restarts and crypto-store wipes.

        State files:
          CRYPTO_MARKER_PATH       — Ed25519 signing key of the last signed identity.
                                     If it matches the current key, device is already
                                     signed and nothing needs to be done.
          CRYPTO_RECOVERY_KEY_PATH — SSSS recovery key from the last successful
                                     generate_recovery_key() call.  Preserved across
                                     DB wipes so the same cross-signing MSK is reused.

        Decision tree:
          1. Marker matches current key → already signed, skip.
          2. Marker missing / mismatch (new Olm identity):
             a. Recovery key file present → verify_with_recovery_key():
                  fetches cross-signing private keys from SSSS, re-signs the new
                  device.  No UIA, no browser interaction — works headlessly.
             b. No recovery key (or SSSS cleared) → generate_recovery_key():
                  generates fresh cross-signing keys, requires UIA on matrix.org
                  the first time.  The new recovery key is saved to disk so future
                  DB wipes use path (a) automatically.
        """
        try:
            current_key = olm.account.signing_key
            logger.debug(
                "Device fingerprint (Ed25519, for manual verification): %s",
                olm.account.fingerprint,
            )

            if CRYPTO_MARKER_PATH.exists():
                if CRYPTO_MARKER_PATH.read_text().strip() == current_key:
                    logger.info("Device already cross-signed.")
                    return
                logger.info(
                    "Signing key changed since last cross-sign "
                    "(crypto store was reset) — re-establishing cross-signing."
                )

            # Path (a): reuse existing cross-signing identity via stored recovery key.
            if CRYPTO_RECOVERY_KEY_PATH.exists():
                stored_key = CRYPTO_RECOVERY_KEY_PATH.read_text().strip()
                try:
                    await olm.verify_with_recovery_key(stored_key)
                    CRYPTO_MARKER_PATH.write_text(current_key)
                    logger.info(
                        "Cross-signing restored from stored recovery key.  "
                        "Same MSK reused — sending verification request to allowed_users "
                        "in case this device has not been verified yet."
                    )
                    # New device detected — ask allowed_users to verify even on path (a),
                    # because the user may not have verified this device before.
                    asyncio.create_task(
                        self._send_verification_requests(), name="send-verify-request",
                    )
                    return
                except Exception as restore_exc:  # noqa: BLE001
                    logger.info(
                        "Stored recovery key unusable (%s) — "
                        "generating fresh cross-signing keys.",
                        restore_exc,
                    )

            # Path (b): generate fresh cross-signing keys (first run or SSSS wiped).
            recovery_key = await olm.generate_recovery_key()
            CRYPTO_RECOVERY_KEY_PATH.write_text(recovery_key)
            CRYPTO_MARKER_PATH.write_text(current_key)
            logger.info(
                "Cross-signing complete.  Recovery key saved to %s\n"
                "  Recovery key: %s\n"
                "  New device identity — sending verification request to allowed_users.\n"
                "  Check your Matrix client for a verification notification and tap Accept.\n"
                "  Wintermute will complete the handshake automatically (no emoji check needed).\n"
                "  All future restarts will reuse this identity automatically.",
                CRYPTO_RECOVERY_KEY_PATH,
                CRYPTO_RECOVERY_KEY_PATH.read_text().strip(),
            )
            # Proactively ask allowed_users to verify — they get an in-app notification.
            asyncio.create_task(self._send_verification_requests(), name="send-verify-request")

        except Exception as exc:  # noqa: BLE001
            approval_url = _extract_uia_url(exc)
            if approval_url:
                logger.warning(
                    "Cross-signing requires interactive approval from your homeserver.\n"
                    "  1. Open this URL in your browser: %s\n"
                    "  2. Approve the cross-signing reset request.\n"
                    "  3. Restart Wintermute.\n"
                    "  If the error persists, also delete data/matrix_crypto.db* and restart.",
                    approval_url,
                )
            else:
                logger.warning(
                    "Cross-signing failed (non-fatal): %s\n"
                    "  If you reset your crypto identity on the server, delete\n"
                    "  data/matrix_crypto.db (and .db-wal, .db-shm) and restart.",
                    exc,
                )

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

        msgtype = getattr(evt.content, "msgtype", None)
        if msgtype not in (MessageType.TEXT, MessageType.IMAGE, MessageType.AUDIO):
            return

        thread_id = str(evt.room_id)
        await self._send_read_receipt(thread_id, evt.event_id)

        # --- Reply context ---
        reply_prefix = ""
        reply_to = evt.content.get_reply_to()
        if reply_to:
            try:
                orig = await self._client.get_event(RoomID(thread_id), reply_to)
                orig_body = getattr(getattr(orig, "content", None), "body", None) or ""
                reply_prefix = f"> {orig.sender}: {orig_body}\n>\n"
            except Exception:  # noqa: BLE001
                logger.debug("Could not fetch replied-to event %s", reply_to)

        # --- Image ---
        if msgtype == MessageType.IMAGE:
            try:
                data = await self._download_media(evt)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to download image from %s", evt.sender)
                return
            mimetype = getattr(getattr(evt.content, "info", None), "mimetype", "image/png") or "image/png"
            b64data = _base64.b64encode(data).decode()
            caption = (evt.content.body or "").strip()
            text_for_db = reply_prefix + (caption or "[image attached]")
            content_parts: list[dict] = []
            if reply_prefix or caption:
                content_parts.append({"type": "text", "text": reply_prefix + caption})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mimetype};base64,{b64data}"},
            })
            logger.info("Received image from %s in %s", evt.sender, thread_id)
            await self._dispatch(text_for_db, thread_id, content=content_parts)
            return

        # --- Audio / voice ---
        if msgtype == MessageType.AUDIO:
            try:
                data = await self._download_media(evt)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to download audio from %s", evt.sender)
                return
            VOICE_DIR.mkdir(parents=True, exist_ok=True)
            body = evt.content.body or str(evt.event_id)
            ext = Path(body).suffix or ".ogg"
            filename = f"{evt.event_id}{ext}".replace("$", "").replace(":", "_")
            voice_path = VOICE_DIR / filename
            voice_path.write_bytes(data)
            text = reply_prefix + f"[Voice message received: {voice_path}]"
            logger.info("Saved voice message from %s to %s", evt.sender, voice_path)
            await self._dispatch(text, thread_id)
            return

        # --- Text ---
        text = (evt.content.body or "").strip()
        if not text:
            return
        text = reply_prefix + text
        logger.info("Received message from %s in %s: %s", evt.sender, thread_id, text[:100])
        await self._dispatch(text, thread_id)

    async def _download_media(self, evt) -> bytes:
        """Download media from a Matrix event, handling E2EE decryption."""
        if evt.content.file:
            data = await self._client.download_media(evt.content.file.url)
            return decrypt_attachment(
                data,
                evt.content.file.key.key,
                evt.content.file.hashes["sha256"],
                evt.content.file.iv,
            )
        return await self._client.download_media(evt.content.url)

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

    async def _accept_pending_invites(self) -> None:
        """Accept room invites that were pending at startup.

        ignore_initial_sync=True causes the first sync's events to be skipped,
        so invites that arrived while wintermute was offline are never dispatched
        to _on_invite.  This method does a one-shot snapshot sync (no since-token,
        zero timeout, minimal filter) to find and accept any pending invites.
        """
        if self._client is None:
            return
        try:
            from mautrix.types import FilterID
            # Inline filter: suppress all joined-room noise.
            # rooms.invite is always returned in full by Matrix servers.
            _filter = FilterID(
                '{"presence":{"not_types":["*"]},'
                '"account_data":{"not_types":["*"]},'
                '"room":{"state":{"not_types":["*"]},'
                '"timeline":{"limit":0},'
                '"ephemeral":{"not_types":["*"]}}}'
            )
            raw = await self._client.sync(timeout=0, filter_id=_filter)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Startup invite scan failed: %s", exc)
            return

        invited: dict = raw.get("rooms", {}).get("invite", {}) if isinstance(raw, dict) else {}
        if not invited:
            return

        logger.info("Found %d pending invite(s) at startup", len(invited))
        for room_id_str, room_data in invited.items():
            room_id = RoomID(room_id_str)
            sender: Optional[str] = None
            for ev in room_data.get("invite_state", {}).get("events", []):
                if (ev.get("type") == "m.room.member"
                        and ev.get("state_key") == self._cfg.user_id):
                    sender = ev.get("sender")
                    break
            if sender and not self._is_user_allowed(sender):
                logger.warning(
                    "Startup: rejecting invite to %s from non-allowed user %s",
                    room_id, sender,
                )
                continue
            if self._cfg.allowed_rooms and str(room_id) not in self._cfg.allowed_rooms:
                logger.warning(
                    "Startup: rejecting invite to non-allowed room %s", room_id,
                )
                continue
            logger.info(
                "Startup: accepting pending invite to %s (inviter: %s)",
                room_id, sender or "unknown",
            )
            try:
                await self._client.join_room_by_id(room_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Startup: failed to join %s: %s", room_id, exc)

    async def _on_utd_event(self, evt) -> None:
        """UTD (Unable To Decrypt) recovery: request missing Megolm session keys.

        mautrix silently drops ROOM_ENCRYPTED events it can't decrypt.  This handler
        intercepts those events, checks whether the session key is absent from the
        crypto store, and sends an m.room_key_request to the sender's known devices.
        A brief notification is also posted to the room so the user knows to resend.

        To-device m.room_key events are processed by the OlmMachine before room
        events within the same sync, so a key shared in the same sync batch will
        already be present in the store when this handler runs (no false positives).
        """
        if str(evt.sender) == self._cfg.user_id:
            return
        if not self._is_user_allowed(str(evt.sender)):
            return
        if self._cfg.allowed_rooms and str(evt.room_id) not in self._cfg.allowed_rooms:
            return
        if self._client is None or self._client.crypto is None:
            return

        try:
            session_id = evt.content.session_id
            sender_key = evt.content.sender_key
        except AttributeError:
            return

        if session_id in self._requested_sessions:
            return

        # Check if the session is already in the crypto store.
        try:
            session = await self._client.crypto.crypto_store.get_group_session(
                evt.room_id, session_id
            )
            if session is not None:
                return  # Session present — decryption will succeed, nothing to do.
        except GroupSessionWithheldError:
            pass  # Session was withheld; fall through to notify the user.
        except Exception:
            return

        self._requested_sessions.add(session_id)
        # Prevent unbounded growth over long uptimes
        if len(self._requested_sessions) > 500:
            # Discard oldest half (sets are unordered, but this is fine —
            # the worst case is a duplicate key request, which is harmless)
            to_remove = list(self._requested_sessions)[:250]
            self._requested_sessions -= set(to_remove)
        sender = str(evt.sender)
        room_id = str(evt.room_id)

        # Request the missing key from the sender's known devices.
        try:
            devices = await self._client.crypto.crypto_store.get_devices(UserID(sender))
        except Exception:
            devices = None

        if devices:
            device_ids = list(devices.keys())
            logger.info(
                "UTD: requesting missing session %.16s for %s from %s (%d device(s))",
                session_id, room_id, sender, len(device_ids),
            )
            asyncio.create_task(
                self._client.crypto.request_room_key(
                    room_id=RoomID(room_id),
                    sender_key=sender_key,
                    session_id=SessionID(session_id),
                    from_devices={UserID(sender): device_ids},
                    timeout=30,
                ),
                name=f"key_req_{session_id[:8]}",
            )
        else:
            logger.warning(
                "UTD: no known devices for %s, cannot request key for session %.16s in %s",
                sender, session_id, room_id,
            )

        await self.send_message(
            "I couldn't decrypt your last message (missing encryption key). "
            "Please try sending `/verify-session` to re-establish trust, then resend your message.",
            room_id,
        )

    async def _on_sync_error(self, error=None, **_kwargs) -> None:
        """Log sync errors.  mautrix handles retry/backoff internally."""
        logger.warning("Matrix sync error: %s", error)

    # ------------------------------------------------------------------
    # SAS (m.sas.v1) interactive verification
    #
    # Flow initiated by the other side (e.g. Element "Verify Session"):
    #   1. Other side sends m.key.verification.request  → we send ready
    #   2. Other side sends m.key.verification.start    → we send accept
    #   3. Other side sends m.key.verification.key      → we send our key
    #   4. Other side confirms emoji and sends mac      → we send mac + done
    # We auto-accept without prompting for emoji confirmation, which is safe
    # because we only respond to verification from allowed_users.
    # ------------------------------------------------------------------

    async def _send_to_device(
        self, event_type: EventType, user_id: str, device_id: str, content: dict,
    ) -> None:
        """Send a single to-device message."""
        try:
            await self._client.send_to_device(
                event_type,
                {UserID(user_id): {DeviceID(device_id): content}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to send to-device %s to %s/%s: %s",
                event_type, user_id, device_id, exc,
            )

    async def _send_verify_cancel(
        self, user_id: str, device_id: str, txn_id: str, code: str, reason: str,
    ) -> None:
        await self._send_to_device(_VERIFY_CANCEL, user_id, device_id, {
            "transaction_id": txn_id,
            "code": code,
            "reason": reason,
        })

    async def _send_verification_requests(self) -> None:
        """Send m.key.verification.request to all allowed_users.

        Called after a fresh device identity is established so the user gets an
        in-app notification ("Wintermute is requesting verification") rather than
        having to discover the unverified-device banner themselves.  The user
        simply taps Accept; wintermute auto-completes the SAS handshake.
        """
        if self._client is None or not self._cfg.allowed_users:
            return
        import time as _time, uuid as _uuid
        for user_id in self._cfg.allowed_users:
            txn_id = str(_uuid.uuid4())
            self._verifications[txn_id] = _SasState(
                sas=None,
                their_user_id=user_id,
                their_device_id="*",
                txn_id=txn_id,
                is_initiator=True,
            )
            content = {
                "from_device": self._cfg.device_id,
                "methods": ["m.sas.v1"],
                "timestamp": int(_time.time() * 1000),
                "transaction_id": txn_id,
            }
            try:
                await self._client.send_to_device(
                    _VERIFY_REQUEST,
                    {UserID(user_id): {DeviceID("*"): content}},
                )
                logger.info("SAS: sent verification request to %s (txn=%s)", user_id, txn_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAS: could not send verification request to %s: %s", user_id, exc)
                del self._verifications[txn_id]

    async def _on_verify_ready(self, evt) -> None:
        """m.key.verification.ready — other side accepted our request; send START."""
        c = evt.content
        txn_id = _v_field(c, "transaction_id")
        state = self._verifications.get(txn_id)
        if state is None or not state.is_initiator:
            return
        their_device = _v_field(c, "from_device")
        state.their_device_id = their_device
        sas = _olm.Sas()
        state.sas = sas
        start_content = {
            "transaction_id": txn_id,
            "from_device": self._cfg.device_id,
            "method": "m.sas.v1",
            "key_agreement_protocols": ["curve25519-hkdf-sha256"],
            "hashes": ["sha256"],
            "message_authentication_codes": ["hkdf-hmac-sha256.v2"],
            "short_authentication_string": ["decimal", "emoji"],
        }
        state.start_content = start_content
        await self._send_to_device(_VERIFY_START, state.their_user_id, their_device, start_content)
        logger.info("SAS: sent start to %s/%s (txn=%s)", state.their_user_id, their_device, txn_id)

    async def _on_verify_accept(self, evt) -> None:
        """m.key.verification.accept — other side accepted our start; send our key."""
        c = evt.content
        txn_id = _v_field(c, "transaction_id")
        state = self._verifications.get(txn_id)
        if state is None or state.sas is None or not state.is_initiator:
            return
        await self._send_to_device(_VERIFY_KEY, state.their_user_id, state.their_device_id, {
            "transaction_id": txn_id,
            "key": state.sas.pubkey,
        })
        state.key_sent = True
        logger.info("SAS: sent key to %s/%s (txn=%s)", state.their_user_id, state.their_device_id, txn_id)

    async def _on_verify_request(self, evt) -> None:
        """m.key.verification.request — accept with ready."""
        c = evt.content
        sender = str(evt.sender)
        # Accept from allowed users AND from the bot's own account
        # (same-account cross-device verification, e.g. Element logged in as the bot).
        if sender != self._cfg.user_id and not self._is_user_allowed(sender):
            return
        txn_id     = _v_field(c, "transaction_id")
        from_dev   = _v_field(c, "from_device")
        methods    = _v_field(c, "methods") or []
        if isinstance(methods, str):
            methods = [methods]
        if "m.sas.v1" not in methods:
            await self._send_verify_cancel(
                sender, from_dev, txn_id,
                "m.unknown_method", "Only m.sas.v1 is supported",
            )
            return
        self._verifications[txn_id] = _SasState(
            sas=None,
            their_user_id=sender,
            their_device_id=from_dev,
            txn_id=txn_id,
        )
        await self._send_to_device(_VERIFY_READY, sender, from_dev, {
            "transaction_id": txn_id,
            "from_device": self._cfg.device_id,
            "methods": ["m.sas.v1"],
        })
        logger.info("SAS: accepted verification request from %s (txn=%s)", sender, txn_id)

    async def _on_verify_start(self, evt) -> None:
        """m.key.verification.start — create SAS, send accept with commitment."""
        c      = evt.content
        sender = str(evt.sender)
        txn_id = _v_field(c, "transaction_id")
        state  = self._verifications.get(txn_id)
        if state is None:
            return
        if _v_field(c, "method") != "m.sas.v1":
            await self._send_verify_cancel(
                sender, state.their_device_id, txn_id,
                "m.unknown_method", "Unknown method",
            )
            del self._verifications[txn_id]
            return

        sas = _olm.Sas()
        state.sas = sas
        # Snapshot the start content for commitment (must be canonical JSON).
        # mautrix typed objects carry extra parsed attributes — use serialize()
        # if available, otherwise fall back to dict() / vars().
        if isinstance(c, dict):
            state.start_content = dict(c)
        elif hasattr(c, "serialize"):
            state.start_content = c.serialize()
        else:
            state.start_content = {
                k: v for k, v in vars(c).items() if not k.startswith("_")
            }
        # commitment = base64(sha256(our_pubkey_b64_string || canonical_json(start)))
        canonical  = _canonical_json(state.start_content)
        commitment = _base64.b64encode(
            _hashlib.sha256((sas.pubkey + canonical).encode()).digest()
        ).decode()

        await self._send_to_device(_VERIFY_ACCEPT, sender, state.their_device_id, {
            "transaction_id": txn_id,
            "key_agreement_protocol": "curve25519-hkdf-sha256",
            "hash": "sha256",
            "message_authentication_code": "hkdf-hmac-sha256.v2",
            "short_authentication_string": ["decimal", "emoji"],
            "commitment": commitment,
        })
        logger.info("SAS: sent accept to %s (txn=%s)", sender, txn_id)

    async def _on_verify_key(self, evt) -> None:
        """m.key.verification.key — set their pubkey, reply with ours."""
        c      = evt.content
        txn_id = _v_field(c, "transaction_id")
        state  = self._verifications.get(txn_id)
        if state is None or state.sas is None:
            return
        their_key = _v_field(c, "key")
        try:
            state.sas.set_their_pubkey(their_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SAS: key exchange failed for %s: %s", txn_id, exc)
            await self._send_verify_cancel(
                state.their_user_id, state.their_device_id, txn_id,
                "m.key_mismatch", str(exc),
            )
            del self._verifications[txn_id]
            return
        # Initiator already sent its key in _on_verify_accept; responder sends it here.
        if not state.key_sent:
            await self._send_to_device(_VERIFY_KEY, state.their_user_id, state.their_device_id, {
                "transaction_id": txn_id,
                "key": state.sas.pubkey,
            })
            state.key_sent = True
        logger.info(
            "SAS: key exchange done with %s (txn=%s) — auto-accepting, awaiting MAC",
            state.their_user_id, txn_id,
        )

    async def _on_verify_mac(self, evt) -> None:
        """m.key.verification.mac — send our MAC, send done, log completion."""
        c      = evt.content
        txn_id = _v_field(c, "transaction_id")
        state  = self._verifications.get(txn_id)
        if state is None or state.sas is None:
            return

        olm_m      = self._client.crypto
        our_user   = self._cfg.user_id
        our_device = self._cfg.device_id
        their_user = state.their_user_id
        their_dev  = state.their_device_id

        # MAC info prefix — we are the sender of this MAC message
        info_pfx = (
            f"MATRIX_KEY_VERIFICATION_MAC"
            f"|{our_user}|{our_device}"
            f"|{their_user}|{their_dev}"
            f"|{txn_id}"
        )
        our_ed25519 = olm_m.account.signing_key
        our_key_id  = f"ed25519:{our_device}"

        key_mac  = state.sas.calculate_mac_fixed_base64(
            our_ed25519, f"{info_pfx}|{our_key_id}",
        )
        keys_mac = state.sas.calculate_mac_fixed_base64(
            our_key_id, f"{info_pfx}|KEY_IDS",
        )

        await self._send_to_device(_VERIFY_MAC, their_user, their_dev, {
            "transaction_id": txn_id,
            "mac":  {our_key_id: key_mac},
            "keys": keys_mac,
        })
        await self._send_to_device(_VERIFY_DONE, their_user, their_dev, {
            "transaction_id": txn_id,
        })
        del self._verifications[txn_id]
        logger.info(
            "SAS: verification complete with %s device %s (txn=%s) — "
            "device should show as verified in Element.",
            their_user, their_dev, txn_id,
        )

    async def _on_verify_cancel(self, evt) -> None:
        """m.key.verification.cancel — clean up state."""
        c      = evt.content
        txn_id = _v_field(c, "transaction_id")
        self._verifications.pop(txn_id, None)
        logger.info(
            "SAS: verification cancelled by %s (txn=%s): %s",
            evt.sender, txn_id, _v_field(c, "reason", "no reason"),
        )

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, text: str, thread_id: str, *, content: list | None = None) -> None:
        if content is None and text == "/new":
            await self._llm.reset_session(thread_id)
            await self.send_message("Session reset. Starting fresh conversation.", thread_id)
            return

        if content is None and text == "/compact":
            before = self._llm.get_token_budget(thread_id)
            await self._llm.force_compact(thread_id)
            after = self._llm.get_token_budget(thread_id)
            await self.send_message(
                f"Context compacted.\n"
                f"Before: {before['total_used']} tokens ({before['msg_count']} msgs, {before['pct']}%)\n"
                f"After: {after['total_used']} tokens ({after['msg_count']} msgs, {after['pct']}%)",
                thread_id,
            )
            return

        if content is None and text == "/reminders":
            from wintermute import tools as tool_module
            result = tool_module.execute_tool("list_reminders", {})
            await self.send_message(f"Reminders:\n```json\n{result}\n```", thread_id)
            return

        if content is None and text == "/pulse":
            await self._llm.enqueue_system_event(
                "The user manually triggered a pulse review. "
                "Review your active pulse items using the pulse tool and report what actions, if any, you take.",
                thread_id,
            )
            await self.send_message("Pulse review triggered.", thread_id)
            return

        if content is None and text == "/status":
            await self._handle_status_command(thread_id)
            return

        if content is None and text == "/dream":
            await self._handle_dream_command(thread_id)
            return

        if content is None and text == "/kimi-auth":
            await self._handle_kimi_auth(thread_id)
            return

        if content is None and text == "/verify-session":
            if not _HAS_OLM or self._client is None or not self._client.crypto:
                await self.send_message("E2EE not available.", thread_id)
                return
            await self.send_message(
                "Sending verification request to all allowed_users...", thread_id,
            )
            await self._send_verification_requests()
            return

        if content is None and text == "/commands":
            await self.send_message(
                "**Available commands:**\n"
                "- `/new` – Reset conversation history\n"
                "- `/compact` – Compact context (summarise old messages)\n"
                "- `/reminders` – List active reminders\n"
                "- `/pulse` – Trigger a pulse review\n"
                "- `/status` – Show system status\n"
                "- `/dream` – Trigger a dream cycle\n"
                "- `/kimi-auth` – Authenticate Kimi-Code backend\n"
                "- `/verify-session` – Send E2EE verification request to allowed_users\n"
                "- `/commands` – Show this list",
                thread_id,
            )
            return

        typing_task = asyncio.create_task(
            self._typing_loop(thread_id), name=f"typing_{thread_id}",
        )
        try:
            reply = await self._llm.enqueue_user_message(text, thread_id, content=content)
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            await self._set_typing(thread_id, False)
        # reply is an LLMReply — only send the visible text to Matrix
        # (reasoning tokens are intentionally omitted to reduce noise).
        await self.send_message(str(reply), thread_id)

    # ------------------------------------------------------------------
    # /status and /dream command helpers
    # ------------------------------------------------------------------

    async def _handle_status_command(self, thread_id: str) -> None:
        import asyncio as _asyncio
        lines = ["**Wintermute Status**\n"]

        # Asyncio tasks
        tasks = sorted(_asyncio.all_tasks(), key=lambda t: t.get_name())
        running_names = [t.get_name() for t in tasks if not t.done()]
        lines.append(f"**Core tasks:** {', '.join(running_names)}\n")

        # Sub-sessions
        if hasattr(self, "_sub_sessions") and self._sub_sessions:
            active = self._sub_sessions.list_active()
            if active:
                lines.append(f"**Active sub-sessions ({len(active)}):**")
                for s in active:
                    lines.append(f"- `{s['session_id']}` [{s['status']}] {s['objective'][:80]}")
            else:
                lines.append("**Sub-sessions:** none active")
            workflows = self._sub_sessions.list_workflows()
            running_wfs = [w for w in workflows if w["status"] == "running"]
            if running_wfs:
                lines.append(f"\n**Active workflows ({len(running_wfs)}):**")
                for w in running_wfs:
                    nodes_summary = ", ".join(
                        f"{n['node_id']}[{n['status']}]" for n in w["nodes"]
                    )
                    lines.append(f"- `{w['workflow_id']}`: {nodes_summary}")
        else:
            lines.append("**Sub-sessions:** not available")

        # Pulse loop
        if hasattr(self, "_pulse_loop") and self._pulse_loop:
            state = "running" if self._pulse_loop._running else "stopped"
            lines.append(f"\n**Pulse loop:** {state} (interval: {self._pulse_loop._interval // 60}m)")

        # Dreaming loop
        if hasattr(self, "_dreaming_loop") and self._dreaming_loop:
            state = "running" if self._dreaming_loop._running else "stopped"
            lines.append(f"**Dreaming loop:** {state} (target: {self._dreaming_loop._cfg.hour:02d}:{self._dreaming_loop._cfg.minute:02d} UTC, model: {self._dreaming_loop._pool.primary.model})")

        # Scheduler
        if hasattr(self, "_scheduler") and self._scheduler:
            reminders = self._scheduler.list_reminders()
            lines.append(f"**Reminders:** {len(reminders.get('active', []))} active")

        await self.send_message("\n".join(lines), thread_id)

    async def _handle_dream_command(self, thread_id: str) -> None:
        from wintermute import dreaming, prompt_assembler

        if not hasattr(self, "_dreaming_loop") or not self._dreaming_loop:
            await self.send_message("Dreaming loop not available.", thread_id)
            return

        dl = self._dreaming_loop
        from wintermute import database as db
        mem_before = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        pulse_before = len(db.list_pulse_items("active"))
        skills_before = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
        skills_size_before = sum(f.stat().st_size for f in skills_before)

        await self.send_message("Starting dream cycle...", thread_id)
        try:
            await dreaming.run_dream_cycle(pool=dl._pool)
        except Exception as exc:
            await self.send_message(f"Dream cycle failed: {exc}", thread_id)
            return

        mem_after = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        pulse_after = len(db.list_pulse_items("active"))
        skills_after = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
        skills_size_after = sum(f.stat().st_size for f in skills_after)

        await self.send_message(
            f"Dream cycle complete.\n"
            f"MEMORIES.txt: {mem_before} -> {mem_after} chars\n"
            f"Pulse items: {pulse_before} -> {pulse_after} active\n"
            f"Skills: {len(skills_before)} -> {len(skills_after)} files, "
            f"{skills_size_before} -> {skills_size_after} bytes",
            thread_id,
        )

    async def _handle_kimi_auth(self, thread_id: str) -> None:
        kimi_client = getattr(self, "_kimi_client", None)
        if kimi_client is None:
            await self.send_message(
                "No kimi-code backend configured. Add a `provider: kimi-code` "
                "entry to inference_backends in config.yaml.",
                thread_id,
            )
            return

        from wintermute import kimi_auth

        async def _broadcast(msg: str) -> None:
            await self.send_message(msg, thread_id)

        try:
            creds = await kimi_auth.run_device_flow(_broadcast)
            kimi_client.update_credentials(creds)
        except Exception as exc:
            await self.send_message(f"Kimi-Code authentication failed: {exc}", thread_id)

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
            logger.warning("Typing notification failed for %s: %s", room_id, exc)

    async def _typing_loop(self, room_id: str) -> None:
        """Continuously refresh the typing indicator every 25 seconds.

        The Matrix typing timeout is 30 s.  By re-sending every 25 s the
        indicator stays visible for the entire duration of inference,
        including multi-step tool-call loops.
        """
        try:
            while True:
                await self._set_typing(room_id, True)
                await asyncio.sleep(25)
        except asyncio.CancelledError:
            pass

    async def _send_read_receipt(self, room_id: str, event_id) -> None:
        """Mark an event as read so the sender sees it was received."""
        if self._client is None:
            return
        try:
            await self._client.send_receipt(RoomID(room_id), event_id, "m.read")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Read receipt failed for %s/%s: %s", room_id, event_id, exc)

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
