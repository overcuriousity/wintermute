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
from mautrix.crypto import OlmMachine
from mautrix.crypto.store import PgCryptoStore
from mautrix.errors import MUnknownToken
from mautrix.types import (
    DeviceID,
    EventType,
    MessageType,
    RoomID,
    UserID,
)
from mautrix.util.async_db import Database

logger = logging.getLogger(__name__)

CRYPTO_DB_PATH = Path("data/matrix_crypto.db")
CRYPTO_MARKER_PATH = Path("data/matrix_signed.marker")
CRYPTO_RECOVERY_KEY_PATH = Path("data/matrix_recovery.key")
CRYPTO_PICKLE_KEY = "wintermute"

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
        self._verifications: dict[str, _SasState] = {}

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

            client.add_event_handler(InternalEventType.SYNC_SUCCESSFUL, _on_first_sync)

            # client.start() creates a sync task with built-in exponential
            # backoff (5 s → 320 s).  Awaiting the returned task blocks
            # until stop() cancels it or a fatal error occurs.
            client.ignore_initial_sync = True
            await client.start(filter_data=None)
        except asyncio.CancelledError:
            logger.info("Matrix task cancelled")
        except MUnknownToken:
            logger.error(
                "Matrix access token is invalid or expired.\n"
                "  Get a new token by logging in again:\n"
                "    curl -s -X POST '%s/_matrix/client/v3/login' \\\n"
                "      -H 'Content-Type: application/json' \\\n"
                "      -d '{\"type\":\"m.login.password\","
                "\"identifier\":{\"type\":\"m.id.user\",\"user\":\"BOT_USER\"},"
                "\"password\":\"BOT_PASSWORD\","
                "\"initial_device_display_name\":\"Wintermute\"}' | python3 -m json.tool\n"
                "  Then update access_token (and device_id if changed) in config.yaml and restart.",
                self._cfg.homeserver,
            )
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

        # SAS verification (m.sas.v1) — to-device events
        if _HAS_OLM:
            client.add_event_handler(_VERIFY_REQUEST, self._on_verify_request)
            client.add_event_handler(_VERIFY_START,   self._on_verify_start)
            client.add_event_handler(_VERIFY_KEY,     self._on_verify_key)
            client.add_event_handler(_VERIFY_MAC,     self._on_verify_mac)
            client.add_event_handler(_VERIFY_CANCEL,  self._on_verify_cancel)

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
            logger.info(
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
                        "Same MSK reused — no re-verification needed."
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
                "  Verify the bot in Element: Settings → Security → Sessions → Verify Session.\n"
                "  The bot will complete the SAS handshake automatically.\n"
                "  All future restarts and DB wipes will reuse this identity automatically.",
                CRYPTO_RECOVERY_KEY_PATH,
            )

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
            await self._client.api.send_to_device(
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

    async def _on_verify_request(self, evt) -> None:
        """m.key.verification.request — accept with ready."""
        c = evt.content
        sender = str(evt.sender)
        if not self._is_user_allowed(sender):
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
        # Snapshot the start content for commitment (must be canonical JSON)
        state.start_content = dict(c) if isinstance(c, dict) else {
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
        await self._send_to_device(_VERIFY_KEY, state.their_user_id, state.their_device_id, {
            "transaction_id": txn_id,
            "key": state.sas.pubkey,
        })
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
        our_ed25519 = olm_m.account.fingerprint
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

        if text == "/fingerprint":
            if self._client is not None and self._client.crypto is not None:
                fp = self._client.crypto.account.fingerprint
                await self.send_message(
                    f"Device fingerprint (Ed25519):\n```\n{fp}\n```\n"
                    "To verify: in Element go to Settings → Security → Sessions, "
                    "select this session and tap **Verify Session**. "
                    "The bot will complete the SAS handshake automatically.",
                    thread_id,
                )
            else:
                await self.send_message("Crypto not initialised yet.", thread_id)
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
