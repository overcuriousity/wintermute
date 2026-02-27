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
  /new            - reset the conversation for the current room
  /compact        - force context compaction for the current room
  /tasks          - list active tasks
  /status         - show system status
  /dream          - trigger a dream cycle
  /config         - view/set per-thread configuration overrides
  /kimi-auth      - authenticate Kimi-Code backend
  /verify-session - send E2EE verification request to allowed_users
  /commands       - list all slash commands
"""

import asyncio
import base64 as _base64
import hashlib as _hashlib
import json as _json
import logging
import mimetypes as _mimetypes
import os as _os
import re as _re
import tempfile as _tempfile
import threading as _threading
from collections.abc import MutableMapping as _Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML as _YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as _DQStr

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


_config_write_lock = _threading.Lock()


def _update_config_yaml(access_token: str, device_id: str) -> None:
    """Write new access_token and device_id back into config.yaml (in-place).

    Uses ruamel.yaml for a comment-preserving round-trip parse that preserves
    comments and key order while minimizing formatting changes.  Writes via a
    tempfile + os.replace to prevent concurrent writes from producing truncated
    YAML.

    Values are stored as double-quoted scalars so that PyYAML's safe_load()
    does not coerce otherwise-ambiguous tokens (e.g. values like 'yes' or
    'null' are read back as strings instead of bool/None).
    If the file cannot be parsed or the matrix section is missing, a warning is
    logged and the function returns without modifying the file.
    """
    if not CONFIG_PATH.exists():
        return

    yaml = _YAML()
    yaml.preserve_quotes = True

    with _config_write_lock:
        try:
            with CONFIG_PATH.open(encoding="utf-8") as f:
                data = yaml.load(f)
        except Exception:
            logger.warning("_update_config_yaml: failed to parse %s — skipping write", CONFIG_PATH, exc_info=True)
            return

        if not isinstance(data, _Mapping):
            logger.warning("_update_config_yaml: %s does not contain a YAML mapping at root — skipping write", CONFIG_PATH)
            return
        if not isinstance(data.get("matrix"), _Mapping):
            logger.warning("_update_config_yaml: 'matrix' section missing or not a mapping in %s — skipping write", CONFIG_PATH)
            return

        # Force double-quoted scalars so PyYAML safe_load() always reads them
        # back as strings regardless of the token value.
        data["matrix"]["access_token"] = _DQStr(access_token)
        data["matrix"]["device_id"] = _DQStr(device_id)

        # Atomic write: temp file in same directory, then os.replace().
        fd, tmp_path = _tempfile.mkstemp(
            dir=str(CONFIG_PATH.parent), suffix=".tmp", prefix=".config_",
        )
        try:
            with _os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                fd = -1  # fdopen took ownership of the descriptor
                yaml.dump(data, f)
            _os.replace(tmp_path, str(CONFIG_PATH))
        except BaseException:
            if fd >= 0:
                _os.close(fd)
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise


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
        self._requested_sessions: dict[str, None] = {}  # ordered dict as LRU set for session IDs
        # Whisper transcription (injected from main.py if enabled).
        self._whisper_client = None
        self._whisper_model: str = ""
        self._whisper_language: str = ""
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    _SEND_FILE_RE = _re.compile(r"\[send_file:(/[^\]\s]+)\]")

    async def send_message(self, text: str, room_id: str = None,
                           _retries: int = 3, _delay: float = 2.0) -> None:
        """Send a message to a room.  Auto-encrypts if room has E2EE.

        Any ``[send_file:/absolute/path]`` markers in *text* are extracted
        and uploaded as separate file/image messages before the text is sent.

        Retries up to *_retries* times on transient failures so that
        system-event broadcasts (sub-session results) are not silently lost.
        """
        if self._client is None:
            logger.warning("send_message called before Matrix client is ready")
            return
        if room_id is None:
            logger.warning("send_message called without room_id, dropping message")
            return

        # Extract and send file markers before the text message.
        file_paths = self._SEND_FILE_RE.findall(text)
        for fpath in file_paths:
            await self._send_file(fpath, room_id)
        text = self._SEND_FILE_RE.sub("", text).strip()

        if not text:
            return  # nothing left after stripping markers

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
                except MUnknownToken:
                    logger.error(
                        "Matrix send to %s: token expired — stopping sync loop for re-auth",
                        room_id,
                    )
                    if self._client is not None:
                        self._client.stop()
                    return
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

    async def _send_file(self, file_path: str, room_id: str,
                         _retries: int = 3, _delay: float = 2.0) -> None:
        """Upload a local file and send it to *room_id* as a file or image."""
        p = Path(file_path)
        if not p.is_file():
            logger.warning("send_file: %s does not exist or is not a file", file_path)
            return

        data = p.read_bytes()
        mime_type = _mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        filename = p.name
        file_size = len(data)

        last_exc: Optional[Exception] = None
        for attempt in range(1, _retries + 1):
            async with self._send_lock:
                try:
                    mxc_uri = await self._client.upload_media(
                        data, mime_type=mime_type, filename=filename,
                    )
                    if mime_type.startswith("image/"):
                        await self._client.send_image(
                            room_id=RoomID(room_id),
                            url=mxc_uri,
                            file_name=filename,
                            info={"mimetype": mime_type, "size": file_size},
                        )
                    else:
                        await self._client.send_file(
                            room_id=RoomID(room_id),
                            url=mxc_uri,
                            file_name=filename,
                            info={"mimetype": mime_type, "size": file_size},
                        )
                    logger.info("Sent file %s (%s, %d bytes) to %s", filename, mime_type, file_size, room_id)
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "Matrix send_file %s to %s failed (attempt %d/%d): %s",
                        file_path, room_id, attempt, _retries, exc,
                    )
            if attempt < _retries:
                await asyncio.sleep(_delay * attempt)
        logger.error("Matrix send_file %s to %s failed after %d attempts: %s",
                     file_path, room_id, _retries, last_exc)

    @property
    def joined_room_ids(self) -> set[str]:
        """Return room IDs the client has joined.  Used by the debug API."""
        if self._client is None:
            return set()
        try:
            # Access rooms through the state store's internal storage
            if isinstance(self._client.state_store, _CryptoMemoryStateStore):
                return {str(r) for r in self._client.state_store.members.keys()}
            return set()
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
            # Don't wipe the crypto store for transient network errors (e.g. HTTP
            # 5xx from the homeserver).  Only wipe when it looks like a local DB
            # corruption / incompatibility issue.
            exc_str = str(exc)
            if exc_str[:3].isdigit():
                # Exception message starts with an HTTP status code → network error,
                # not a stale DB.  Re-raise so the outer retry loop handles it.
                raise
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
                _t1 = asyncio.create_task(
                    self._ensure_cross_signed(client.crypto),
                    name="cross-sign",
                )
                self._background_tasks.add(_t1)
                _t1.add_done_callback(self._background_tasks.discard)
                _t2 = asyncio.create_task(
                    self._accept_pending_invites(),
                    name="accept-pending-invites",
                )
                self._background_tasks.add(_t2)
                _t2.add_done_callback(self._background_tasks.discard)

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
                    _t = asyncio.create_task(
                        self._send_verification_requests(), name="send-verify-request",
                    )
                    self._background_tasks.add(_t)
                    _t.add_done_callback(self._background_tasks.discard)
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
            _t = asyncio.create_task(self._send_verification_requests(), name="send-verify-request")
            self._background_tasks.add(_t)
            _t.add_done_callback(self._background_tasks.discard)

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
            # Derive extension: prefer explicit mimetype, fall back to body suffix, then .ogg.
            mimetype = getattr(getattr(evt.content, "info", None), "mimetype", None) or ""
            _MIME_TO_EXT = {"audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
                            "audio/webm": ".webm", "audio/wav": ".wav", "audio/x-wav": ".wav"}
            ext = _MIME_TO_EXT.get(mimetype.split(";")[0].strip())
            if not ext:
                body = evt.content.body or str(evt.event_id)
                ext = Path(body).suffix or ".ogg"
            filename = f"{evt.event_id}{ext}".replace("$", "").replace(":", "_")
            voice_path = VOICE_DIR / filename
            try:
                VOICE_DIR.mkdir(parents=True, exist_ok=True)
                voice_path.write_bytes(data)
            except OSError:
                logger.exception("Failed to save voice message to %s", voice_path)
                await self._dispatch(reply_prefix + "[Voice message received but could not be saved to disk]", thread_id)
                return
            logger.info("Saved voice message from %s to %s", evt.sender, voice_path)

            # Transcribe via Whisper if configured.
            if self._whisper_client is not None:
                try:
                    from openai import NOT_GIVEN
                    # Convert to WAV via ffmpeg if the audio is not already WAV.
                    # Many backends (including llama.cpp whisper) reject OGG/Opus.
                    audio_data = data
                    audio_filename = filename
                    if not filename.lower().endswith(".wav"):
                        proc = await asyncio.create_subprocess_exec(
                            "ffmpeg", "-i", "pipe:0",
                            "-ar", "16000", "-ac", "1", "-f", "wav",
                            "-loglevel", "error", "pipe:1",
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        wav_bytes, ffmpeg_err = await proc.communicate(input=data)
                        if proc.returncode != 0 or not wav_bytes:
                            logger.error("ffmpeg conversion failed for %s: %s", voice_path, ffmpeg_err.decode())
                        else:
                            audio_data = wav_bytes
                            audio_filename = Path(filename).stem + ".wav"
                    resp = await self._whisper_client.audio.transcriptions.create(
                        file=(audio_filename, audio_data),
                        model=self._whisper_model,
                        language=self._whisper_language or NOT_GIVEN,
                        timeout=60.0,
                    )
                    transcript = resp.text.strip()
                    if not transcript:
                        logger.warning("Whisper returned empty transcript for %s", voice_path)
                        text = reply_prefix + "[Voice message received — transcription was empty (silence?)]"
                    else:
                        logger.info("Whisper transcript (%s): %s", evt.sender, transcript[:120])
                        text = reply_prefix + f"[Transcribed voice message] {transcript}"
                    await self._dispatch(text, thread_id)
                    return
                except Exception:  # noqa: BLE001
                    logger.exception("Whisper transcription failed for %s — falling back to placeholder", voice_path)

            text = reply_prefix + f"[Voice message received: {voice_path}]"
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

        self._requested_sessions[session_id] = None
        # Prevent unbounded growth over long uptimes
        if len(self._requested_sessions) > 500:
            # Discard oldest half — dict preserves insertion order so we
            # evict the genuinely oldest entries first.
            to_remove = list(self._requested_sessions)[:250]
            for k in to_remove:
                self._requested_sessions.pop(k, None)
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
            _t = asyncio.create_task(
                self._client.crypto.request_room_key(
                    room_id=RoomID(room_id),
                    sender_key=sender_key,
                    session_id=SessionID(session_id),
                    from_devices={UserID(sender): device_ids},
                    timeout=30,
                ),
                name=f"key_req_{session_id[:8]}",
            )
            self._background_tasks.add(_t)
            _t.add_done_callback(self._background_tasks.discard)
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

        # MAC info prefix — we are the sender of this MAC message.
        # Per the Matrix spec the info is a plain concatenation (no separators).
        info_pfx = (
            f"MATRIX_KEY_VERIFICATION_MAC"
            f"{our_user}{our_device}"
            f"{their_user}{their_dev}"
            f"{txn_id}"
        )
        our_ed25519 = olm_m.account.signing_key
        our_key_id  = f"ed25519:{our_device}"

        key_mac  = state.sas.calculate_mac_fixed_base64(
            our_ed25519, f"{info_pfx}{our_key_id}",
        )
        keys_mac = state.sas.calculate_mac_fixed_base64(
            our_key_id, f"{info_pfx}KEY_IDS",
        )

        logger.debug(
            "SAS MAC debug: our_user=%s our_device=%s their_user=%s their_dev=%s "
            "ed25519_key=%s key_id=%s key_mac=%s keys_mac=%s",
            our_user, our_device, their_user, their_dev,
            our_ed25519, our_key_id, key_mac, keys_mac,
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
            from wintermute.infra import prompt_loader
            try:
                seed_prompt = prompt_loader.load_seed(self._llm.seed_language)
                await self._llm.enqueue_system_event(seed_prompt, thread_id)
            except Exception:  # noqa: BLE001
                logger.exception("Seed after /new failed (non-fatal)")
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

        if content is None and text == "/tasks":
            from wintermute import tools as tool_module
            result = tool_module.execute_tool("task", {"action": "list"})
            await self.send_message(f"Tasks:\n```json\n{result}\n```", thread_id)
            return

        if content is None and text == "/status":
            await self._handle_status_command(thread_id)
            return

        if content is None and text == "/dream":
            await self._handle_dream_command(thread_id)
            return

        if content is None and text == "/reflect":
            await self._handle_reflect_command(thread_id)
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

        if content is None and text == "/rebuild-index":
            from wintermute.infra import database as db, memory_store
            if not memory_store.is_vector_enabled():
                await self.send_message("Vector memory is not enabled (backend: flat_file).", thread_id)
                return
            await self.send_message("Rebuilding memory index...", thread_id)
            try:
                await db.async_call(memory_store.rebuild)
                st = await db.async_call(memory_store.stats)
                await self.send_message(
                    f"Memory index rebuilt.\n```json\n{_json.dumps(st, indent=2)}\n```", thread_id,
                )
            except Exception as exc:
                await self.send_message(f"Rebuild failed: {exc}", thread_id)
            return

        if content is None and text == "/memory-stats":
            from wintermute.infra import database as db, memory_store
            try:
                st = await db.async_call(memory_store.stats)
                await self.send_message(
                    f"**Memory Store**\n```json\n{_json.dumps(st, indent=2)}\n```", thread_id,
                )
            except Exception as exc:
                await self.send_message(f"Failed to get memory stats: {exc}", thread_id)
            return

        if content is None and text.startswith("/config"):
            await self._handle_config_command(text, thread_id)
            return

        if content is None and text == "/commands":
            await self.send_message(
                "**Wintermute — Slash Commands**\n\n"
                "**Conversation**\n"
                "- `/new` — Wipe history and start a fresh session (also cancels running sub-sessions)\n"
                "- `/compact` — Force context compaction now; shows before/after token counts\n\n"
                "**Autonomy**\n"
                "- `/tasks` — List all active tasks\n"
                "- `/dream` — Run a dream cycle (memory consolidation + task pruning)\n"
                "- `/reflect` — Trigger a reflection cycle; shows findings and self-model update\n\n"
                "**Memory**\n"
                "- `/memory-stats` — Show memory store backend, entry count, and status\n"
                "- `/rebuild-index` — Rebuild the vector memory index from MEMORIES.txt\n\n"
                "**Configuration**\n"
                "- `/config` — Show current resolved config for this thread\n"
                "- `/config <key> <value>` — Set a per-thread override (keys: backend_name, session_timeout_minutes, sub_sessions_enabled, system_prompt_mode)\n"
                "- `/config reset` — Remove all per-thread overrides\n"
                "- `/config reset <key>` — Remove a single per-thread override\n\n"
                "**System**\n"
                "- `/status` — Show runtime status: models, token budget, memory, loops, sub-sessions\n"
                "- `/kimi-auth` — Start Kimi-Code OAuth device-code flow\n"
                "- `/verify-session` — Send E2EE SAS verification request to all allowed users\n"
                "- `/commands` — Show this list",
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
        from wintermute.infra import database as db, prompt_assembler

        lines = ["**Wintermute Status**"]

        # --- LLM backends ---
        lines.append("\n**LLM Backends**")

        def _fmt_pool(label, pool):
            if not pool or not pool.enabled:
                return f"{label}: disabled"
            chain = " → ".join(f"`{b[0].model}`" for b in pool._backends)
            ctx = pool.primary.context_size // 1024
            return f"{label}: {chain} ({ctx}k ctx)"

        lines.append(_fmt_pool("Main", self._llm.main_pool))
        lines.append(_fmt_pool("Compaction", self._llm.compaction_pool))
        lines.append(_fmt_pool("Turing Protocol", self._llm.turing_protocol_pool))
        lines.append(_fmt_pool("NL Translation", self._llm.nl_translation_pool))
        if hasattr(self, "_sub_sessions") and self._sub_sessions:
            lines.append(_fmt_pool("Sub-sessions", self._sub_sessions._pool))
        if hasattr(self, "_dreaming_loop") and self._dreaming_loop:
            lines.append(_fmt_pool("Dreaming", self._dreaming_loop._pool))

        # --- Context budget for current thread ---
        try:
            budget = self._llm.get_token_budget(thread_id)
            lines.append(
                f"\n**Context** (thread: `{thread_id}`)\n"
                f"{budget['total_used']:,} / {budget['total_limit']:,} tokens"
                f" ({budget['pct']}%) — {budget['msg_count']} messages"
            )
        except Exception:  # noqa: BLE001
            pass

        # Queue depth
        qsize = self._llm.queue_size
        if qsize:
            lines.append(f"Queue: {qsize} item(s) pending")

        # --- Memory & knowledge ---
        lines.append("\n**Memory & Knowledge**")
        mem_text = prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or ""
        mem_lines = mem_text.count("\n") + (1 if mem_text.strip() else 0)
        skills_count = len(list(prompt_assembler.SKILLS_DIR.glob("*.md"))) if prompt_assembler.SKILLS_DIR.exists() else 0
        lines.append(f"MEMORIES.txt: {mem_lines} lines ({len(mem_text):,} chars)")
        if skills_count:
            lines.append(f"Skills ({skills_count}):")
            for p in sorted(prompt_assembler.SKILLS_DIR.glob("*.md")):
                lines.append(f"- {p.stem}")
        else:
            lines.append("Skills: none")

        # Active tasks
        try:
            task_items = db.list_tasks("active")
            if task_items:
                lines.append(f"Tasks ({len(task_items)} active):")
                for item in task_items:
                    content = (item["content"] or "")[:80]
                    prio = item.get("priority", "?")
                    sched = f" [{item['schedule_desc']}]" if item.get("schedule_desc") else ""
                    lines.append(f"- [P{prio}] #{item['id']}: {content}{sched}")
            else:
                lines.append("Tasks: none")
        except Exception:  # noqa: BLE001
            pass

        # --- Background loops ---
        lines.append("\n**Background Loops**")
        if hasattr(self, "_dreaming_loop") and self._dreaming_loop:
            state = "running" if self._dreaming_loop._running else "stopped"
            dl_cfg = self._dreaming_loop._cfg
            lines.append(
                f"Dreaming: {state} (nightly at {dl_cfg.hour:02d}:{dl_cfg.minute:02d} UTC,"
                f" model: `{self._dreaming_loop._pool.primary.model}`)"
            )
        if hasattr(self, "_memory_harvest") and self._memory_harvest:
            mh = self._memory_harvest
            state = "running" if getattr(mh, "_running", False) else "stopped"
            threshold = getattr(getattr(mh, "_cfg", None), "message_threshold", "?")
            pending = sum(mh._msg_counts.values()) if hasattr(mh, "_msg_counts") else 0
            in_flight = len(mh._in_flight) if hasattr(mh, "_in_flight") else 0
            extra = f" (threshold: {threshold} msgs"
            if pending:
                extra += f", {pending} pending"
            if in_flight:
                extra += f", {in_flight} in-flight"
            extra += ")"
            lines.append(f"Memory harvest: {state}{extra}")
        if hasattr(self, "_scheduler") and self._scheduler:
            jobs = self._scheduler.list_jobs()
            if jobs:
                lines.append(f"Scheduler jobs ({len(jobs)} active):")
                for j in jobs:
                    lines.append(f"- {j.get('id', '?')}: next {j.get('next_run', '?')}")
            else:
                lines.append("Scheduler jobs: none")

        # --- Reflection loop ---
        if hasattr(self, "_reflection_loop") and self._reflection_loop:
            rl = self._reflection_loop
            state = "running" if getattr(rl, "_running", False) else "stopped"
            cfg = rl._cfg
            pending = getattr(rl, "_completed_count", 0)
            lines.append(
                f"Reflection: {state} (batch every {cfg.batch_threshold} completions,"
                f" {pending}/{cfg.batch_threshold} pending, failure_limit={cfg.consecutive_failure_limit})"
            )

        # --- Self-model ---
        if hasattr(self, "_self_model") and self._self_model:
            sm = self._self_model
            summary = sm.get_summary()
            last_updated = sm._state.get("last_updated")
            last_changes = sm._state.get("last_tuning_changes", [])
            ts_str = ""
            if last_updated:
                from datetime import datetime as _dt, timezone as _tz
                ts_str = " (updated " + _dt.fromtimestamp(last_updated, tz=_tz.utc).strftime("%Y-%m-%d %H:%M UTC") + ")"
            lines.append(f"\n**Self-Model**{ts_str}")
            if summary:
                lines.append(summary)
            else:
                lines.append("No summary yet (runs with next reflection cycle)")
            if last_changes:
                lines.append("Last tuning: " + "; ".join(last_changes))

        # --- Update checker ---
        if hasattr(self, "_update_checker") and self._update_checker:
            uc = self._update_checker
            state = "running" if uc._running else "stopped"
            lines.append(f"\n**Updates**")
            lines.append(f"Update checker: {state} (every {uc._config.interval_hours}h)")
            try:
                msg = await uc.check()
                lines.append(f"Status: {msg}" if msg else "Status: up-to-date")
            except Exception:
                logger.debug("Update check in /status failed", exc_info=True)
                cached = uc.last_result
                if cached:
                    lines.append(f"Status: {cached} (cached)")
                else:
                    lines.append("Status: check failed")

        # --- Sub-sessions ---
        lines.append("\n**Sub-sessions**")
        if hasattr(self, "_sub_sessions") and self._sub_sessions:
            active = self._sub_sessions.list_active()
            if active:
                lines.append(f"{len(active)} running:")
                for s in active:
                    lines.append(f"- `{s['session_id']}` [{s['status']}] {s['objective'][:80]}")
            else:
                lines.append("None active")
            workflows = self._sub_sessions.list_workflows()
            running_wfs = [w for w in workflows if w["status"] == "running"]
            if running_wfs:
                lines.append(f"\n{len(running_wfs)} active workflow(s):")
                for w in running_wfs:
                    nodes_summary = ", ".join(
                        f"{n['node_id']}[{n['status']}]" for n in w["nodes"]
                    )
                    lines.append(f"- `{w['workflow_id']}`: {nodes_summary}")
        else:
            lines.append("Not available")

        await self.send_message("\n".join(lines), thread_id)

    async def _handle_dream_command(self, thread_id: str) -> None:
        from wintermute.workers import dreaming; from wintermute.infra import prompt_assembler

        if not hasattr(self, "_dreaming_loop") or not self._dreaming_loop:
            await self.send_message("Dreaming loop not available.", thread_id)
            return

        dl = self._dreaming_loop
        from wintermute.infra import database as db
        mem_before = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        tasks_before = len(db.list_tasks("active"))
        skills_before = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
        skills_size_before = sum(f.stat().st_size for f in skills_before)

        await self.send_message("Starting dream cycle...", thread_id)
        try:
            report = await dreaming.run_dream_cycle(pool=dl._pool)
        except Exception as exc:
            await self.send_message(f"Dream cycle failed: {exc}", thread_id)
            return

        mem_after = len(prompt_assembler._read(prompt_assembler.MEMORIES_FILE) or "")
        tasks_after = len(db.list_tasks("active"))
        skills_after = sorted(prompt_assembler.SKILLS_DIR.glob("*.md")) if prompt_assembler.SKILLS_DIR.exists() else []
        skills_size_after = sum(f.stat().st_size for f in skills_after)

        # Build phase summary from DreamReport.
        phase_lines = []
        for r in report.results:
            status = "\u2713" if not r.error else "\u2717"
            phase_lines.append(f"  {status} {r.phase_name}: {r.summary}")
        phases_text = "\n".join(phase_lines) if phase_lines else "  (no phases ran)"
        errors_text = f"\nErrors: {', '.join(report.errors)}" if report.errors else ""

        await self.send_message(
            f"Dream cycle complete ({len(report.phases_run)} phases).\n"
            f"MEMORIES.txt: {mem_before} -> {mem_after} chars\n"
            f"Tasks: {tasks_before} -> {tasks_after} active\n"
            f"Skills: {len(skills_before)} -> {len(skills_after)} files, "
            f"{skills_size_before} -> {skills_size_after} bytes\n"
            f"Phases:\n{phases_text}{errors_text}",
            thread_id,
        )

    async def _handle_reflect_command(self, thread_id: str) -> None:
        if not hasattr(self, "_reflection_loop") or not self._reflection_loop:
            await self.send_message("Reflection loop not available.", thread_id)
            return

        rl = self._reflection_loop
        if not rl._cfg.enabled:
            await self.send_message("Reflection loop is disabled by config.", thread_id)
            return

        await self.send_message("Running reflection cycle...", thread_id)
        try:
            findings = await rl._run_rules()
            if findings and rl._pool and rl._pool.enabled:
                await rl._run_analysis(findings)
            if rl._self_model:
                await rl._self_model.update(findings)
            rl._checked_failures.clear()
        except Exception as exc:
            await self.send_message(f"Reflection cycle failed: {exc}", thread_id)
            return

        lines = [f"Reflection cycle complete. {len(findings)} finding(s)."]
        for f in findings:
            action = f" → {f.action_taken}" if f.action_taken else ""
            lines.append(f"- [{f.severity.upper()}] {f.rule}: {f.detail[:120]}{action}")

        if hasattr(self, "_self_model") and self._self_model:
            sm_summary = self._self_model.get_summary()
            if sm_summary:
                lines.append(f"\n**Self-Assessment updated:**\n{sm_summary}")
            tuning = self._self_model._state.get("last_tuning_changes", [])
            if tuning:
                lines.append("Tuning changes: " + "; ".join(tuning))

        await self.send_message("\n".join(lines), thread_id)

    async def _handle_config_command(self, text: str, thread_id: str) -> None:
        """Handle /config, /config <key> <value>, /config reset [<key>]."""
        mgr = getattr(self, "_thread_config_manager", None)
        if mgr is None:
            await self.send_message("Thread configuration is not available.", thread_id)
            return

        parts = text.strip().split(None, 2)  # ["/config", key?, value?]

        # /config — show resolved config with sources
        if len(parts) == 1:
            resolved = mgr.resolve_as_dict(thread_id)
            lines = [f"**Configuration** (thread: `{thread_id}`)\n"]
            for key, info in resolved.items():
                src_tag = f" _(from {info['source']})_" if info["source"] != "default" else ""
                lines.append(f"- `{key}`: **{info['value']}**{src_tag}")
            backends = mgr.get_available_backends()
            if backends:
                lines.append(f"\nAvailable backends: {', '.join(f'`{b}`' for b in sorted(backends))}")
            await self.send_message("\n".join(lines), thread_id)
            return

        # /config reset [<key>]
        if parts[1] == "reset":
            if len(parts) == 3:
                key = parts[2]
                current = mgr.get(thread_id)
                if current is None or getattr(current, key, None) is None:
                    await self.send_message(f"No override set for `{key}` on this thread.", thread_id)
                    return
                try:
                    mgr.set(thread_id, key, None)
                    resolved = mgr.resolve(thread_id)
                    new_val = getattr(resolved, key, "?")
                    await self.send_message(
                        f"Override for `{key}` removed. Effective value: **{new_val}**", thread_id,
                    )
                except (ValueError, AttributeError) as exc:
                    await self.send_message(f"Error: {exc}", thread_id)
            else:
                mgr.reset(thread_id)
                await self.send_message("All per-thread overrides removed.", thread_id)
            return

        # /config <key> <value>
        if len(parts) < 3:
            await self.send_message(
                "Usage: `/config <key> <value>` or `/config reset [<key>]`\n"
                "Keys: `backend_name`, `session_timeout_minutes`, `sub_sessions_enabled`, `system_prompt_mode`",
                thread_id,
            )
            return

        key, value = parts[1], parts[2]
        try:
            mgr.set(thread_id, key, value)
            resolved = mgr.resolve(thread_id)
            new_val = getattr(resolved, key, "?")
            await self.send_message(f"`{key}` set to **{new_val}** for this thread.", thread_id)
        except (ValueError, TypeError) as exc:
            await self.send_message(f"Error: {exc}", thread_id)

    async def _handle_kimi_auth(self, thread_id: str) -> None:
        kimi_client = getattr(self, "_kimi_client", None)
        if kimi_client is None:
            await self.send_message(
                "No kimi-code backend configured. Add a `provider: kimi-code` "
                "entry to inference_backends in config.yaml.",
                thread_id,
            )
            return

        from wintermute.backends import kimi_auth

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
