"""
Signal Interface Thread

Connects to Signal via signal-cli's JSON-RPC daemon mode (subprocess).
Wintermute spawns and owns the signal-cli process, reading JSON lines from
stdout and writing JSON-RPC requests to stdin.

Features: 1:1 and group messages, file sending, read receipts, typing
indicators, voice message transcription (Whisper), image support (base64
multimodal content).

Thread ID convention:
  - 1:1 chats:  sig_+491234567890
  - Groups:     sig_group_<base64id>

Special commands handled directly (before reaching the LLM):
  /new            - reset the conversation
  /compact        - force context compaction
  /tasks          - list active tasks
  /status         - show system status
  /dream          - trigger a dream cycle
  /commands       - list all slash commands
"""

import asyncio
import base64 as _base64
import json as _json
import logging
import mimetypes as _mimetypes
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VOICE_DIR = Path("data/voice")


def _to_urlsafe_b64(s: str) -> str:
    """Convert standard base64 to URL-safe base64 (no padding)."""
    return s.replace("+", "-").replace("/", "_").rstrip("=")


def _from_urlsafe_b64(s: str) -> str:
    """Convert URL-safe base64 back to standard base64."""
    s = s.replace("-", "+").replace("_", "/")
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return s


@dataclass
class SignalConfig:
    phone_number: str
    signal_cli_path: str = "signal-cli"
    allowed_users: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    group_mode: bool = False
    trust_new_keys: bool = True


class SignalThread:
    """
    Runs as an asyncio task.  After construction, call ``run()``.
    ``send_message`` may be called from any task in the same event loop.
    """

    def __init__(self, config: SignalConfig, llm_thread,
                 *,
                 whisper_client=None,
                 whisper_model: str = "",
                 whisper_language: str = "",
                 slash_handler=None,
                 event_bus=None) -> None:
        self._cfg = config
        self._llm = llm_thread
        self._running = False
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stdin_lock = asyncio.Lock()
        self._rpc_id = 0
        self._pending_rpcs: dict[int, asyncio.Future] = {}
        self._background_tasks: set[asyncio.Task] = set()
        # Whisper transcription (passed from main.py if enabled).
        self._whisper_client = whisper_client
        self._whisper_model: str = whisper_model
        self._whisper_language: str = whisper_language
        # Shared slash-command handler.
        self._slash_handler = slash_handler
        # Subscribe to send_file events from the tool.
        self._event_bus = event_bus
        self._send_file_sub_id: Optional[str] = None
        if event_bus is not None:
            self._send_file_sub_id = event_bus.subscribe("send_file", self._handle_send_file_event)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send_message(self, text: str, thread_id: str,
                           _retries: int = 3, _delay: float = 2.0) -> None:
        """Send a message to a Signal recipient or group."""
        if self._process is None or self._process.stdin is None:
            logger.warning("send_message called before signal-cli is ready")
            return
        if not text.strip():
            return

        params = self._thread_id_to_send_params(thread_id)
        if params is None:
            logger.warning("Cannot parse thread_id for Signal send: %s", thread_id)
            return
        params["message"] = text

        last_exc: Optional[Exception] = None
        for attempt in range(1, _retries + 1):
            try:
                await self._send_jsonrpc("send", params)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Signal send to %s failed (attempt %d/%d): %s",
                    thread_id, attempt, _retries, exc,
                )
            if attempt < _retries:
                await asyncio.sleep(_delay * attempt)
        logger.error("Signal send to %s failed after %d attempts: %s",
                     thread_id, _retries, last_exc)

    async def run(self) -> None:
        """Entry point — validate config, spawn subprocess, receive loop with reconnection."""
        self._running = True

        if not self._cfg.phone_number:
            logger.error("Signal: phone_number is empty. Cannot start.")
            self._running = False
            return

        if self._cfg.group_mode and not self._cfg.allowed_groups:
            logger.error(
                "Configuration error: Signal group_mode is enabled but allowed_groups "
                "is empty. Set allowed_groups in config.yaml and restart."
            )
            self._running = False
            return

        backoff = 1.0
        while self._running:
            try:
                await self._start_daemon()
                await self._receive_loop()
            except asyncio.CancelledError:
                logger.info("Signal task cancelled")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Signal daemon error: %s", exc, exc_info=True)
            finally:
                await self._kill_process()

            if not self._running:
                break

            logger.info("Signal daemon exited — reconnecting in %.0fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._running = False
        if self._event_bus is not None and self._send_file_sub_id is not None:
            self._event_bus.unsubscribe(self._send_file_sub_id)
            self._send_file_sub_id = None
        self._kill_process_sync()

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    async def _start_daemon(self) -> None:
        """Spawn signal-cli daemon subprocess."""
        cmd = [self._cfg.signal_cli_path, "-a", self._cfg.phone_number]
        if self._cfg.trust_new_keys:
            cmd.extend(["--trust-new-identities", "always"])
        cmd.append("jsonRpc")

        logger.info("Starting signal-cli daemon (account=<redacted>)")
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # Read stderr in background (Java warnings)
        _t = asyncio.create_task(self._read_stderr(), name="signal-stderr")
        self._background_tasks.add(_t)
        _t.add_done_callback(self._background_tasks.discard)

    async def _read_stderr(self) -> None:
        """Log signal-cli stderr at DEBUG level."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.warning("signal-cli stderr: %s", line.decode(errors="replace").rstrip())
        except Exception:  # noqa: BLE001
            pass

    async def _receive_loop(self) -> None:
        """Read JSON lines from stdout, dispatch receive events."""
        if self._process is None or self._process.stdout is None:
            return
        while self._running:
            line = await self._process.stdout.readline()
            if not line:
                logger.warning("signal-cli stdout closed (process exited?)")
                break
            try:
                data = _json.loads(line)
            except _json.JSONDecodeError:
                logger.debug("signal-cli non-JSON line: %s", line.decode(errors="replace").rstrip())
                continue

            # Resolve pending RPC futures (request/response correlation)
            rpc_id = data.get("id")
            if rpc_id is not None and rpc_id in self._pending_rpcs:
                future = self._pending_rpcs.pop(rpc_id)
                if not future.done():
                    if "error" in data:
                        future.set_exception(
                            RuntimeError(f"signal-cli RPC error: {data['error']}")
                        )
                    else:
                        future.set_result(data.get("result", {}))
                continue

            # Log unexpected RPC error responses (no matching future)
            if "error" in data:
                logger.warning("signal-cli RPC error (id=%s): %s",
                               rpc_id, data["error"])
                continue

            # JSON-RPC notification (incoming message)
            if "method" in data and data["method"] == "receive":
                params = data.get("params", {})
                _t = asyncio.create_task(self._on_message(params))
                self._background_tasks.add(_t)
                _t.add_done_callback(self._background_tasks.discard)

    async def _send_jsonrpc(self, method: str, params: dict,
                            *, timeout: float = 30.0) -> dict:
        """Send a JSON-RPC request and await the response.

        Raises RuntimeError on RPC-level errors so callers can retry.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("signal-cli process not running")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._stdin_lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id
            self._pending_rpcs[rpc_id] = future
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "id": rpc_id,
                "params": params,
            }
            line = _json.dumps(request) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_rpcs.pop(rpc_id, None)
            raise RuntimeError(f"signal-cli RPC timeout for {method} (id={rpc_id})")

    async def _send_jsonrpc_fire_and_forget(self, method: str, params: dict) -> None:
        """Send a JSON-RPC request without waiting for a response.

        Used for non-critical calls (typing indicators, read receipts).
        """
        if self._process is None or self._process.stdin is None:
            return

        async with self._stdin_lock:
            self._rpc_id += 1
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "id": self._rpc_id,
                "params": params,
            }
            line = _json.dumps(request) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

    async def _kill_process(self) -> None:
        # Cancel all tracked background tasks before killing the process.
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()

        # Cancel any pending RPC futures.
        for future in self._pending_rpcs.values():
            if not future.done():
                future.cancel()
        self._pending_rpcs.clear()

        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass
            self._process = None

    def _kill_process_sync(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _on_message(self, params: dict) -> None:
        """Process an incoming message from signal-cli."""
        envelope = params.get("envelope", params)
        source = envelope.get("source", "")
        source_number = envelope.get("sourceNumber", source)

        # Data message (text, attachments)
        data_msg = envelope.get("dataMessage")
        if data_msg is None:
            return

        # Determine thread_id
        group_info = data_msg.get("groupInfo")
        if group_info:
            group_id = group_info.get("groupId", "")
            safe_group_id = _to_urlsafe_b64(group_id)
            thread_id = f"sig_group_{safe_group_id}"
            if not self._is_group_allowed(group_id):
                return
        else:
            thread_id = f"sig_{source_number}"
            if not self._is_user_allowed(source_number):
                return

        is_group = group_info is not None
        group = self._cfg.group_mode and is_group

        # In group mode, only respond when mentioned.
        if group and not self._is_bot_mentioned(data_msg):
            return
        # Group mode + mentioned: gate on allowed_users (matches Matrix behavior).
        if group and not self._is_user_allowed(source_number):
            logger.debug("Group-mode mention from non-allowed user %s — ignoring", source_number)
            return

        # Send read receipt
        timestamp = data_msg.get("timestamp")
        if timestamp and source_number:
            _t = asyncio.create_task(self._send_read_receipt(source_number, timestamp))
            self._background_tasks.add(_t)
            _t.add_done_callback(self._background_tasks.discard)

        sender_prefix = f"[{source_number}]: " if group else ""

        # --- Attachments ---
        attachments = data_msg.get("attachments", [])
        body = (data_msg.get("message") or "").strip()

        # Process image attachments
        for att in attachments:
            content_type = att.get("contentType", "")
            att_path = att.get("filename") or att.get("id", "")

            if content_type.startswith("image/") and att_path:
                try:
                    att_data = Path(att_path).read_bytes()
                except (OSError, FileNotFoundError):
                    logger.warning("Cannot read Signal attachment: %s", att_path)
                    continue
                b64data = _base64.b64encode(att_data).decode()
                text_for_db = sender_prefix + (body or "[image attached]")
                content_parts: list[dict] = []
                text_part = body
                if group:
                    text_part = sender_prefix + text_part
                    text_part = self._strip_bot_mention(text_part)
                    text_for_db = self._strip_bot_mention(text_for_db)
                if text_part.strip() and text_part.strip() != sender_prefix.strip():
                    content_parts.append({"type": "text", "text": text_part})
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{b64data}"},
                })
                logger.info("Received image from %s in %s", source_number, thread_id)
                await self._dispatch(text_for_db, thread_id, content=content_parts, ephemeral=group)
                return

            # Voice message (audio attachment)
            if content_type.startswith("audio/") and att_path:
                try:
                    audio_data = Path(att_path).read_bytes()
                except (OSError, FileNotFoundError):
                    logger.warning("Cannot read Signal audio attachment: %s", att_path)
                    continue

                # Save voice file
                VOICE_DIR.mkdir(parents=True, exist_ok=True)
                ext = _mimetypes.guess_extension(content_type) or ".ogg"
                filename = f"signal_{source_number}_{timestamp}{ext}"
                voice_path = VOICE_DIR / filename
                try:
                    voice_path.write_bytes(audio_data)
                except OSError:
                    logger.exception("Failed to save voice message to %s", voice_path)
                    await self._dispatch(
                        sender_prefix + "[Voice message received but could not be saved]",
                        thread_id, ephemeral=group,
                    )
                    return
                logger.info("Saved voice message from %s to %s", source_number, voice_path)

                # Transcribe via Whisper if configured
                if self._whisper_client is not None:
                    try:
                        from openai import NOT_GIVEN
                        transcribe_data = audio_data
                        transcribe_filename = filename
                        if not filename.lower().endswith(".wav"):
                            proc = await asyncio.create_subprocess_exec(
                                "ffmpeg", "-i", "pipe:0",
                                "-ar", "16000", "-ac", "1", "-f", "wav",
                                "-loglevel", "error", "pipe:1",
                                stdin=asyncio.subprocess.PIPE,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                            )
                            wav_bytes, ffmpeg_err = await proc.communicate(input=audio_data)
                            if proc.returncode != 0 or not wav_bytes:
                                logger.error("ffmpeg conversion failed for %s: %s",
                                             voice_path, ffmpeg_err.decode())
                            else:
                                transcribe_data = wav_bytes
                                transcribe_filename = Path(filename).stem + ".wav"
                        resp = await self._whisper_client.audio.transcriptions.create(
                            file=(transcribe_filename, transcribe_data),
                            model=self._whisper_model,
                            language=self._whisper_language or NOT_GIVEN,
                            timeout=60.0,
                        )
                        transcript = resp.text.strip()
                        if not transcript:
                            text = sender_prefix + "[Voice message received — transcription was empty (silence?)]"
                        else:
                            logger.info("Whisper transcript (%s): %s", source_number, transcript[:120])
                            text = sender_prefix + f"[Transcribed voice message] {transcript}"
                        if group:
                            text = self._strip_bot_mention(text)
                        await self._dispatch(text, thread_id, ephemeral=group)
                        return
                    except Exception:  # noqa: BLE001
                        logger.exception("Whisper transcription failed for %s", voice_path)

                text = sender_prefix + f"[Voice message received: {voice_path}]"
                await self._dispatch(text, thread_id, ephemeral=group)
                return

        # --- Text ---
        if not body:
            return
        text = sender_prefix + body
        if group:
            text = self._strip_bot_mention(text).strip()
            if not text or text == sender_prefix:
                return

        logger.info("Received message from %s in %s: %s", source_number, thread_id, text[:100])
        await self._dispatch(text, thread_id, ephemeral=group)

    # ------------------------------------------------------------------
    # Dispatch to LLM
    # ------------------------------------------------------------------

    async def _dispatch(self, text: str, thread_id: str, *,
                        content: list | None = None,
                        ephemeral: bool = False) -> None:
        """Handle slash commands first, then enqueue to LLM with typing loop."""
        # Shared slash commands
        if content is None and self._slash_handler is not None:
            async def send_fn(msg: str) -> None:
                await self.send_message(msg, thread_id)
            if await self._slash_handler.dispatch(text, thread_id, send_fn):
                return

        typing_task = asyncio.create_task(
            self._typing_loop(thread_id), name=f"sig_typing_{thread_id}",
        )
        try:
            reply = await self._llm.enqueue_user_message(
                text, thread_id, content=content, ephemeral=ephemeral,
            )
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        await self.send_message(str(reply), thread_id)

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------

    async def _typing_loop(self, thread_id: str) -> None:
        """Send typing indicator every 25 seconds."""
        try:
            while True:
                await self._send_typing(thread_id)
                await asyncio.sleep(25)
        except asyncio.CancelledError:
            pass

    async def _send_typing(self, thread_id: str) -> None:
        """Send a typing indicator to the recipient/group."""
        if self._process is None or self._process.stdin is None:
            return
        params = self._thread_id_to_send_params(thread_id)
        if params is None:
            return
        try:
            await self._send_jsonrpc_fire_and_forget("sendTyping", params)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Typing indicator failed for %s: %s", thread_id, exc)

    # ------------------------------------------------------------------
    # Read receipts
    # ------------------------------------------------------------------

    async def _send_read_receipt(self, sender: str, timestamp: int) -> None:
        """Send a read receipt for a message."""
        try:
            await self._send_jsonrpc_fire_and_forget("sendReceipt", {
                "recipient": [sender],
                "targetTimestamp": [timestamp],
                "type": "read",
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Read receipt failed for %s/%s: %s", sender, timestamp, exc)

    # ------------------------------------------------------------------
    # File sending
    # ------------------------------------------------------------------

    async def _handle_send_file_event(self, event) -> None:
        """EventBus handler for ``send_file`` events — filter by sig_ prefix."""
        data = event.data if hasattr(event, "data") else event
        file_path = data.get("path", "")
        thread_id = data.get("thread_id", "")
        if not file_path or not thread_id:
            logger.warning("send_file event missing path or thread_id: %s", data)
            return
        if not thread_id.startswith("sig_"):
            return
        await self._send_file(file_path, thread_id)

    async def _send_file(self, file_path: str, thread_id: str,
                         _retries: int = 3, _delay: float = 2.0) -> None:
        """Send a file via signal-cli."""
        p = Path(file_path)
        if not p.is_file():
            logger.warning("send_file: %s does not exist or is not a file", file_path)
            return

        params = self._thread_id_to_send_params(thread_id)
        if params is None:
            logger.warning("Cannot parse thread_id for Signal file send: %s", thread_id)
            return
        params["attachment"] = [str(p.resolve())]

        last_exc: Optional[Exception] = None
        for attempt in range(1, _retries + 1):
            try:
                await self._send_jsonrpc("send", params)
                logger.info("Sent file %s to %s", p.name, thread_id)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Signal send_file %s to %s failed (attempt %d/%d): %s",
                    file_path, thread_id, attempt, _retries, exc,
                )
            if attempt < _retries:
                await asyncio.sleep(_delay * attempt)
        logger.error("Signal send_file %s to %s failed after %d attempts: %s",
                     file_path, thread_id, _retries, last_exc)

    # ------------------------------------------------------------------
    # ACL helpers
    # ------------------------------------------------------------------

    def _is_user_allowed(self, phone: str) -> bool:
        """Check if a phone number is in the allowed_users list (empty = allow all)."""
        if not self._cfg.allowed_users:
            return True
        return phone in self._cfg.allowed_users

    def _is_group_allowed(self, group_id: str) -> bool:
        """Check if a group ID is in the allowed_groups list (empty = allow all)."""
        if not self._cfg.allowed_groups:
            return True
        return group_id in self._cfg.allowed_groups

    # ------------------------------------------------------------------
    # Thread ID parsing
    # ------------------------------------------------------------------

    def _thread_id_to_send_params(self, thread_id: str) -> dict | None:
        """Convert a thread_id to signal-cli JSON-RPC send parameters."""
        if thread_id.startswith("sig_group_"):
            safe_group_id = thread_id[len("sig_group_"):]
            group_id = _from_urlsafe_b64(safe_group_id)
            return {"groupId": [group_id]}
        if thread_id.startswith("sig_"):
            recipient = thread_id[len("sig_"):]
            return {"recipient": [recipient]}
        return None

    # ------------------------------------------------------------------
    # Mention detection (group mode)
    # ------------------------------------------------------------------

    def _is_bot_mentioned(self, data_msg: dict) -> bool:
        """Check if the bot's phone number appears in the message body or mentions."""
        phone = self._cfg.phone_number

        # Check signal-cli mention metadata
        mentions = data_msg.get("mentions", [])
        for m in mentions:
            if m.get("number") == phone or m.get("uuid") == phone:
                return True

        # Fallback: check if phone number appears in body text
        body = data_msg.get("message") or ""
        if phone in body:
            return True

        return False

    def _strip_bot_mention(self, text: str) -> str:
        """Remove the bot's phone number from plain text."""
        phone = self._cfg.phone_number
        pattern = rf"(?<!\S){_re.escape(phone)}(?=[\s\.,:;!?)]|$)"
        cleaned = _re.sub(pattern, "", text)
        cleaned = _re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()
