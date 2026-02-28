"""
LLMBackend client for Google Cloud Code Assist API.

Implements the :class:`LLMBackend` protocol via ``complete()``, translating
between OpenAI-style kwargs and the Google Cloud Code Assist format.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any

import httpx

from wintermute.backends import gemini_auth
from wintermute.core.types import LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)

CLOUDCODE_BASE = "https://cloudcode-pa.googleapis.com"
CLOUDCODE_STREAM_URL = f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent"

# Retry settings
MAX_RETRIES = 15
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 60.0


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class GeminiCloudClient:
    """LLMBackend client backed by Google Cloud Code Assist."""

    # Max concurrent requests to avoid bursting rate limits
    _MAX_CONCURRENT = 2
    _MIN_REQUEST_INTERVAL = 1.0  # seconds between requests

    def __init__(self, creds: dict):
        self._creds = dict(creds)
        self._http = httpx.AsyncClient(timeout=300)
        self._refresh_lock = asyncio.Lock()
        self._request_sem = asyncio.Semaphore(self._MAX_CONCURRENT)
        self._last_request_time = 0.0

    async def complete(self, **kwargs: Any) -> LLMResponse:
        """Send an inference request and return a normalized LLMResponse."""
        return await self._create(**kwargs)

    async def _ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        if time.time() < self._creds.get("expires_at", 0) - 60:
            return self._creds["access_token"]
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if time.time() < self._creds.get("expires_at", 0) - 60:
                return self._creds["access_token"]
            logger.info("Refreshing Gemini access token")
            loop = asyncio.get_running_loop()
            self._creds = await loop.run_in_executor(
                None, gemini_auth.refresh_access_token, self._creds
            )
            await loop.run_in_executor(
                None, gemini_auth.save_credentials, self._creds
            )
            return self._creds["access_token"]

    # -------------------------------------------------------------------
    # Request translation: OpenAI → Google Cloud Code Assist
    # -------------------------------------------------------------------

    def _translate_request(self, **kwargs) -> dict:
        """Convert OpenAI-style kwargs to a Cloud Code Assist request body."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "gemini-2.5-pro")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens", 8192)

        # Build mappings from tool_call_id -> function name and signature status
        tc_id_to_name: dict[str, str] = {}
        tc_id_has_sig: set[str] = set()
        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            if role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            fn_name = tc.get("function", {}).get("name", "")
                            has_sig = bool(tc.get("thought_signature"))
                        else:
                            tc_id = tc.id
                            fn_name = tc.function.name
                            has_sig = bool(getattr(tc, "thought_signature", None))
                        if tc_id and fn_name:
                            tc_id_to_name[tc_id] = fn_name
                        if tc_id and has_sig:
                            tc_id_has_sig.add(tc_id)

        # Build contents and extract system instruction
        system_parts = []
        contents = []

        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.role
            content = msg.get("content", "") if isinstance(msg, dict) else msg.content

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "assistant":
                google_role = "model"
            else:
                google_role = "user"

            parts = []

            # Handle tool call results
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                fn_name = name or tc_id_to_name.get(tool_call_id, "") or tool_call_id
                if tool_call_id in tc_id_has_sig:
                    # Has matching signed call — send as proper functionResponse
                    parts.append({
                        "functionResponse": {
                            "name": fn_name,
                            "response": {"result": content or ""},
                        }
                    })
                # else: skip unsigned tool results from old history
            # Handle assistant messages with tool calls
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", "{}")
                            thought_sig = tc.get("thought_signature")
                        else:
                            fn_name = tc.function.name
                            fn_args = tc.function.arguments
                            thought_sig = getattr(tc, "thought_signature", None)
                        try:
                            args_dict = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except json.JSONDecodeError as _exc:
                            logger.warning(
                                "Malformed tool args in history for %s: %s",
                                fn_name, _exc,
                            )
                            args_dict = {}
                        if thought_sig:
                            # Has signature — send as proper functionCall
                            parts.append({"functionCall": {
                                "name": fn_name,
                                "args": args_dict,
                                "thoughtSignature": thought_sig,
                            }})
                        # else: skip unsigned tool calls from old history
                if content:
                    parts.append({"text": content})
            else:
                if content:
                    parts.append({"text": content})

            if parts:
                # Merge with previous entry if same role (Gemini requires strict alternation)
                if contents and contents[-1]["role"] == google_role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": google_role, "parts": parts})

        # Gemini requires the first message to have role "user"
        if contents and contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": "(conversation continues)"}]})

        body: dict[str, Any] = {
            "model": model,
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            },
        }

        if system_parts:
            body["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": "\n\n".join(system_parts)}],
            }

        # Translate tools — all declarations go in a single tools entry
        if tools:
            fn_decls = []
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    fn = tool["function"]
                    google_fn = {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                    }
                    if "parameters" in fn:
                        google_fn["parameters"] = fn["parameters"]
                    fn_decls.append(google_fn)
            if fn_decls:
                body["tools"] = [{"functionDeclarations": fn_decls}]

        # Translate tool_choice
        if tool_choice and tools:
            if tool_choice == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            elif tool_choice == "auto":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
            elif tool_choice == "required":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                fn_name = tool_choice.get("function", {}).get("name", "")
                if fn_name:
                    body["toolConfig"] = {
                        "functionCallingConfig": {
                            "mode": "ANY",
                            "allowedFunctionNames": [fn_name],
                        }
                    }

        if self._creds.get("project_id"):
            body["projectId"] = self._creds["project_id"]

        return body

    # -------------------------------------------------------------------
    # Response translation: Google SSE → OpenAI ChatCompletion
    # -------------------------------------------------------------------

    def _translate_response(self, chunks: list[dict]) -> LLMResponse:
        """Assemble Google SSE chunks into an LLMResponse."""
        text_parts = []
        tool_calls: list[LLMToolCall] = []
        finish_reason = "stop"
        tc_counter = 0

        for chunk in chunks:
            # Cloud Code Assist wraps the Gemini response in a "response" envelope
            inner = chunk.get("response", chunk)
            candidates = inner.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if "text" in part:
                        text_parts.append(part["text"])
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tc_counter += 1
                        tool_calls.append(LLMToolCall(
                            id=f"call_{tc_counter}",
                            name=fc.get("name", ""),
                            arguments=json.dumps(fc.get("args", {})),
                            thought_signature=fc.get("thoughtSignature"),
                        ))

                # Map finish reason
                fr = candidate.get("finishReason", "")
                if fr == "MAX_TOKENS":
                    finish_reason = "length"
                elif fr == "STOP":
                    finish_reason = "stop"
                elif fr == "SAFETY":
                    finish_reason = "stop"

        if tool_calls:
            finish_reason = "tool_calls"

        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
        )

    # -------------------------------------------------------------------
    # Core create method
    # -------------------------------------------------------------------

    async def _create(self, **kwargs) -> LLMResponse:
        """Send an inference request to Cloud Code Assist."""
        async with self._request_sem:
            # Enforce minimum interval between requests
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._MIN_REQUEST_INTERVAL:
                await asyncio.sleep(self._MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.time()
            return await self._create_inner(**kwargs)

    async def _create_inner(self, **kwargs) -> LLMResponse:
        """Internal: translate, send, and parse a request."""
        inner = self._translate_request(**kwargs)
        model = inner.pop("model", "gemini-2.5-pro")
        project_id = inner.pop("projectId", None) or self._creds.get("project_id", "")

        # Wrap in the Cloud Code Assist envelope
        body = {
            "model": model,
            "project": project_id,
            "user_prompt_id": str(uuid.uuid4()),
            "request": inner,
        }
        logger.debug("Gemini request: model=%s, contents=%d blocks", model, len(inner.get("contents", [])))
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES + 1):
            token = await self._ensure_valid_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            try:
                resp = await self._http.post(
                    f"{CLOUDCODE_STREAM_URL}?alt=sse",
                    headers=headers,
                    json=body,
                )
            except httpx.TransportError as exc:
                if attempt < MAX_RETRIES:
                    jitter = random.uniform(0, backoff * 0.5)
                    wait = min(backoff + jitter, MAX_BACKOFF)
                    logger.warning("Gemini transport error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, exc)
                    await asyncio.sleep(wait)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                raise

            # 401: refresh token and retry once
            if resp.status_code == 401 and attempt == 0:
                logger.info("Gemini 401 — refreshing token and retrying")
                async with self._refresh_lock:
                    loop = asyncio.get_running_loop()
                    self._creds = await loop.run_in_executor(
                        None, gemini_auth.refresh_access_token, self._creds
                    )
                    await loop.run_in_executor(
                        None, gemini_auth.save_credentials, self._creds
                    )
                continue

            # 429/503: exponential backoff with jitter
            if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    jitter = random.uniform(0, backoff * 0.5)
                    wait = min(backoff + jitter, MAX_BACKOFF)
                logger.warning("Gemini %d — retrying after %.1fs (attempt %d/%d)",
                               resp.status_code, wait, attempt + 1, MAX_RETRIES)
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                raise RuntimeError(
                    f"Gemini Cloud Code Assist API error {resp.status_code}: {error_text}"
                )

            # Parse SSE response
            chunks = self._parse_sse(resp.text)
            if not chunks:
                logger.warning("Gemini: no SSE chunks parsed from response (%d bytes)", len(resp.text))
            # Temporary: log response parts for debugging tool call parsing
            for ci, chunk in enumerate(chunks):
                inner = chunk.get("response", chunk)
                for cand in inner.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        keys = list(part.keys())
                        preview = ""
                        if "text" in part:
                            preview = part["text"][:150]
                        elif "functionCall" in part:
                            preview = f"name={part['functionCall'].get('name', '?')}"
                        logger.debug("Gemini response part [%d]: keys=%s preview=%s", ci, keys, preview)
            return self._translate_response(chunks)

        raise RuntimeError("Gemini request failed after all retries")

    @staticmethod
    def _parse_sse(text: str) -> list[dict]:
        """Parse SSE data: lines into JSON chunks."""
        chunks = []
        for line in text.splitlines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        chunks.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable SSE chunk: %s", data_str[:100])
        # If no SSE markers found, try parsing as plain JSON (non-streaming response)
        if not chunks:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    chunks = parsed
                else:
                    chunks = [parsed]
            except json.JSONDecodeError:
                pass
        return chunks
