"""
AsyncOpenAI-compatible client for Google Cloud Code Assist API.

Implements the ``client.chat.completions.create()`` interface used by all
wintermute inference code, translating between OpenAI and Google formats.

Drop-in replacement for ``AsyncOpenAI`` — no changes needed in llm_thread.py,
sub_session.py, dreaming.py, or supervisor.py.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from wintermute import gemini_auth

logger = logging.getLogger(__name__)

CLOUDCODE_BASE = "https://cloudcode-pa.googleapis.com"
CLOUDCODE_STREAM_URL = f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent"

# Retry settings
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


# ---------------------------------------------------------------------------
# Response dataclasses (mimic OpenAI ChatCompletion shape)
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: FunctionCall = None


@dataclass
class Message:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


@dataclass
class Choice:
    index: int = 0
    message: Message = None
    finish_reason: str = "stop"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatCompletion:
    id: str = ""
    object: str = "chat.completion"
    choices: list[Choice] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Namespace objects to mimic client.chat.completions.create()
# ---------------------------------------------------------------------------

class _Completions:
    def __init__(self, client: "GeminiCloudClient"):
        self._client = client

    async def create(self, **kwargs) -> ChatCompletion:
        return await self._client._create(**kwargs)


class _Chat:
    def __init__(self, client: "GeminiCloudClient"):
        self.completions = _Completions(client)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class GeminiCloudClient:
    """AsyncOpenAI-compatible client backed by Google Cloud Code Assist."""

    def __init__(self, creds: dict):
        self._creds = dict(creds)
        self.chat = _Chat(self)
        self._http = httpx.AsyncClient(timeout=300)
        self._refresh_lock = asyncio.Lock()

    async def _ensure_valid_token(self) -> str:
        """Return a valid access token, refreshing if expired."""
        if time.time() < self._creds.get("expires_at", 0) - 60:
            return self._creds["access_token"]
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if time.time() < self._creds.get("expires_at", 0) - 60:
                return self._creds["access_token"]
            logger.info("Refreshing Gemini access token")
            loop = asyncio.get_event_loop()
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
            elif role == "tool":
                google_role = "function"
            else:
                google_role = "user"

            parts = []

            # Handle tool call results
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "") if isinstance(msg, dict) else getattr(msg, "tool_call_id", "")
                name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
                parts.append({
                    "functionResponse": {
                        "name": name or tool_call_id,
                        "response": {"result": content or ""},
                    }
                })
            # Handle assistant messages with tool calls
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", "{}")
                        else:
                            fn_name = tc.function.name
                            fn_args = tc.function.arguments
                        try:
                            args_dict = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except json.JSONDecodeError:
                            args_dict = {}
                        parts.append({
                            "functionCall": {
                                "name": fn_name,
                                "args": args_dict,
                            }
                        })
                if content:
                    parts.append({"text": content})
            else:
                if content:
                    parts.append({"text": content})

            if parts:
                contents.append({"role": google_role, "parts": parts})

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

    def _translate_response(self, chunks: list[dict]) -> ChatCompletion:
        """Assemble Google SSE chunks into an OpenAI-shaped ChatCompletion."""
        text_parts = []
        tool_calls = []
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        tc_counter = 0

        for chunk in chunks:
            candidates = chunk.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                for part in content.get("parts", []):
                    if "text" in part:
                        text_parts.append(part["text"])
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tc_counter += 1
                        tool_calls.append(ToolCall(
                            id=f"call_{tc_counter}",
                            function=FunctionCall(
                                name=fc.get("name", ""),
                                arguments=json.dumps(fc.get("args", {})),
                            ),
                        ))

                # Map finish reason
                fr = candidate.get("finishReason", "")
                if fr == "MAX_TOKENS":
                    finish_reason = "length"
                elif fr == "STOP":
                    finish_reason = "stop"
                elif fr == "SAFETY":
                    finish_reason = "stop"

            # Usage metadata
            usage_meta = chunk.get("usageMetadata", {})
            if usage_meta:
                prompt_tokens = max(prompt_tokens, usage_meta.get("promptTokenCount", 0))
                completion_tokens = max(completion_tokens, usage_meta.get("candidatesTokenCount", 0))

        if tool_calls:
            finish_reason = "tool_calls"

        message = Message(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )

        return ChatCompletion(
            id=f"gemini-{int(time.time())}",
            choices=[Choice(message=message, finish_reason=finish_reason)],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # -------------------------------------------------------------------
    # Core create method
    # -------------------------------------------------------------------

    async def _create(self, **kwargs) -> ChatCompletion:
        """Send an inference request to Cloud Code Assist."""
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
                    logger.warning("Gemini request transport error (attempt %d): %s", attempt + 1, exc)
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise

            # 401: refresh token and retry once
            if resp.status_code == 401 and attempt == 0:
                logger.info("Gemini 401 — refreshing token and retrying")
                async with self._refresh_lock:
                    loop = asyncio.get_event_loop()
                    self._creds = await loop.run_in_executor(
                        None, gemini_auth.refresh_access_token, self._creds
                    )
                    await loop.run_in_executor(
                        None, gemini_auth.save_credentials, self._creds
                    )
                continue

            # 429/503: backoff with Retry-After
            if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                logger.warning("Gemini %d — retrying after %.1fs", resp.status_code, wait)
                await asyncio.sleep(wait)
                backoff *= 2
                continue

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                raise RuntimeError(
                    f"Gemini Cloud Code Assist API error {resp.status_code}: {error_text}"
                )

            # Parse SSE response
            chunks = self._parse_sse(resp.text)
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
