"""
LLMBackend client for Anthropic's native Messages API.

Uses the ``anthropic`` Python SDK directly instead of routing through an
OpenAI-compatible proxy.  Supports prompt caching (cache_control on large
system prompts) and native tool calling.

Implements the :class:`LLMBackend` protocol via ``complete()``.
"""

import json
import logging
from typing import Any

import anthropic

from wintermute.core.types import LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)

# Cache system prompts larger than this (bytes).  Anthropic charges less for
# cache hits, so caching the (typically large) system prompt saves money.
_CACHE_SYSTEM_THRESHOLD = 3072


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class AnthropicClient:
    """Wraps the Anthropic Python SDK with the LLMBackend protocol."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, **kwargs: Any) -> LLMResponse:
        """Send an inference request and return a normalized LLMResponse."""
        return await self._call(**kwargs)

    # -- OpenAI → Anthropic translation ------------------------------------

    async def _call(self, **kwargs) -> LLMResponse:
        messages_in: list[dict] = kwargs.get("messages", [])
        tools_in: list[dict] | None = kwargs.get("tools")
        model: str = kwargs.get("model", "claude-sonnet-4-20250514")

        # Anthropic uses max_tokens (required), OpenAI uses max_tokens or
        # max_completion_tokens.  Accept either.
        max_tokens = (
            kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens")
            or 4096
        )

        # -- Extract system prompt ----------------------------------------
        system_text = ""
        conversation: list[dict] = []
        for msg in messages_in:
            if msg.get("role") == "system":
                # Concatenate multiple system messages
                system_text += (msg.get("content") or "") + "\n"
            else:
                conversation.append(msg)

        system_text = system_text.strip()

        # -- Convert messages to Anthropic format --------------------------
        anthropic_messages = self._convert_messages(conversation)

        # -- Build kwargs for the Anthropic SDK ----------------------------
        call_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        # System prompt — use cache_control for large prompts
        if system_text:
            if len(system_text.encode()) > _CACHE_SYSTEM_THRESHOLD:
                call_kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                call_kwargs["system"] = system_text

        # Tools
        if tools_in:
            call_kwargs["tools"] = self._convert_tools(tools_in)
            # tool_choice: map OpenAI's "auto" → Anthropic's {"type": "auto"}
            tc = kwargs.get("tool_choice", "auto")
            if tc == "auto":
                call_kwargs["tool_choice"] = {"type": "auto"}
            elif tc == "none":
                pass  # omit tool_choice to let Anthropic default
            elif tc == "required":
                call_kwargs["tool_choice"] = {"type": "any"}

        # -- Call Anthropic ------------------------------------------------
        response = await self._client.messages.create(**call_kwargs)

        # -- Translate response back to OpenAI shape -----------------------
        return self._translate_response(response)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages to Anthropic format."""
        result: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "assistant":
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError) as _exc:
                            logger.warning(
                                "Malformed tool args in history for %s: %s",
                                fn.get("name", "?"), _exc,
                            )
                            args = {}
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args,
                        })
                if blocks:
                    result.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                # Tool result — Anthropic expects role=user with tool_result block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id or "",
                        "content": content or "",
                    }],
                })

            elif role == "user":
                result.append({"role": "user", "content": content or ""})

        # Anthropic requires strict user/assistant alternation.
        # Merge consecutive same-role messages.
        merged: list[dict] = []
        for msg in result:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                new_content = msg["content"]
                # Normalise to list
                if isinstance(prev_content, str):
                    prev_content = [{"type": "text", "text": prev_content}]
                if isinstance(new_content, str):
                    new_content = [{"type": "text", "text": new_content}]
                merged[-1]["content"] = prev_content + new_content
            else:
                merged.append(msg)

        # Anthropic requires the first message to be from user
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Continue."})

        return merged

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool schemas to Anthropic format."""
        result: list[dict] = []
        for i, tool in enumerate(tools):
            fn = tool.get("function", {})
            entry: dict[str, Any] = {
                "name": fn.get("name", f"tool_{i}"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
            # Cache the last tool definition for prompt caching savings
            if i == len(tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            result.append(entry)
        return result

    def _translate_response(self, response) -> LLMResponse:
        """Convert Anthropic response to normalized LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[LLMToolCall] = []
        thinking_text: list[str] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(LLMToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input),
                ))
            elif block.type == "thinking":
                thinking_text.append(block.thinking)

        # Map stop_reason
        stop_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
        }
        finish_reason = stop_map.get(response.stop_reason, "stop")

        content = "\n".join(text_parts) if text_parts else None
        reasoning = "\n".join(thinking_text) if thinking_text else None

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            reasoning_content=reasoning,
        )
