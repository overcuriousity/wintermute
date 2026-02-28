"""
LLMBackend wrapper for AsyncOpenAI clients.

Provides ``OpenAIBackend`` which wraps ``AsyncOpenAI`` with a ``complete()``
method returning :class:`LLMResponse`, and a shared helper
``openai_response_to_llm_response()`` reused by :mod:`kimi_client`.
"""

from typing import Any

from openai import AsyncOpenAI

from wintermute.core.types import LLMResponse, LLMToolCall


def openai_response_to_llm_response(raw: Any) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` object to :class:`LLMResponse`."""
    if not raw.choices:
        return LLMResponse(content=None, tool_calls=None, finish_reason="stop")

    choice = raw.choices[0]
    message = choice.message

    tool_calls: list[LLMToolCall] | None = None
    if message.tool_calls:
        tool_calls = [
            LLMToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in message.tool_calls
        ]

    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        reasoning = reasoning.strip()

    return LLMResponse(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        reasoning_content=reasoning or None,
    )


class OpenAIBackend:
    """Wraps ``AsyncOpenAI`` with the :class:`LLMBackend` protocol."""

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, **kwargs: Any) -> LLMResponse:
        raw = await self._client.chat.completions.create(**kwargs)
        return openai_response_to_llm_response(raw)
