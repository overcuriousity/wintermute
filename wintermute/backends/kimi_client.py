"""
LLMBackend client for Kimi-Code API.

Kimi-Code exposes an OpenAI-compatible endpoint at
https://api.kimi.com/coding/v1, so this is a thin wrapper that handles
OAuth token refresh transparently.

Implements the :class:`LLMBackend` protocol via ``complete()``.
"""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from wintermute.backends import kimi_auth
from wintermute.backends.openai_wrapper import openai_response_to_llm_response
from wintermute.core.types import LLMResponse

logger = logging.getLogger(__name__)

KIMI_BASE_URL = "https://api.kimi.com/coding/v1"


class KimiCodeClient:
    """Wraps AsyncOpenAI with automatic Kimi OAuth token refresh."""

    def __init__(self, creds: dict | None = None) -> None:
        self._creds = creds or {}
        self._refresh_lock = asyncio.Lock()
        self._openai_client = self._build_client()

    async def complete(self, **kwargs: Any) -> LLMResponse:
        """Send an inference request and return a normalized LLMResponse."""
        await self._ensure_valid_token()
        raw = await self._openai_client.chat.completions.create(**kwargs)
        return openai_response_to_llm_response(raw)

    def _build_client(self) -> AsyncOpenAI:
        token = self._creds.get("access_token", "placeholder")
        return AsyncOpenAI(
            api_key=token,
            base_url=KIMI_BASE_URL,
            default_headers=kimi_auth.api_headers(),
        )

    def update_credentials(self, creds: dict) -> None:
        """Update credentials (e.g. after /kimi-auth completes)."""
        self._creds = creds
        self._openai_client = self._build_client()

    @property
    def authenticated(self) -> bool:
        return bool(self._creds.get("access_token"))

    async def _ensure_valid_token(self) -> None:
        if not self._creds.get("access_token"):
            raise RuntimeError(
                "Kimi-Code is not authenticated. Use /kimi-auth to authorize."
            )
        if not kimi_auth.is_token_expired(self._creds):
            return
        async with self._refresh_lock:
            # Double-check after acquiring lock
            if not kimi_auth.is_token_expired(self._creds):
                return
            logger.info("Kimi-Code token expired — refreshing")
            self._creds = await kimi_auth.refresh_access_token(self._creds)
            self._openai_client = self._build_client()
