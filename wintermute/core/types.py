"""
Shared type definitions for the wintermute.core package.

Houses configuration data classes and exception types that are used across
multiple modules — extracted from llm_thread.py to break import coupling.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ContextTooLargeError(Exception):
    """The request payload or token count exceeds backend limits."""


class RateLimitError(Exception):
    """The provider returned a rate-limit (HTTP 429) error."""


_CONTEXT_TOO_LARGE_PHRASES = (
    "context length", "too many tokens", "maximum context",
    "token limit", "content too large", "payload too large",
)


def classify_api_error(exc: Exception) -> type | None:
    """Classify a provider exception as a normalized error type.

    Returns :class:`RateLimitError` or :class:`ContextTooLargeError` if
    the exception matches the corresponding pattern, or ``None`` for
    unrecognised errors.
    """
    status = getattr(exc, "status_code", None)

    # Rate-limit: HTTP 429 or "429" in the exception class name or code.
    if (
        status == 429
        or "429" in type(exc).__name__
        or (hasattr(exc, "code") and str(getattr(exc, "code", "")) == "429")
    ):
        return RateLimitError

    # Context too large: HTTP 413, or 400 with matching phrases.
    if status == 413:
        return ContextTooLargeError
    if status == 400:
        msg = str(exc).lower()
        if any(phrase in msg for phrase in _CONTEXT_TOO_LARGE_PHRASES):
            return ContextTooLargeError

    return None


@dataclass
class ProviderConfig:
    name: str               # unique backend name from inference_backends
    model: str
    context_size: int       # total token window the model supports (e.g. 65536)
    max_tokens: int = 4096  # maximum tokens in a single response
    reasoning: bool = False  # enable reasoning/thinking token support (o1/o3, DeepSeek R1, etc.)
    provider: str = "openai"  # "openai", "anthropic", "gemini-cli", or "kimi-code"
    api_key: str = ""
    base_url: str = ""       # e.g. http://localhost:8080/v1  or  https://api.openai.com/v1


class BackendPool:
    """Ordered list of LLM backends for a role, with automatic failover.

    On API errors the next backend in the list is tried automatically.
    An empty pool (``len(pool) == 0``) signals "disabled" — relevant for
    optional roles like turing_protocol.
    """

    def __init__(self, backends: "list[tuple[ProviderConfig, object]]") -> None:
        self._backends = backends
        self.last_used: str = backends[0][0].name if backends else ""

    # -- Convenience accessors ------------------------------------------------

    @property
    def primary(self) -> ProviderConfig:
        """Primary (first) backend config — used for context_size, model name, etc."""
        return self._backends[0][0]

    @property
    def primary_client(self) -> object:
        """Primary (first) client instance."""
        return self._backends[0][1]

    @property
    def enabled(self) -> bool:
        return len(self._backends) > 0

    def __len__(self) -> int:
        return len(self._backends)

    # -- Rate-limit retry settings --------------------------------------------
    RATE_LIMIT_MAX_RETRIES = 5
    RATE_LIMIT_INITIAL_BACKOFF = 2.0   # seconds
    RATE_LIMIT_MAX_BACKOFF = 60.0      # seconds

    # -- API call with failover -----------------------------------------------

    async def call(self, *, messages: list[dict],
                   tools: "list[dict] | None" = None,
                   max_tokens_override: "int | None" = None,
                   **extra_kwargs) -> object:
        """Call ``chat.completions.create`` with automatic failover.

        Each backend uses its own model, max_tokens, and reasoning setting.
        *max_tokens_override* replaces the backend's configured max_tokens
        (useful for compaction's hard-coded 2048).

        Rate-limit errors (HTTP 429) are retried with exponential backoff
        before failing over to the next backend.
        """
        last_error: Exception | None = None
        for cfg, client in self._backends:
            call_kwargs: dict = {"model": cfg.model, "messages": messages}
            if tools is not None:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = "auto"
            max_tok = max_tokens_override if max_tokens_override is not None else cfg.max_tokens
            if cfg.reasoning:
                call_kwargs["max_completion_tokens"] = max_tok
            else:
                call_kwargs["max_tokens"] = max_tok
            call_kwargs.update(extra_kwargs)

            backoff = self.RATE_LIMIT_INITIAL_BACKOFF
            for attempt in range(self.RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    result = await client.chat.completions.create(**call_kwargs)
                    self.last_used = cfg.name
                    return result
                except Exception as exc:  # noqa: BLE001
                    error_cls = classify_api_error(exc)

                    if error_cls is RateLimitError and attempt < self.RATE_LIMIT_MAX_RETRIES:
                        jitter = random.uniform(0, backoff * 0.5)
                        wait = min(backoff + jitter, self.RATE_LIMIT_MAX_BACKOFF)
                        backend_desc = f"'{cfg.name}' ({cfg.model})"
                        logger.warning(
                            "Backend %s rate-limited — retrying in %.1fs (attempt %d/%d)",
                            backend_desc, wait, attempt + 1, self.RATE_LIMIT_MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        backoff = min(backoff * 2, self.RATE_LIMIT_MAX_BACKOFF)
                        continue

                    if error_cls is ContextTooLargeError:
                        raise ContextTooLargeError(str(exc)) from exc

                    last_error = exc
                    backend_desc = f"'{cfg.name}' ({cfg.model})"
                    if len(self._backends) > 1:
                        logger.warning("Backend %s failed: %s — trying next", backend_desc, exc)
                    else:
                        raise
                    break  # try next backend

        if last_error is None:
            if not self._backends:
                last_error = RuntimeError("No backends configured or enabled for this role")
            else:
                last_error = RuntimeError("All backends exhausted without a concrete error")
        assert last_error is not None
        raise last_error


@dataclass
class MultiProviderConfig:
    """Per-purpose LLM backend pools.

    Configured via ``inference_backends`` (named backend definitions) and
    ``llm`` (role-to-backend-name mapping).  Each field holds an ordered
    list of ProviderConfig objects; the runtime BackendPool is built from
    these lists plus the corresponding clients.

    An empty list for *turing_protocol* means "disabled".
    """
    main: list[ProviderConfig]
    compaction: list[ProviderConfig]
    sub_sessions: list[ProviderConfig]
    dreaming: list[ProviderConfig]
    turing_protocol: list[ProviderConfig]
    memory_harvest: list[ProviderConfig] = field(default_factory=list)
    nl_translation: list[ProviderConfig] = field(default_factory=list)
    reflection: list[ProviderConfig] = field(default_factory=list)
