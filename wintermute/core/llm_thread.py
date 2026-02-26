"""
LLM Inference Thread

Owns the conversation history and all interactions with any OpenAI-compatible
API endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).

Receives user messages via an asyncio Queue, runs inference (including
multi-step tool-use loops), and delivers responses back through reply Futures.

Public API used by other modules
---------------------------------
  LLMThread.enqueue_user_message(text, thread_id)  -> LLMReply  (awaitable)
  LLMThread.enqueue_system_event(text, thread_id)  -> None (fire-and-forget)
  LLMThread.reset_session(thread_id)
  LLMThread.force_compact(thread_id)
"""

import asyncio
import json
import logging
import random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader
from wintermute.core import turing_protocol as turing_protocol_module
from wintermute.core.inference_engine import ToolCallContext, process_tool_call
from wintermute import tools as tool_module

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)


class ContextTooLargeError(Exception):
    """The request payload or token count exceeds backend limits."""


# Keep the last N messages untouched during compaction.
COMPACTION_KEEP_RECENT = 10



def _count_tokens(text: str, model: str) -> int:
    """
    Estimate token count using tiktoken.
    Falls back to cl100k_base (GPT-4 / DeepSeek / Qwen BPE) for unknown
    model names, and to len//4 if tiktoken itself is unavailable.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return len(text) // 4


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
                    is_rate_limit = (
                        getattr(exc, "status_code", None) == 429
                        or "429" in str(type(exc).__name__)
                        or (hasattr(exc, "code") and str(getattr(exc, "code", "")) == "429")
                    )
                    if is_rate_limit and attempt < self.RATE_LIMIT_MAX_RETRIES:
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

                    is_context_too_large = (
                        getattr(exc, "status_code", None) == 413
                        or (
                            getattr(exc, "status_code", None) == 400
                            and any(
                                phrase in str(exc).lower()
                                for phrase in ("context length", "too many tokens", "maximum context",
                                               "token limit", "content too large", "payload too large")
                            )
                        )
                    )
                    if is_context_too_large:
                        raise ContextTooLargeError(str(exc)) from exc

                    last_error = exc
                    backend_desc = f"'{cfg.name}' ({cfg.model})"
                    if len(self._backends) > 1:
                        logger.warning("Backend %s failed: %s — trying next", backend_desc, exc)
                    else:
                        raise
                    break  # try next backend

        raise last_error  # type: ignore[misc]


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


@dataclass
class LLMReply:
    """Response from the LLM, separating visible content from reasoning tokens."""
    text: str
    reasoning: Optional[str] = None  # reasoning/thinking tokens (if model supports it)
    tool_calls_made: list[str] = field(default_factory=list)  # tool names called during inference
    tool_call_details: list[dict] = field(default_factory=list)  # [{name, arguments, result}, ...]

    def __str__(self) -> str:
        return self.text


@dataclass
class _QueueItem:
    text: str
    thread_id: str = "default"
    is_system_event: bool = False
    future: Optional[asyncio.Future] = field(default=None, compare=False)
    turing_depth: int = 0  # 0=normal, 1=first correction, 2=re-check correction (max)
    content: Optional[list] = None  # multimodal content parts (OpenAI vision format)
    # Sequence number of the user-message turn this correction was issued for.
    # If the thread has advanced past this number by the time the correction is
    # dequeued, the correction is stale and will be dropped.
    correction_for_seq: Optional[int] = None


class LLMThread:
    """Runs as an asyncio task within the shared event loop."""

    def __init__(self, main_pool: BackendPool, compaction_pool: BackendPool,
                 turing_protocol_pool: BackendPool, broadcast_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None,
                 turing_protocol_validators: "Optional[dict[str, bool]]" = None,
                 nl_translation_pool: "Optional[BackendPool]" = None,
                 nl_translation_config: "Optional[dict]" = None,
                 seed_language: str = "en",
                 event_bus: "Optional[EventBus]" = None) -> None:
        self._main_pool = main_pool
        self._compaction_pool = compaction_pool
        self._turing_protocol_pool = turing_protocol_pool
        self._turing_protocol_validators = turing_protocol_validators
        self._nl_translation_pool = nl_translation_pool
        self._nl_translation_config = nl_translation_config or {}
        self._seed_language = seed_language
        self._event_bus = event_bus
        # Convenience: primary config for context_size / model name lookups.
        self._cfg = main_pool.primary
        self._broadcast = broadcast_fn  # async callable(text, thread_id, *, reasoning=None)
        self._sub_sessions = sub_session_manager  # set post-init via inject_sub_session_manager
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._running = False
        # Per-thread compaction summaries: thread_id -> summary text
        self._compaction_summaries: dict[str, Optional[str]] = {}
        # Per-thread sequence counter for user-facing turns.  Incremented each
        # time a non-system-event item is processed.  Used to detect stale
        # Turing Protocol corrections that arrived after the conversation moved on.
        self._thread_seq: dict[str, int] = {}
        # Per-thread cache of the last system prompt actually sent to the LLM.
        self._last_system_prompt: dict[str, str] = {}

    def inject_sub_session_manager(self, manager: "SubSessionManager") -> None:
        """Called after construction once SubSessionManager is built."""
        self._sub_sessions = manager

    # ------------------------------------------------------------------
    # Read-only accessors for external consumers (interfaces, /status)
    # ------------------------------------------------------------------

    @property
    def seed_language(self) -> str:
        return self._seed_language

    @property
    def main_pool(self) -> BackendPool:
        return self._main_pool

    @property
    def compaction_pool(self) -> BackendPool:
        return self._compaction_pool

    @property
    def turing_protocol_pool(self) -> BackendPool:
        return self._turing_protocol_pool

    @property
    def nl_translation_pool(self) -> "Optional[BackendPool]":
        return self._nl_translation_pool

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def enqueue_user_message(
        self, text: str, thread_id: str = "default",
        content: Optional[list] = None,
    ) -> "LLMReply":
        """Submit a user message and await the AI reply (returns LLMReply).

        *content* is an optional list of OpenAI multimodal content parts
        (e.g. text + image_url).  When set, it is used as the message
        payload sent to the API instead of *text*.  *text* is always used
        for DB storage and logging.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(
            text=text, thread_id=thread_id, future=fut, content=content,
        ))
        return await fut

    async def enqueue_system_event(self, text: str, thread_id: str = "default") -> None:
        """Submit an autonomous system event (heartbeat, scheduled task, etc.)."""
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id, is_system_event=True))

    async def enqueue_system_event_with_reply(self, text: str,
                                              thread_id: str = "default") -> str:
        """Submit a system event and await the AI reply.

        Like enqueue_system_event but returns the reply text. The prompt is
        NOT saved to the DB as a user message; only the assistant response is.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id,
                                         is_system_event=True, future=fut))
        return await fut

    async def reset_session(self, thread_id: str = "default") -> None:
        await database.async_call(database.clear_active_messages, thread_id)
        self._compaction_summaries.pop(thread_id, None)
        if self._sub_sessions:
            n = await self._sub_sessions.cancel_for_thread(thread_id)
            if n:
                logger.info("Cancelled %d sub-session(s) for reset thread %s", n, thread_id)
        logger.info("Session reset for thread %s", thread_id)

    async def force_compact(self, thread_id: str = "default") -> None:
        await self._compact_context(thread_id)

    def get_compaction_summary(self, thread_id: str = "default") -> Optional[str]:
        """Return the current in-memory compaction summary for a thread, or None."""
        return self._compaction_summaries.get(thread_id)

    def get_last_system_prompt(self, thread_id: str = "default") -> Optional[str]:
        """Return the last system prompt actually sent to the LLM for *thread_id*."""
        return self._last_system_prompt.get(thread_id)

    def get_token_budget(self, thread_id: str = "default") -> dict:
        """Return precise token accounting for a thread."""
        total_limit = max(self._cfg.context_size - self._cfg.max_tokens, 1)
        model = self._cfg.model

        # Prefer the cached prompt that was actually sent to the LLM.
        sp_text = self._last_system_prompt.get(thread_id)
        if sp_text is None:
            summary = self._compaction_summaries.get(thread_id)
            try:
                sp_text = prompt_assembler.assemble(extra_summary=summary)
            except Exception:  # noqa: BLE001
                sp_text = ""
        sp_tokens = _count_tokens(sp_text, model)

        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        active_schemas = tool_module.get_tool_schemas(nl_tools=nl_tools)
        tools_tokens = _count_tokens(json.dumps(active_schemas), model)

        stats = database.get_thread_stats(thread_id)
        hist_tokens = stats["token_used"]

        total_used = sp_tokens + tools_tokens + hist_tokens
        pct = round(min(total_used / total_limit * 100, 100), 1)

        return {
            "total_limit": total_limit,
            "sp_tokens": sp_tokens,
            "tools_tokens": tools_tokens,
            "hist_tokens": hist_tokens,
            "total_used": total_used,
            "pct": pct,
            "msg_count": stats["msg_count"],
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        # Load summaries for all known threads
        for tid in await database.async_call(database.get_active_thread_ids):
            summary = await database.async_call(database.load_latest_summary, tid)
            if summary:
                self._compaction_summaries[tid] = summary
        logger.info("LLM thread started (endpoint=%s model=%s)", self._cfg.base_url, self._cfg.model)
        if self._turing_protocol_pool.enabled:
            logger.info("Turing Protocol enabled (model=%s)", self._turing_protocol_pool.primary.model)
        else:
            logger.info("Turing Protocol disabled")

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Drop Turing Protocol corrections that are stale — i.e. the thread
            # has already processed at least one new user-facing turn since the
            # correction was issued, making it no longer relevant.
            if item.turing_depth > 0 and item.correction_for_seq is not None:
                current_seq = self._thread_seq.get(item.thread_id, 0)
                if current_seq > item.correction_for_seq:
                    logger.warning(
                        "Dropping stale Turing Protocol correction for thread %s "
                        "(issued at seq=%d, current seq=%d)",
                        item.thread_id, item.correction_for_seq, current_seq,
                    )
                    try:
                        await database.async_call(
                            database.save_interaction_log,
                            _time.time(), "turing_stale_drop", item.thread_id,
                            self._main_pool.last_used,
                            item.text[:2000], "", "stale",
                        )
                    except Exception:
                        pass
                    self._queue.task_done()
                    continue

            # Advance the per-thread sequence counter for every user-facing turn
            # and for Turing corrections that passed the staleness check.
            # Corrections must also advance the counter so that cascaded
            # corrections (depth > 1) record a higher correction_for_seq and
            # aren't falsely dropped if a user turn arrives in between.
            if not item.is_system_event or item.turing_depth > 0:
                self._thread_seq[item.thread_id] = (
                    self._thread_seq.get(item.thread_id, 0) + 1
                )

            # Seed empty threads on first real user message
            if not item.is_system_event and not await database.async_call(database.thread_has_messages, item.thread_id):
                try:
                    seed_prompt = prompt_loader.load_seed(self._seed_language)
                    seed_item = _QueueItem(
                        text=seed_prompt,
                        thread_id=item.thread_id,
                        is_system_event=True,
                    )
                    seed_reply = await self._process(seed_item)
                    if seed_reply.text:
                        try:
                            await self._broadcast(
                                seed_reply.text, item.thread_id,
                                reasoning=seed_reply.reasoning,
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("Failed to broadcast seed reply")
                except Exception:  # noqa: BLE001
                    logger.exception("Seed injection failed (non-fatal)")

            # Collect recent assistant messages BEFORE _process() saves
            # the new reply to the DB.  This avoids the repetition detector
            # comparing the current response against itself (which would
            # always yield 100% similarity).
            _prior_assistant = None
            _recent_assistant: list[str] = []
            try:
                _db_msgs = await database.async_call(database.load_active_messages, item.thread_id)
                for m in reversed(_db_msgs):
                    if m.get("role") == "assistant":
                        content = m.get("content", "")
                        if isinstance(content, str) and content:
                            _recent_assistant.append(content)
                            if _prior_assistant is None:
                                _prior_assistant = content
                            if len(_recent_assistant) >= 3:
                                break
                _recent_assistant.reverse()
            except Exception:
                pass

            try:
                reply = await self._process(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM processing error")
                try:
                    await database.async_call(
                        database.save_interaction_log,
                        _time.time(), "chat", item.thread_id,
                        self._main_pool.last_used,
                        item.text, str(exc), "error",
                    )
                except Exception:  # noqa: BLE001
                    pass
                err_msg = str(exc)
                # Provide actionable hint for Gemini auth failures
                if "401" in err_msg and "Gemini" in err_msg:
                    err_msg += (
                        "\n\nGemini credentials have expired or are invalid. "
                        "Re-run authentication on the server:\n"
                        "  uv run python -m wintermute.gemini_auth"
                    )
                reply = LLMReply(text=f"[Error during inference: {err_msg}]")

            if item.future and not item.future.done():
                item.future.set_result(reply)
            elif item.is_system_event and not item.future and item.turing_depth == 0:
                # System events without a future (sub-session results,
                # scheduled tasks, /tasks commands) have no caller waiting for
                # the reply.  Broadcast the LLM's response directly so
                # it reaches the user.
                text_to_send = reply.text or item.text  # fallback to raw event
                logger.info(
                    "Broadcasting system-event reply for thread %s (%d chars, reply_empty=%s)",
                    item.thread_id, len(text_to_send), not reply.text,
                )
                try:
                    await self._broadcast(text_to_send, item.thread_id,
                                          reasoning=reply.reasoning)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to broadcast system-event reply for thread %s",
                                     item.thread_id)
            elif item.turing_depth > 0 and reply.text:
                # Turing correction response — always broadcast so the user
                # sees the self-correction.  The model may have complied with
                # tool calls, or explained why it can't — either way the user
                # should see the result.
                logger.info(
                    "Broadcasting Turing correction response for thread %s "
                    "(depth=%d, tools=%s)",
                    item.thread_id, item.turing_depth,
                    reply.tool_calls_made or "none",
                )
                try:
                    await self._broadcast(reply.text, item.thread_id,
                                          reasoning=reply.reasoning)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to broadcast Turing correction response")

            # -- Turing Protocol validation --
            # Fire async after the reply is delivered.  Checks user-facing
            # messages AND correction responses (up to depth 2 to prevent
            # infinite loops).  If the model ignores a correction and
            # repeats the violation, the re-check catches it.
            if (
                not item.thread_id.startswith("sub_")
                and reply.text
                and (not item.is_system_event or item.turing_depth > 0)
                and item.turing_depth < 2
            ):
                # Snapshot the current sequence number so the correction can
                # be dropped if the conversation has moved on before it lands.
                seq_at_fire = self._thread_seq.get(item.thread_id, 0)
                # _prior_assistant and _recent_assistant were collected
                # before _process() above, so they don't include the
                # current reply.
                asyncio.create_task(
                    self._run_turing_check(
                        user_message=item.text,
                        assistant_response=reply.text,
                        tool_calls_made=reply.tool_calls_made,
                        thread_id=item.thread_id,
                        issued_for_seq=seq_at_fire,
                        turing_depth=item.turing_depth,
                        prior_assistant_message=_prior_assistant,
                        recent_assistant_messages=_recent_assistant,
                    ),
                    name=f"turing_{item.thread_id}",
                )

            # Emit main-thread turn event for reflection/synthesis.
            if (
                self._event_bus
                and not item.thread_id.startswith("sub_")
                and not item.is_system_event
            ):
                # Extract skills loaded from read_file calls on data/skills/.
                skills_loaded = []
                for tc in reply.tool_call_details:
                    if tc.get("name") == "read_file":
                        try:
                            args = json.loads(tc.get("arguments", "{}"))
                            p = args.get("path", "")
                            parts = Path(p).parts
                            if (
                                "data" in parts
                                and "skills" in parts
                                and parts.index("skills") == parts.index("data") + 1
                                and p.endswith(".md")
                            ):
                                import re as _re
                                m = _re.search(r'data/skills/([^/]+)\.md', p)
                                if m:
                                    skills_loaded.append(m.group(1))
                        except Exception:
                            pass
                self._event_bus.emit(
                    "main_thread.turn_completed",
                    thread_id=item.thread_id,
                    tools_used=reply.tool_calls_made,
                    skills_loaded=skills_loaded,
                    had_error="[Error" in (reply.text or ""),
                )

            self._queue.task_done()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Turing Protocol validation
    # ------------------------------------------------------------------

    async def _run_turing_check(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls_made: list[str],
        thread_id: str,
        issued_for_seq: int = 0,
        turing_depth: int = 0,
        prior_assistant_message: Optional[str] = None,
        recent_assistant_messages: Optional[list[str]] = None,
    ) -> None:
        """Fire the Turing Protocol pipeline to detect violations.

        Runs asynchronously after the main reply has already been delivered.
        If violations are confirmed, a corrective system event is enqueued.

        Correction responses are re-checked once (depth 0 → 1 → 2 max) to
        catch models that ignore the correction and repeat the violation.
        Depth 2 responses are never re-checked, preventing infinite loops.

        ``issued_for_seq`` records the per-thread sequence number at the time
        the check was fired.  If the thread has advanced past this number by
        the time the correction is dequeued, the correction is dropped as
        stale (the user has already sent a follow-up and the context has moved
        on).
        """
        if not self._turing_protocol_pool.enabled:
            return

        active_sessions = self._sub_sessions.list_active() if self._sub_sessions else []

        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None

        try:
            result = await turing_protocol_module.run_turing_protocol(
                pool=self._turing_protocol_pool,
                user_message=user_message,
                assistant_response=assistant_response,
                tool_calls_made=tool_calls_made,
                active_sessions=active_sessions,
                enabled_validators=self._turing_protocol_validators,
                thread_id=thread_id,
                phase="post_inference",
                scope="main",
                nl_tools=nl_tools,
                prior_assistant_message=prior_assistant_message,
                recent_assistant_messages=recent_assistant_messages,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Turing Protocol check raised (non-fatal)")
            return

        if result.correction:
            new_depth = turing_depth + 1
            # At depth >= 2, the model already failed to comply with a
            # correction once.  Repeating the same demanding prompt won't
            # help — the model is likely incapable of making the tool call
            # in this context.  Switch to a graceful fallback that tells
            # the model to stop trying and explain the limitation instead.
            if turing_depth >= 1:
                correction_text = (
                    "[TURING PROTOCOL — UNABLE TO COMPLY] "
                    "The previous correction could not be fulfilled. "
                    "Simply continue the conversation naturally. "
                    "Do NOT claim tools are blocked or unavailable. "
                    "Do NOT repeat the failed action. Just respond "
                    "helpfully to the user's last real message."
                )
            else:
                correction_text = result.correction
            logger.info(
                "Turing Protocol injecting correction into thread %s (depth=%d, hooks=%s)",
                thread_id, new_depth,
                [m["hook"] for m in result.correction_metadata],
            )
            await self._queue.put(_QueueItem(
                text=correction_text,
                thread_id=thread_id,
                is_system_event=True,
                turing_depth=new_depth,
                correction_for_seq=issued_for_seq,
            ))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _process(self, item: _QueueItem) -> LLMReply:
        thread_id = item.thread_id
        messages = await self._build_messages(item.text, item.is_system_event, thread_id, item.content)

        # Build a query for vector memory retrieval (user message + last assistant reply).
        _query_parts = [item.text] if item.text else []
        for m in reversed(messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                _query_parts.append(m["content"][:500])
                break
        _memory_query = " ".join(_query_parts) if _query_parts else None

        # Pre-fetch memories off the event loop to avoid blocking I/O.
        from wintermute.infra import memory_store
        if memory_store.is_vector_enabled() and _memory_query:
            try:
                _memory_results = await asyncio.to_thread(memory_store.search, _memory_query)
            except Exception as e:
                logger.warning("Vector memory search failed, continuing without memory context: %s", e)
                _memory_results = None
        else:
            _memory_results = None

        # Assemble system prompt first so we can measure its real token cost.
        summary = self._compaction_summaries.get(thread_id)
        system_prompt = prompt_assembler.assemble(extra_summary=summary, query=_memory_query, memory_results=_memory_results)
        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        active_schemas = tool_module.get_tool_schemas(nl_tools=nl_tools)
        overhead_tokens = (
            _count_tokens(system_prompt, self._cfg.model)
            + _count_tokens(json.dumps(active_schemas), self._cfg.model)
        )

        history_tokens = sum(
            _count_tokens(m["content"] if isinstance(m["content"], str) else
                          " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict)),
                          self._cfg.model)
            for m in messages
        )
        compaction_threshold = self._cfg.context_size - self._cfg.max_tokens - overhead_tokens
        if history_tokens > compaction_threshold:
            logger.info(
                "History at %d tokens (overhead %d, threshold %d) – compacting before inference (thread=%s)",
                history_tokens, overhead_tokens, compaction_threshold, thread_id,
            )
            await self._compact_context(thread_id)
            messages = await self._build_messages(item.text, item.is_system_event, thread_id, item.content)
            # Reassemble with the updated compaction summary.
            summary = self._compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(extra_summary=summary, query=_memory_query, memory_results=_memory_results)

        self._last_system_prompt[thread_id] = system_prompt

        is_sub_session_result = item.is_system_event and "[SUB-SESSION " in item.text
        if not item.is_system_event:
            db_text = item.text
            if item.content is not None:
                db_text = item.text or "[image attached]"
            await database.async_call(
                database.save_message, "user", db_text, thread_id,
                token_count=_count_tokens(db_text, self._cfg.model))
            if self._event_bus:
                self._event_bus.emit("message.received", thread_id=thread_id, text=db_text)
        elif is_sub_session_result:
            _se_text = f"[SYSTEM EVENT] {item.text}"
            await database.async_call(
                database.save_message, "user", _se_text, thread_id,
                token_count=_count_tokens(_se_text, self._cfg.model))
        elif item.turing_depth > 0:
            # Turing correction prompt: saved to DB before inference so the
            # model sees it.  If the model ignores the correction (no tool
            # calls), both the prompt and the response are deleted afterward
            # to avoid polluting future turns (see cleanup below).
            await database.async_call(
                database.save_message, "user", item.text, thread_id,
                token_count=_count_tokens(item.text, self._cfg.model))

        _inference_start = _time.time()
        try:
            reply = await self._inference_loop(
                system_prompt, messages, thread_id,
                disable_tools=is_sub_session_result,
            )
        except ContextTooLargeError:
            logger.warning("Context too large for thread %s — forcing compaction", thread_id)
            await self._compact_context(thread_id)
            messages = await self._build_messages(item.text, item.is_system_event, thread_id, item.content)
            summary = self._compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(extra_summary=summary, query=_memory_query, memory_results=_memory_results)
            # Retry once after compaction
            reply = await self._inference_loop(
                system_prompt, messages, thread_id,
                disable_tools=is_sub_session_result,
            )

        # Determine action type for interaction log
        if item.turing_depth > 0:
            _action = "turing_response"
        elif thread_id.startswith("sub_"):
            _action = "sub_session"
        elif item.is_system_event:
            _action = "system_event"
        else:
            _action = "chat"

        try:
            raw_output_data = {
                "content": reply.text,
                "tool_calls": reply.tool_call_details,
                "reasoning": reply.reasoning,
            }
            await database.async_call(
                database.save_interaction_log,
                _time.time(), _action, thread_id,
                self._main_pool.last_used,
                item.text, reply.text, "ok",
                raw_output=json.dumps(raw_output_data),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save interaction log entry", exc_info=True)

        # Emit inference.completed event for self-model metrics.
        _inference_duration = _time.time() - _inference_start
        if self._event_bus:
            self._event_bus.emit(
                "inference.completed",
                thread_id=thread_id,
                duration_s=round(_inference_duration, 2),
                tool_calls=len(reply.tool_call_details) if reply.tool_call_details else 0,
                model=self._main_pool.last_used,
            )
        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(), "inference_completed", thread_id,
                self._main_pool.last_used,
                f"duration={_inference_duration:.2f}s",
                f"tool_calls={len(reply.tool_call_details) if reply.tool_call_details else 0}",
                "ok",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to log inference_completed", exc_info=True)

        _assistant_text = reply.text or "..."
        await database.async_call(
            database.save_message, "assistant", _assistant_text, thread_id,
            token_count=_count_tokens(_assistant_text, self._cfg.model))
        if self._event_bus:
            self._event_bus.emit("message.sent", thread_id=thread_id, text=_assistant_text)

        # Turing correction cleanup: if the re-check (at depth+1) confirms
        # the model STILL violated after the correction, the failed exchange
        # will be cleaned up by the next depth's check.  For now, keep the
        # response in DB — it may be a valid explanation of why the model
        # can't comply.
        await self._maybe_summarise_components(
            thread_id, _from_system_event=item.is_system_event,
        )
        return reply

    # ------------------------------------------------------------------
    # OpenAI inference loop (handles tool-use rounds)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reasoning(message) -> Optional[str]:
        """Extract reasoning/thinking content from a response message, if present."""
        # OpenAI o-series and DeepSeek R1 use reasoning_content
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            return reasoning.strip()
        return None

    def _trim_tool_results(self, messages: list[dict], token_budget: int) -> None:
        """Truncate oldest tool-result messages if total tokens exceed budget.

        Modifies *messages* in place.  Only tool-role messages are truncated
        (replaced with a short notice).  This prevents 400 errors from
        providers when tool results cause the payload to exceed the context
        window.
        """
        model = self._cfg.model
        total = sum(
            _count_tokens(
                m.get("content") if isinstance(m, dict) and isinstance(m.get("content"), str)
                else " ".join(
                    p.get("text", "") for p in m.get("content", [])
                    if isinstance(p, dict)
                ) if isinstance(m, dict) and isinstance(m.get("content"), list)
                else str(getattr(m, "content", "") or ""),
                model,
            )
            for m in messages
        )
        if total <= token_budget:
            return

        # Collect indices of tool-result messages, oldest first.
        tool_indices = [
            i for i, m in enumerate(messages)
            if (m["role"] if isinstance(m, dict) else getattr(m, "role", "")) == "tool"
        ]
        truncation_notice = "[tool output truncated to fit context window]"
        for idx in tool_indices:
            if total <= token_budget:
                break
            msg = messages[idx]
            old_content = msg["content"] if isinstance(msg, dict) else msg.content
            if old_content == truncation_notice:
                continue
            old_tokens = _count_tokens(
                old_content if isinstance(old_content, str) else str(old_content),
                model,
            )
            new_tokens = _count_tokens(truncation_notice, model)
            if isinstance(msg, dict):
                msg["content"] = truncation_notice
            else:
                msg.content = truncation_notice
            total -= (old_tokens - new_tokens)
            logger.info("Trimmed tool result at index %d (saved ~%d tokens)", idx, old_tokens - new_tokens)

    async def _inference_loop(self, system_prompt: str, messages: list[dict],
                              thread_id: str = "default",
                              disable_tools: bool = False) -> LLMReply:
        """
        Repeatedly call the API until finish_reason is not 'tool_calls'.
        The system prompt is prepended as a role=system message each call
        (not stored in the DB so it stays fresh on every inference).

        Turing Protocol hooks are fired at three phases:
          - pre_execution:  before each tool call (can block execution)
          - post_execution: after each tool call (can flag results)
          - post_inference:  handled by the caller (_run_turing_check)
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        if disable_tools:
            tools = None
        elif nl_tools:
            tools = tool_module.get_tool_schemas(nl_tools=nl_tools)
        else:
            tools = tool_module.TOOL_SCHEMAS
        token_budget = self._cfg.context_size - self._cfg.max_tokens
        reasoning_parts: list[str] = []
        tool_calls_made: list[str] = []
        tool_call_details: list[dict] = []

        tp_enabled = self._turing_protocol_pool.enabled

        # Build a shared context for tool-call processing.
        async def _tp_check_main(phase, *, tool_name=None, tool_args=None,
                                 tool_result=None, assistant_response="",
                                 tool_calls_made=None, nl_tools=None):
            return await self._run_phase_check(
                phase=phase, scope="main", thread_id=thread_id,
                tool_calls_made=tool_calls_made or [],
                assistant_response=assistant_response,
                tool_name=tool_name, tool_args=tool_args,
                tool_result=tool_result, nl_tools=nl_tools,
            )

        tc_ctx = ToolCallContext(
            thread_id=thread_id,
            nesting_depth=0,
            scope="main",
            pool_last_used=self._main_pool.last_used,
            event_bus=self._event_bus,
            nl_enabled=nl_enabled,
            nl_tools=nl_tools,
            nl_translation_pool=getattr(self, "_nl_translation_pool", None),
            timezone_str=prompt_assembler._timezone,
            tp_enabled=tp_enabled,
            tp_check=_tp_check_main if tp_enabled else None,
        )

        while True:
            # Trim oldest tool results if accumulated context exceeds budget.
            self._trim_tool_results(full_messages, token_budget)

            response = await self._main_pool.call(
                messages=full_messages,
                tools=tools,
            )

            if not response.choices:
                logger.warning("LLM returned empty choices, retrying")
                logger.debug("Empty choices raw response: %s", response)
                full_messages.append({"role": "assistant", "content": ""})
                full_messages.append({"role": "user", "content": "Continue."})
                continue

            choice = response.choices[0]

            # Collect reasoning tokens from every round (including tool-use rounds).
            if self._cfg.reasoning:
                r = self._extract_reasoning(choice.message)
                if r:
                    reasoning_parts.append(r)

            if choice.message.tool_calls:
                logger.debug(
                    "Tool calls detected (finish_reason=%s): %s",
                    choice.finish_reason,
                    [tc.function.name for tc in choice.message.tool_calls],
                )
                # Log this inference round (intermediate, tool-use round).
                try:
                    _tc_names = [tc.function.name for tc in choice.message.tool_calls]
                    _round_content = (choice.message.content or "").strip()
                    await database.async_call(
                        database.save_interaction_log,
                        _time.time(), "inference_round", thread_id,
                        self._main_pool.last_used,
                        _round_content[:500] or f"[requesting {len(_tc_names)} tool call(s)]",
                        f"[tool_calls: {', '.join(_tc_names)}]",
                        "ok",
                        raw_output=json.dumps({
                            "tool_calls": [
                                {"name": tc.function.name, "arguments": tc.function.arguments}
                                for tc in choice.message.tool_calls
                            ],
                            "content": _round_content,
                        }),
                    )
                except Exception:
                    pass
                # Append the assistant's tool-call message.
                full_messages.append(choice.message)

                # Execute each tool via the shared pipeline.
                tc_ctx.pool_last_used = self._main_pool.last_used
                for tc in choice.message.tool_calls:
                    outcome = await process_tool_call(
                        tc, tc_ctx, tool_calls_made,
                        assistant_response=(choice.message.content or ""),
                    )
                    tool_call_details.extend(outcome.call_details)
                    full_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      outcome.content,
                    })
                continue  # next round

            # Terminal response.
            content = (choice.message.content or "").strip()
            reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
            return LLMReply(text=content, reasoning=reasoning,
                            tool_calls_made=tool_calls_made,
                            tool_call_details=tool_call_details)

    async def _run_phase_check(
        self,
        phase: str,
        scope: str,
        thread_id: str,
        tool_calls_made: list[str],
        assistant_response: str = "",
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
        nl_tools: "set[str] | None" = None,
    ) -> Optional["turing_protocol_module.TuringResult"]:
        """Run Turing Protocol hooks for a specific phase/scope.

        Returns the TuringResult if any violations are confirmed, None otherwise.
        Used by _inference_loop for pre/post_execution hooks.
        """
        # Quick check: are there any hooks for this phase+scope?
        hooks = turing_protocol_module.get_hooks(
            self._turing_protocol_validators,
            phase_filter=phase,
            scope_filter=scope,
        )
        if not hooks:
            return None

        try:
            return await turing_protocol_module.run_turing_protocol(
                pool=self._turing_protocol_pool,
                user_message="",
                assistant_response=assistant_response,
                tool_calls_made=tool_calls_made,
                active_sessions=[],
                enabled_validators=self._turing_protocol_validators,
                thread_id=thread_id,
                phase=phase,
                scope=scope,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                nl_tools=nl_tools,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Turing Protocol %s check raised (non-fatal)", phase)
            return None

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _compact_context(self, thread_id: str = "default") -> None:
        rows = await database.async_call(database.load_active_messages, thread_id)
        if len(rows) <= COMPACTION_KEEP_RECENT:
            return

        to_summarise = rows[:-COMPACTION_KEEP_RECENT]

        # Include the previous compaction summary so information chains
        # across compaction cycles instead of being lost.
        prior_summary = self._compaction_summaries.get(thread_id)
        parts = []
        if prior_summary:
            parts.append(f"[PRIOR SUMMARY]\n{prior_summary}\n")
        parts.append("[NEW MESSAGES]\n" + "\n".join(
            f"{r['role'].upper()}: {r['content']}" for r in to_summarise
        ))
        history_text = "\n\n".join(parts)

        summary_prompt = prompt_loader.load("COMPACTION_PROMPT.txt", history=history_text)

        try:
            summary_response = await self._compaction_pool.call(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens_override=2048,
            )
            summary = (summary_response.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            logger.exception("Compaction failed for thread %s — skipping", thread_id)
            return

        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(), "compaction", thread_id,
                self._compaction_pool.last_used,
                summary_prompt, summary, "ok",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save compaction interaction log", exc_info=True)

        if to_summarise:
            await database.async_call(database.archive_messages, to_summarise[-1]["id"], thread_id)
        await database.async_call(database.save_summary, summary, thread_id)
        self._compaction_summaries[thread_id] = summary

        logger.info("Compacted %d messages into summary (%d chars) for thread %s",
                     len(to_summarise), len(summary), thread_id)
        try:
            await self._broadcast(
                "\U0001f4e6 Context compacted: old messages archived and summarised.",
                thread_id,
            )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Component size monitoring
    # ------------------------------------------------------------------

    async def _maybe_summarise_components(self, thread_id: str = "default",
                                            *, _from_system_event: bool = False) -> None:
        if _from_system_event:
            return  # Avoid self-reinforcing summarisation loops.
        sizes = prompt_assembler.check_component_sizes()
        for component, oversized in sizes.items():
            if not oversized:
                continue
            logger.info("Component '%s' oversized – requesting AI summarisation", component)
            prompt = prompt_loader.load("COMPONENT_OVERSIZE.txt", component=component)
            try:
                await self.enqueue_system_event(prompt, thread_id)
                await self._broadcast(
                    f"\u2139\ufe0f Auto-summarising {component} (size limit reached).",
                    thread_id,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Could not request summarisation for %s", component)

    # ------------------------------------------------------------------
    # Message list construction
    # ------------------------------------------------------------------

    async def _build_messages(self, new_text: str, is_system_event: bool,
                              thread_id: str = "default",
                              content: Optional[list] = None) -> list[dict]:
        rows = await database.async_call(database.load_active_messages, thread_id)
        messages = [
            {"role": r["role"], "content": r["content"] or "..."}
            for r in rows
        ]
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        if content is not None and not is_system_event:
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": f"{prefix}{new_text}"})
        return messages
