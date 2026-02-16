"""
LLM Inference Thread

Owns the conversation history and all interactions with any OpenAI-compatible
API endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).

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
import time as _time
from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path

from wintermute import database
from wintermute import prompt_assembler
from wintermute import supervisor as supervisor_module
from wintermute import tools as tool_module

# Maximum number of consecutive supervisor corrections per turn.
# After this many corrections the supervisor gives up to prevent infinite loops.
MAX_CORRECTION_DEPTH = 2
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

# Keep the last N messages untouched during compaction.
COMPACTION_KEEP_RECENT = 10

COMPACTION_PROMPT_FILE = Path("data") / "COMPACTION_PROMPT.txt"

_DEFAULT_COMPACTION_PROMPT = (
    "Summarise the following into a single structured reference document. "
    "This summary will be injected into a system prompt, so optimise for quick "
    "scanning by an AI assistant.\n\n"
    "The input may contain a [PRIOR SUMMARY] section (from an earlier compaction) "
    "followed by [NEW MESSAGES]. Merge both into one cohesive summary — do not "
    "keep them separate. When new information supersedes prior summary content, "
    "keep only the latest state.\n\n"
    "Preserve: decisions made and their reasoning, facts learned about the user, "
    "task outcomes and current status of ongoing work, file paths, URLs, "
    "credentials references, and technical details mentioned, any commitments "
    "or promises made to the user.\n\n"
    "Omit: greetings, pleasantries, filler, superseded information (only keep "
    "the latest state), tool call details unless the outcome matters.\n\n"
    "Format as a concise bulleted list grouped by topic. Do not add commentary.\n\n"
    "--- conversation history ---\n"
)


def _load_compaction_prompt() -> str:
    """Load the compaction prompt from a file, falling back to the built-in default."""
    try:
        text = COMPACTION_PROMPT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.error("Cannot read %s: %s", COMPACTION_PROMPT_FILE, exc)
    return _DEFAULT_COMPACTION_PROMPT


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
    provider: str = "openai"  # "openai", "gemini-cli", or "kimi-code"
    api_key: str = ""
    base_url: str = ""       # e.g. http://localhost:11434/v1  or  https://api.openai.com/v1



class BackendPool:
    """Ordered list of LLM backends for a role, with automatic failover.

    On API errors the next backend in the list is tried automatically.
    An empty pool (``len(pool) == 0``) signals "disabled" — relevant for
    optional roles like supervisor.
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

    # -- API call with failover -----------------------------------------------

    async def call(self, *, messages: list[dict],
                   tools: "list[dict] | None" = None,
                   max_tokens_override: "int | None" = None,
                   **extra_kwargs) -> object:
        """Call ``chat.completions.create`` with automatic failover.

        Each backend uses its own model, max_tokens, and reasoning setting.
        *max_tokens_override* replaces the backend's configured max_tokens
        (useful for compaction's hard-coded 2048).
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
            try:
                result = await client.chat.completions.create(**call_kwargs)
                self.last_used = cfg.name
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                backend_desc = f"'{cfg.name}' ({cfg.model})"
                if len(self._backends) > 1:
                    logger.warning("Backend %s failed: %s — trying next", backend_desc, exc)
                else:
                    raise
        raise last_error  # type: ignore[misc]


@dataclass
class MultiProviderConfig:
    """Per-purpose LLM backend pools.

    Configured via ``inference_backends`` (named backend definitions) and
    ``llm`` (role-to-backend-name mapping).  Each field holds an ordered
    list of ProviderConfig objects; the runtime BackendPool is built from
    these lists plus the corresponding clients.

    An empty list for *supervisor* means "disabled".
    """
    main: list[ProviderConfig]
    compaction: list[ProviderConfig]
    sub_sessions: list[ProviderConfig]
    dreaming: list[ProviderConfig]
    supervisor: list[ProviderConfig]


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
    is_supervisor_correction: bool = False  # marks items injected by supervisor
    # How many consecutive supervisor corrections have been issued for this
    # turn.  The supervisor will re-check responses up to MAX_CORRECTION_DEPTH
    # times before giving up (prevents infinite loops while still catching
    # persistent hallucinations).
    correction_depth: int = 0
    # Sequence number of the user-message turn this correction was issued for.
    # If the thread has advanced past this number by the time the correction is
    # dequeued, the correction is stale and will be dropped.
    correction_for_seq: Optional[int] = None


class LLMThread:
    """Runs as an asyncio task within the shared event loop."""

    def __init__(self, main_pool: BackendPool, compaction_pool: BackendPool,
                 supervisor_pool: BackendPool, broadcast_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._main_pool = main_pool
        self._compaction_pool = compaction_pool
        self._supervisor_pool = supervisor_pool
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
        # supervisor corrections that arrived after the conversation moved on.
        self._thread_seq: dict[str, int] = {}

    def inject_sub_session_manager(self, manager: "SubSessionManager") -> None:
        """Called after construction once SubSessionManager is built."""
        self._sub_sessions = manager

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def enqueue_user_message(self, text: str, thread_id: str = "default") -> "LLMReply":
        """Submit a user message and await the AI reply (returns LLMReply)."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id, future=fut))
        return await fut

    async def enqueue_system_event(self, text: str, thread_id: str = "default") -> None:
        """Submit an autonomous system event (heartbeat, reminder, etc.)."""
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id, is_system_event=True))

    async def enqueue_system_event_with_reply(self, text: str,
                                              thread_id: str = "default") -> str:
        """Submit a system event and await the AI reply.

        Like enqueue_system_event but returns the reply text. The prompt is
        NOT saved to the DB as a user message; only the assistant response is.
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_QueueItem(text=text, thread_id=thread_id,
                                         is_system_event=True, future=fut))
        return await fut

    async def reset_session(self, thread_id: str = "default") -> None:
        database.clear_active_messages(thread_id)
        self._compaction_summaries.pop(thread_id, None)
        if self._sub_sessions:
            n = self._sub_sessions.cancel_for_thread(thread_id)
            if n:
                logger.info("Cancelled %d sub-session(s) for reset thread %s", n, thread_id)
        logger.info("Session reset for thread %s", thread_id)

    async def force_compact(self, thread_id: str = "default") -> None:
        await self._compact_context(thread_id)

    def get_compaction_summary(self, thread_id: str = "default") -> Optional[str]:
        """Return the current in-memory compaction summary for a thread, or None."""
        return self._compaction_summaries.get(thread_id)

    def get_token_budget(self, thread_id: str = "default") -> dict:
        """Return precise token accounting for a thread."""
        total_limit = max(self._cfg.context_size - self._cfg.max_tokens, 1)
        model = self._cfg.model

        summary = self._compaction_summaries.get(thread_id)
        try:
            sp_text = prompt_assembler.assemble(extra_summary=summary)
        except Exception:  # noqa: BLE001
            sp_text = ""
        sp_tokens = _count_tokens(sp_text, model)

        tools_tokens = _count_tokens(json.dumps(tool_module.TOOL_SCHEMAS), model)

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
        for tid in database.get_active_thread_ids():
            summary = database.load_latest_summary(tid)
            if summary:
                self._compaction_summaries[tid] = summary
        logger.info("LLM thread started (endpoint=%s model=%s)", self._cfg.base_url, self._cfg.model)
        if self._supervisor_pool.enabled:
            logger.info("Supervisor enabled (model=%s)", self._supervisor_pool.primary.model)
        else:
            logger.info("Supervisor disabled")

        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Drop supervisor corrections that are stale — i.e. the thread has
            # already processed at least one new user-facing turn since the
            # correction was issued, making it no longer relevant.
            if item.is_supervisor_correction and item.correction_for_seq is not None:
                current_seq = self._thread_seq.get(item.thread_id, 0)
                if current_seq > item.correction_for_seq:
                    logger.warning(
                        "Dropping stale supervisor correction for thread %s "
                        "(issued at seq=%d, current seq=%d)",
                        item.thread_id, item.correction_for_seq, current_seq,
                    )
                    self._queue.task_done()
                    continue

            # Advance the per-thread sequence counter for every user-facing turn.
            if not item.is_system_event:
                self._thread_seq[item.thread_id] = (
                    self._thread_seq.get(item.thread_id, 0) + 1
                )

            try:
                reply = await self._process(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM processing error")
                try:
                    database.save_interaction_log(
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
            elif item.is_system_event and not item.future and not item.is_supervisor_correction:
                # System events without a future (sub-session results,
                # reminders, /pulse commands) have no caller waiting for
                # the reply.  Broadcast the LLM's response directly so
                # it reaches the user.
                # Supervisor corrections are excluded: both the correction
                # prompt and the model's response to it stay in the DB only
                # (visible in the web debug interface but silent in Matrix).
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

            # -- Supervisor workflow validation --
            # Fire async after the reply is delivered.  Only check
            # user-facing messages (not system events, not sub-session
            # threads, and not the supervisor's own corrections).
            if (
                not item.thread_id.startswith("sub_")
                and reply.text
                and (
                    # Normal user message — always check
                    (not item.is_system_event and not item.is_supervisor_correction)
                    # Supervisor correction response — re-check up to depth limit
                    or (item.is_supervisor_correction
                        and item.correction_depth < MAX_CORRECTION_DEPTH)
                )
            ):
                # Snapshot the current sequence number so the correction can
                # be dropped if the conversation has moved on before it lands.
                seq_at_fire = self._thread_seq.get(item.thread_id, 0)
                asyncio.create_task(
                    self._run_supervisor_check(
                        user_message=item.text,
                        assistant_response=reply.text,
                        tool_calls_made=reply.tool_calls_made,
                        thread_id=item.thread_id,
                        issued_for_seq=seq_at_fire,
                        correction_depth=item.correction_depth,
                    ),
                    name=f"supervisor_{item.thread_id}",
                )

            self._queue.task_done()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Supervisor workflow validation
    # ------------------------------------------------------------------

    async def _run_supervisor_check(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls_made: list[str],
        thread_id: str,
        issued_for_seq: int = 0,
        correction_depth: int = 0,
    ) -> None:
        """Fire a one-shot supervisor agent to detect hallucinated workflow spawns.

        Runs asynchronously after the main reply has already been delivered.
        If the supervisor detects a mismatch it enqueues a corrective system
        event.  ``correction_depth`` tracks how many consecutive corrections
        have been issued for this turn; the gate in ``_loop`` stops re-checking
        once ``MAX_CORRECTION_DEPTH`` is reached.

        ``issued_for_seq`` records the per-thread sequence number at the time
        the check was fired.  If the thread has advanced past this number by
        the time the correction is dequeued, the correction is dropped as
        stale (the user has already sent a follow-up and the context has moved
        on).
        """
        if not self._supervisor_pool.enabled or not self._sub_sessions:
            return

        active_sessions = self._sub_sessions.list_active()

        try:
            correction = await supervisor_module.check_workflow_consistency(
                pool=self._supervisor_pool,
                user_message=user_message,
                assistant_response=assistant_response,
                tool_calls_made=tool_calls_made,
                active_sessions=active_sessions,
                correction_depth=correction_depth,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Supervisor check raised (non-fatal)")
            return

        if correction:
            new_depth = correction_depth + 1
            if new_depth >= MAX_CORRECTION_DEPTH:
                logger.warning(
                    "Supervisor injecting FINAL correction (depth %d/%d) "
                    "into thread %s — no further re-checks will fire",
                    new_depth, MAX_CORRECTION_DEPTH, thread_id,
                )
            else:
                logger.info(
                    "Supervisor injecting correction (depth %d/%d) into thread %s",
                    new_depth, MAX_CORRECTION_DEPTH, thread_id,
                )
            await self._queue.put(_QueueItem(
                text=correction,
                thread_id=thread_id,
                is_system_event=True,
                is_supervisor_correction=True,
                correction_depth=correction_depth + 1,
                correction_for_seq=issued_for_seq,
            ))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _process(self, item: _QueueItem) -> LLMReply:
        thread_id = item.thread_id
        messages = self._build_messages(item.text, item.is_system_event, thread_id)

        # Assemble system prompt first so we can measure its real token cost.
        summary = self._compaction_summaries.get(thread_id)
        system_prompt = prompt_assembler.assemble(extra_summary=summary)
        overhead_tokens = (
            _count_tokens(system_prompt, self._cfg.model)
            + _count_tokens(json.dumps(tool_module.TOOL_SCHEMAS), self._cfg.model)
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
            messages = self._build_messages(item.text, item.is_system_event, thread_id)
            # Reassemble with the updated compaction summary.
            summary = self._compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(extra_summary=summary)

        is_sub_session_result = item.is_system_event and "[SUB-SESSION " in item.text
        if not item.is_system_event:
            database.save_message("user", item.text, thread_id)
        elif is_sub_session_result:
            database.save_message("user", f"[SYSTEM EVENT] {item.text}", thread_id)
        elif item.is_supervisor_correction:
            # Save the correction prompt to the DB so it is visible in the web
            # debug interface, but do NOT broadcast it to Matrix (see broadcast
            # guard below).  The model's reply is saved normally via the
            # assistant save below.
            database.save_message("user", item.text, thread_id)

        reply = await self._inference_loop(
            system_prompt, messages, thread_id,
            disable_tools=is_sub_session_result,
        )

        # Determine action type for interaction log
        if item.is_supervisor_correction:
            _action = "supervisor"
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
            database.save_interaction_log(
                _time.time(), _action, thread_id,
                self._main_pool.last_used,
                item.text, reply.text, "ok",
                raw_output=json.dumps(raw_output_data),
            )
            # Log each individual tool call as a separate entry
            for tc in reply.tool_call_details:
                database.save_interaction_log(
                    _time.time(), "tool_call", thread_id,
                    self._main_pool.last_used,
                    json.dumps({"tool": tc["name"], "arguments": tc["arguments"]}),
                    tc.get("result", ""),
                    "ok",
                )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save interaction log entry", exc_info=True)

        database.save_message("assistant", reply.text, thread_id)
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

    async def _inference_loop(self, system_prompt: str, messages: list[dict],
                              thread_id: str = "default",
                              disable_tools: bool = False) -> LLMReply:
        """
        Repeatedly call the API until finish_reason is not 'tool_calls'.
        The system prompt is prepended as a role=system message each call
        (not stored in the DB so it stays fresh on every inference).
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        tools = None if disable_tools else tool_module.TOOL_SCHEMAS
        reasoning_parts: list[str] = []
        tool_calls_made: list[str] = []
        tool_call_details: list[dict] = []

        while True:
            response = await self._main_pool.call(
                messages=full_messages,
                tools=tools,
            )

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
                # Append the assistant's tool-call message.
                full_messages.append(choice.message)

                # Execute each tool and collect results.
                for tc in choice.message.tool_calls:
                    tool_calls_made.append(tc.function.name)
                    try:
                        inputs = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        inputs = {}
                    result = tool_module.execute_tool(tc.function.name, inputs,
                                                     thread_id=thread_id,
                                                     nesting_depth=0)
                    tool_call_details.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                        "result": result[:500],
                    })
                    logger.debug("Tool %s -> %s", tc.function.name, result[:200])
                    full_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue  # next round

            # Terminal response.
            content = (choice.message.content or "").strip()
            reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
            return LLMReply(text=content, reasoning=reasoning,
                            tool_calls_made=tool_calls_made,
                            tool_call_details=tool_call_details)

    # ------------------------------------------------------------------
    # Context compaction
    # ------------------------------------------------------------------

    async def _compact_context(self, thread_id: str = "default") -> None:
        rows = database.load_active_messages(thread_id)
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

        compaction_prompt = _load_compaction_prompt()
        if "{history}" in compaction_prompt:
            summary_prompt = compaction_prompt.format(history=history_text)
        else:
            summary_prompt = compaction_prompt + history_text

        summary_response = await self._compaction_pool.call(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens_override=2048,
        )
        summary = (summary_response.choices[0].message.content or "").strip()

        try:
            database.save_interaction_log(
                _time.time(), "compaction", thread_id,
                self._compaction_pool.last_used,
                summary_prompt, summary, "ok",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save compaction interaction log", exc_info=True)

        if to_summarise:
            database.archive_messages(to_summarise[-1]["id"], thread_id)
        database.save_summary(summary, thread_id)
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
            prompt = (
                f"The {component} section of your memory has grown large. "
                f"Read its current content via read_file, condense and prioritise it, "
                f"then update it using the appropriate tool."
            )
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

    def _build_messages(self, new_text: str, is_system_event: bool,
                        thread_id: str = "default") -> list[dict]:
        rows = database.load_active_messages(thread_id)
        messages = [{"role": r["role"], "content": r["content"]} for r in rows]
        prefix = "[SYSTEM EVENT] " if is_system_event else ""
        messages.append({"role": "user", "content": f"{prefix}{new_text}"})
        return messages
