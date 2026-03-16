"""
LLM Inference Thread

Queue-based inference orchestrator.  Receives user messages via an asyncio
Queue, runs inference (including multi-step tool-use loops), and delivers
responses back through reply Futures.

Conversation persistence, context compaction, and session lifecycle are
delegated to ``ConversationStore``, ``ContextCompactor``, and
``SessionManager`` respectively.

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
import re
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.infra import prompt_loader
from wintermute.core import convergence_protocol as convergence_protocol_module
from wintermute.core.inference_engine import (
    ToolCallContext, extract_content_text, make_tool_context,
    process_tool_call,
)
from wintermute.core.types import (  # noqa: F401 — re-exported for backwards compat
    BackendPool,
    ContextTooLargeError,
    LLMBackend,
    LLMResponse,
    MultiProviderConfig,
    ProviderConfig,
    RateLimitError,
    classify_api_error,
)
from wintermute.core.tool_call_rescue import rescue_tool_calls
from wintermute import tools as tool_module

from wintermute.core.tool_deps import ToolDeps
from wintermute.core.conversation_store import ConversationStore, count_tokens
from wintermute.core.context_compactor import ContextCompactor, COMPACTION_KEEP_RECENT
from wintermute.core.session_manager import SessionManager
if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.infra.event_bus import EventBus
    from wintermute.infra.thread_config import ThreadConfigManager, ResolvedThreadConfig

logger = logging.getLogger(__name__)

# Maximum consecutive empty-choices responses before aborting the inference loop.
MAX_EMPTY_RETRIES = 3

# Backwards-compat alias: external consumers (web_interface) import this.
_count_tokens = count_tokens


@dataclass
class LLMReply:
    """Response from the LLM, separating visible content from reasoning tokens."""
    text: str
    reasoning: Optional[str] = None  # reasoning/thinking tokens (if model supports it)
    tool_calls_made: list[str] = field(default_factory=list)  # tool names called during inference
    tool_call_details: list[dict] = field(default_factory=list)  # [{name, arguments, result}, ...]
    duration_seconds: Optional[float] = None   # wall-clock inference time
    backend_used: Optional[str] = None         # model/backend that served the request

    def __str__(self) -> str:
        return self.text


@dataclass
class _QueueItem:
    text: str
    thread_id: str = "default"
    is_system_event: bool = False
    future: Optional[asyncio.Future] = field(default=None, compare=False)
    convergence_depth: int = 0  # 0=normal, 1=first correction, 2=re-check correction (max)
    content: Optional[list] = None  # multimodal content parts (OpenAI vision format)
    # Sequence number of the user-message turn this correction was issued for.
    # If the thread has advanced past this number by the time the correction is
    # dequeued, the correction is stale and will be dropped.
    correction_for_seq: Optional[int] = None


class LLMThread:
    """Runs as an asyncio task within the shared event loop."""

    def __init__(self, main_pool: BackendPool, compaction_pool: BackendPool,
                 convergence_protocol_pool: BackendPool, broadcast_fn,
                 sub_session_getter: "Optional[Callable[[], Optional[SubSessionManager]]]" = None,
                 convergence_protocol_validators: "Optional[dict[str, bool]]" = None,
                 nl_translation_pool: "Optional[BackendPool]" = None,
                 nl_translation_config: "Optional[dict]" = None,
                 seed_language: str = "en",
                 event_bus: "Optional[EventBus]" = None,
                 thread_config_manager: "Optional[ThreadConfigManager]" = None,
                 backend_pools_by_name: "Optional[dict[str, BackendPool]]" = None,
                 compaction_keep_recent: int = COMPACTION_KEEP_RECENT,
                 tool_deps: "Optional[ToolDeps]" = None) -> None:
        self._main_pool = main_pool
        self._convergence_protocol_pool = convergence_protocol_pool
        from wintermute.core.cp_runner import ConvergenceProtocolRunner
        self._cp_runner = ConvergenceProtocolRunner(
            pool=convergence_protocol_pool,
            scope="main",
            enabled_validators=convergence_protocol_validators,
        )
        self._nl_translation_pool = nl_translation_pool
        self._nl_translation_config = nl_translation_config or {}
        self._seed_language = seed_language
        self._event_bus = event_bus
        self._tool_deps = tool_deps
        # Convenience: primary config for context_size / model name lookups.
        self._cfg = main_pool.primary
        self._broadcast = broadcast_fn  # async callable(text, thread_id, *, reasoning=None)
        self._get_sub_sessions = sub_session_getter  # lazy getter breaks LLMThread↔SSM cycle
        self._running = False
        self._background_tasks: set[asyncio.Task] = set()
        # Per-thread queues and worker tasks for concurrent thread processing.
        self._queues: dict[str, asyncio.Queue[_QueueItem]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._queues_lock = asyncio.Lock()
        # Idle timeout (seconds) before a per-thread worker self-terminates.
        self._worker_idle_timeout = 300.0
        # Gate: workers wait on this before processing so summaries are loaded.
        self._ready = asyncio.Event()

        # --- Composed components ---
        self._store = ConversationStore(
            primary_config=main_pool.primary,
            event_bus=event_bus,
            nl_translation_config=self._nl_translation_config,
            tool_deps=tool_deps,
        )
        self._compactor = ContextCompactor(
            compaction_pool=compaction_pool,
            broadcast_fn=broadcast_fn,
            store=self._store,
            keep_recent=compaction_keep_recent,
            enqueue_system_event_fn=self.enqueue_system_event,
            event_bus=event_bus,
        )
        self._session_mgr = SessionManager(
            main_pool=main_pool,
            thread_config_manager=thread_config_manager,
            backend_pools_by_name=backend_pools_by_name,
            sub_session_getter=sub_session_getter,
            store=self._store,
        )

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
        return self._compactor.pool

    @property
    def convergence_protocol_pool(self) -> BackendPool:
        return self._convergence_protocol_pool

    @property
    def nl_translation_pool(self) -> "Optional[BackendPool]":
        return self._nl_translation_pool

    @property
    def queue_size(self) -> int:
        return sum(q.qsize() for q in list(self._queues.values()))

    @property
    def thread_config_manager(self) -> "Optional[ThreadConfigManager]":
        return self._session_mgr.thread_config_manager

    @property
    def session_manager(self) -> "SessionManager":
        return self._session_mgr

    def check_session_timeouts(self) -> list[str]:
        return self._session_mgr.check_session_timeouts()

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
        await self._dispatch(_QueueItem(
            text=text, thread_id=thread_id, future=fut, content=content,
        ))
        return await fut

    async def enqueue_system_event(self, text: str, thread_id: str = "default") -> None:
        """Submit an autonomous system event (heartbeat, scheduled task, etc.)."""
        await self._dispatch(_QueueItem(text=text, thread_id=thread_id, is_system_event=True))

    async def enqueue_system_event_with_reply(self, text: str,
                                              thread_id: str = "default") -> str:
        """Submit a system event and await the AI reply.

        Like enqueue_system_event but returns the reply text. The prompt is
        NOT saved to the DB as a user message; only the assistant response is.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._dispatch(_QueueItem(text=text, thread_id=thread_id,
                                        is_system_event=True, future=fut))
        return await fut

    async def reset_session(self, thread_id: str = "default") -> None:
        await self._session_mgr.reset_session(thread_id)

    async def store_message_silent(self, text: str, thread_id: str = "default") -> None:
        """Store a user message without triggering inference (group mode)."""
        token_count = count_tokens(text, self._cfg.model)
        await database.async_call(
            database.save_message, "user", text, thread_id,
            token_count=token_count,
        )
        self._session_mgr.record_activity(thread_id)
        if self._store._event_bus:
            self._store._event_bus.emit("message.received", thread_id=thread_id, text=text)

    async def force_compact(self, thread_id: str = "default") -> None:
        await self._compactor.compact(thread_id)

    def get_compaction_summary(self, thread_id: str = "default") -> Optional[str]:
        return self._store.get_compaction_summary(thread_id)

    def get_last_system_prompt(self, thread_id: str = "default") -> Optional[str]:
        return self._store.get_last_system_prompt(thread_id)

    def get_token_budget(self, thread_id: str = "default") -> dict:
        return self._store.get_token_budget(thread_id)

    # ------------------------------------------------------------------
    # Dispatcher and per-thread worker infrastructure
    # ------------------------------------------------------------------

    async def _dispatch(self, item: _QueueItem) -> None:
        """Route a queue item to the per-thread queue, spawning a worker if needed."""
        tid = item.thread_id
        async with self._queues_lock:
            queue = self._queues.get(tid)
            if queue is None:
                queue = asyncio.Queue()
                self._queues[tid] = queue
            # (Re)spawn worker if missing or finished (idle exit / crash).
            worker = self._workers.get(tid)
            if worker is None or worker.done():
                task = asyncio.create_task(
                    self._thread_worker(tid), name=f"llm_worker_{tid}",
                )
                self._workers[tid] = task
                logger.debug("Spawned per-thread worker for %s", tid)
        await queue.put(item)

    async def _cleanup_worker(self, thread_id: str) -> bool:
        """Remove queue/worker entry for an idle thread if still owned by this worker.

        Returns True if cleanup succeeded (worker should exit).  Returns False
        if new work arrived in the queue — the worker should keep running.
        """
        current_task = asyncio.current_task()
        async with self._queues_lock:
            registered_worker = self._workers.get(thread_id)
            if registered_worker is not current_task:
                return True  # another worker was spawned — just exit
            queue = self._queues.get(thread_id)
            if queue is not None and not queue.empty():
                return False  # new work arrived — caller should keep running
            self._queues.pop(thread_id, None)
            self._workers.pop(thread_id, None)
            return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        await self._store.load_summaries()
        self._ready.set()
        logger.info("LLM thread started (endpoint=%s model=%s)", self._cfg.base_url, self._cfg.model)
        if self._convergence_protocol_pool.enabled:
            logger.info("Convergence Protocol enabled (model=%s)", self._convergence_protocol_pool.primary.model)
        else:
            logger.info("Convergence Protocol disabled")

        # The run loop waits for shutdown.  Per-thread workers are spawned
        # on demand by _dispatch() when messages arrive.
        try:
            while self._running:
                await asyncio.sleep(1.0)
        finally:
            # Cancel all per-thread workers and background tasks.
            # Snapshot to avoid RuntimeError from concurrent dict mutation.
            worker_tasks = list(self._workers.values())
            bg_tasks = list(self._background_tasks)
            for task in worker_tasks:
                task.cancel()
            for task in bg_tasks:
                task.cancel()
            all_tasks = worker_tasks + bg_tasks
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)

    async def _thread_worker(self, thread_id: str) -> None:
        """Per-thread consumer loop.  Processes items strictly in order.

        Self-terminates after ``_worker_idle_timeout`` seconds of inactivity,
        cleaning up its queue and worker entry so resources are not leaked for
        idle threads.  Runs until cancelled (not tied to ``_running`` flag) so
        items enqueued before ``run()`` starts are still processed.
        """
        queue = self._queues[thread_id]
        logger.debug("Worker started for thread %s", thread_id)
        # Wait for run() to finish loading summaries before processing.
        await self._ready.wait()

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=self._worker_idle_timeout,
                    )
                except asyncio.TimeoutError:
                    # Attempt cleanup; if new work arrived, keep running.
                    if await self._cleanup_worker(thread_id):
                        logger.debug("Worker idle timeout for thread %s — exiting", thread_id)
                        return  # cleanup succeeded, exit without finally cleanup
                    continue  # new items in queue — keep processing

                # Drop stale Convergence Protocol corrections.
                if item.convergence_depth > 0 and item.correction_for_seq is not None:
                    current_seq = self._store.thread_seq.get(thread_id, 0)
                    if current_seq > item.correction_for_seq:
                        logger.warning(
                            "Dropping stale Convergence Protocol correction for thread %s "
                            "(issued at seq=%d, current seq=%d)",
                            thread_id, item.correction_for_seq, current_seq,
                        )
                        try:
                            await database.async_call(
                                database.save_interaction_log,
                                _time.time(), "convergence_stale_drop", thread_id,
                                self._main_pool.last_used,
                                item.text[:2000], "", "stale",
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug("Failed to log stale drop", exc_info=True)
                        queue.task_done()
                        continue

                # Advance per-thread sequence counter.
                if not item.is_system_event or item.convergence_depth > 0:
                    self._store.thread_seq[thread_id] = (
                        self._store.thread_seq.get(thread_id, 0) + 1
                    )

                # Seed empty threads on first real user message.
                if not item.is_system_event and not await database.async_call(database.thread_has_messages, thread_id):
                    try:
                        seed_prompt = prompt_loader.load_seed(self._seed_language)
                        seed_item = _QueueItem(
                            text=seed_prompt,
                            thread_id=thread_id,
                            is_system_event=True,
                        )
                        seed_reply = await self._process(seed_item)
                        if seed_reply.text:
                            try:
                                await self._broadcast(
                                    seed_reply.text, thread_id,
                                    reasoning=seed_reply.reasoning,
                                )
                            except Exception:  # noqa: BLE001
                                logger.exception("Failed to broadcast seed reply")
                    except Exception:  # noqa: BLE001
                        logger.exception("Seed injection failed (non-fatal)")

                # Collect recent assistant messages BEFORE _process().
                _prior_assistant = None
                _recent_assistant: list[str] = []
                try:
                    _db_msgs = await database.async_call(database.load_active_messages, thread_id)
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
                            _time.time(), "chat", thread_id,
                            self._main_pool.last_used,
                            item.text, str(exc), "error",
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to log chat error", exc_info=True)
                    err_msg = str(exc)
                    if "401" in err_msg and "Gemini" in err_msg:
                        err_msg += (
                            "\n\nGemini credentials have expired or are invalid. "
                            "Re-run authentication on the server:\n"
                            "  uv run python -m wintermute.gemini_auth"
                        )
                    reply = LLMReply(text=f"[Error during inference: {err_msg}]")

                if item.future and not item.future.done():
                    item.future.set_result(reply)
                elif item.is_system_event and not item.future and item.convergence_depth == 0:
                    text_to_send = reply.text or item.text
                    logger.info(
                        "Broadcasting system-event reply for thread %s (%d chars, reply_empty=%s)",
                        thread_id, len(text_to_send), not reply.text,
                    )
                    try:
                        await self._broadcast(text_to_send, thread_id,
                                              reasoning=reply.reasoning)
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to broadcast system-event reply for thread %s",
                                         thread_id)
                elif item.convergence_depth > 0 and reply.text:
                    logger.info(
                        "Broadcasting Convergence correction response for thread %s "
                        "(depth=%d, tools=%s)",
                        thread_id, item.convergence_depth,
                        reply.tool_calls_made or "none",
                    )
                    try:
                        await self._broadcast(reply.text, thread_id,
                                              reasoning=reply.reasoning)
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to broadcast Convergence correction response")

                # -- Convergence Protocol validation --
                if (
                    not thread_id.startswith("sub_")
                    and reply.text
                    and item.convergence_depth < 2
                ):
                    seq_at_fire = self._store.thread_seq.get(thread_id, 0)
                    _prior_tc = self._session_mgr.prior_tool_calls.get(thread_id, [])
                    _cp_task = asyncio.create_task(
                        self._run_convergence_check(
                            user_message=item.text,
                            assistant_response=reply.text,
                            tool_calls_made=reply.tool_calls_made,
                            thread_id=thread_id,
                            issued_for_seq=seq_at_fire,
                            convergence_depth=item.convergence_depth,
                            prior_assistant_message=_prior_assistant,
                            prior_tool_calls_made=_prior_tc,
                            recent_assistant_messages=_recent_assistant,
                        ),
                        name=f"convergence_{thread_id}",
                    )
                    self._background_tasks.add(_cp_task)
                    _cp_task.add_done_callback(self._background_tasks.discard)

                # Update prior-turn tool calls.
                self._session_mgr.prior_tool_calls[thread_id] = reply.tool_calls_made or []

                # Emit main-thread turn event.
                if (
                    self._event_bus
                    and not thread_id.startswith("sub_")
                    and not item.is_system_event
                ):
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
                                    m = re.search(r'data/skills/([^/]+)\.md', p)
                                    if m:
                                        skills_loaded.append(m.group(1))
                            except Exception:
                                pass
                    self._event_bus.emit(
                        "main_thread.turn_completed",
                        thread_id=thread_id,
                        tools_used=reply.tool_calls_made,
                        skills_loaded=skills_loaded,
                        had_error="[Error" in (reply.text or ""),
                    )

                # Record main-thread turn in outcomes table.
                if (
                    not thread_id.startswith("sub_")
                    and not item.is_system_event
                    and item.convergence_depth == 0
                ):
                    _had_error = "[Error" in (reply.text or "")
                    try:
                        await database.async_call(
                            database.save_sub_session_outcome,
                            session_id=thread_id,
                            timestamp=_time.time(),
                            objective=item.text[:200] if item.text else "",
                            system_prompt_mode="main_thread",
                            tools_used=reply.tool_calls_made or None,
                            tool_call_count=len(reply.tool_call_details) if reply.tool_call_details else 0,
                            duration_seconds=reply.duration_seconds,
                            status="error" if _had_error else "completed",
                            result_length=len(reply.text) if reply.text else 0,
                            backend_used=reply.backend_used,
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to save main-thread outcome", exc_info=True)

                queue.task_done()

        except asyncio.CancelledError:
            logger.debug("Worker cancelled for thread %s", thread_id)
            await self._cleanup_worker(thread_id)
        except Exception:  # noqa: BLE001
            logger.exception("Worker crashed for thread %s — will respawn on next message", thread_id)
            await self._cleanup_worker(thread_id)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Convergence Protocol validation
    # ------------------------------------------------------------------

    async def _run_convergence_check(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls_made: list[str],
        thread_id: str,
        issued_for_seq: int = 0,
        convergence_depth: int = 0,
        prior_assistant_message: Optional[str] = None,
        prior_tool_calls_made: Optional[list[str]] = None,
        recent_assistant_messages: Optional[list[str]] = None,
    ) -> None:
        """Fire the Convergence Protocol pipeline to detect violations.

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
        if not self._cp_runner.enabled:
            return

        ssm = self._get_sub_sessions() if self._get_sub_sessions else None
        active_sessions = ssm.list_active() if ssm else []

        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None

        result = await self._cp_runner.run_phase(
            "post_inference",
            thread_id=thread_id,
            user_message=user_message,
            assistant_response=assistant_response,
            tool_calls_made=tool_calls_made,
            active_sessions=active_sessions,
            nl_tools=nl_tools,
            prior_assistant_message=prior_assistant_message,
            prior_tool_calls_made=prior_tool_calls_made,
            recent_assistant_messages=recent_assistant_messages,
        )
        if result is None:
            return

        if result.correction:
            new_depth = convergence_depth + 1
            # At depth >= 2, the model already failed to comply with a
            # correction once.  Repeating the same demanding prompt won't
            # help — the model is likely incapable of making the tool call
            # in this context.  Switch to a graceful fallback that tells
            # the model to stop trying and explain the limitation instead.
            if convergence_depth >= 1:
                correction_text = (
                    "[CONVERGENCE PROTOCOL — UNABLE TO COMPLY] "
                    "The previous correction could not be fulfilled. "
                    "Simply continue the conversation naturally. "
                    "Do NOT claim tools are blocked or unavailable. "
                    "Do NOT repeat the failed action. Just respond "
                    "helpfully to the user's last real message."
                )
            else:
                correction_text = result.correction
            logger.info(
                "Convergence Protocol injecting correction into thread %s (depth=%d, hooks=%s)",
                thread_id, new_depth,
                [m["hook"] for m in result.correction_metadata],
            )
            await self._dispatch(_QueueItem(
                text=correction_text,
                thread_id=thread_id,
                is_system_event=True,
                convergence_depth=new_depth,
                correction_for_seq=issued_for_seq,
            ))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    async def _prepare_inference_context(
        self, item: _QueueItem,
    ) -> tuple[list[dict], str, str | None, "BackendPool", "ProviderConfig", bool, str, list | None, list | None]:
        """Resolve config, build messages, fetch memories, assemble system prompt.

        Returns (messages, system_prompt, memory_query, pool, pool_cfg,
        is_sub_session_result, prompt_mode, memory_results, prediction_results).
        Also handles pre-compaction if the history exceeds the token budget.
        """
        thread_id = item.thread_id
        messages = await self._store.build_messages(item.text, item.is_system_event, thread_id, item.content)

        # Track last activity for session timeout checking (#58 plumbing).
        if not item.is_system_event:
            self._session_mgr.record_activity(thread_id)

        # Resolve per-thread config overrides.
        resolved_cfg = self._session_mgr.resolve_config(thread_id)
        pool = self._session_mgr.resolve_pool(thread_id)
        pool_cfg = pool.primary if pool.enabled else self._cfg
        prompt_mode = resolved_cfg.system_prompt_mode if resolved_cfg else "full"

        # Build a query for vector memory retrieval (user message + last assistant reply).
        _query_parts = [item.text] if item.text else []
        for m in reversed(messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                _query_parts.append(m["content"][:500])
                break
        _memory_query = " ".join(_query_parts) if _query_parts else None

        # Pre-fetch memories and predictions off the event loop to avoid blocking I/O.
        from wintermute.infra import memory_store

        async def _fetch_memories():
            if memory_store.is_memory_backend_initialized() and _memory_query:
                try:
                    return await asyncio.to_thread(memory_store.search, _memory_query)
                except Exception as e:
                    logger.warning("Memory search failed, continuing without memory context: %s", e)
                    return []  # empty list (not None) so assembler won't retry
            return None

        async def _fetch_predictions():
            try:
                return await asyncio.to_thread(prompt_assembler.fetch_predictions)
            except Exception:
                logger.debug("Prediction pre-fetch failed", exc_info=True)
                return None

        if prompt_mode == "minimal":
            _memory_results = await _fetch_memories()
            _prediction_results = None
        else:
            _memory_results, _prediction_results = await asyncio.gather(
                _fetch_memories(), _fetch_predictions(),
            )

        # Assemble system prompt first so we can measure its real token cost.
        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        summary = self._store.compaction_summaries.get(thread_id)
        system_prompt = prompt_assembler.assemble(
            extra_summary=summary, query=_memory_query,
            memory_results=_memory_results, prompt_mode=prompt_mode,
            tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
            nl_tools=nl_tools,
            prediction_results=_prediction_results,
        )
        active_schemas = tool_module.get_tool_schemas(
            nl_tools=nl_tools,
            tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
        )
        overhead_tokens = (
            _count_tokens(system_prompt, pool_cfg.model)
            + _count_tokens(json.dumps(active_schemas), pool_cfg.model)
        )

        history_tokens = sum(
            _count_tokens(m["content"] if isinstance(m["content"], str) else
                          " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict)),
                          pool_cfg.model)
            for m in messages
        )
        compaction_threshold = pool_cfg.context_size - pool_cfg.max_tokens - overhead_tokens
        if history_tokens > compaction_threshold:
            logger.info(
                "History at %d tokens (overhead %d, threshold %d) – compacting before inference (thread=%s)",
                history_tokens, overhead_tokens, compaction_threshold, thread_id,
            )
            await self._compactor.compact(thread_id)
            messages = await self._store.build_messages(item.text, item.is_system_event, thread_id, item.content)
            # Reassemble with the updated compaction summary.
            summary = self._store.compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(
                extra_summary=summary, query=_memory_query,
                memory_results=_memory_results, prompt_mode=prompt_mode,
                tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
                nl_tools=nl_tools,
                prediction_results=_prediction_results,
            )

        self._store.last_system_prompt[thread_id] = system_prompt

        is_sub_session_result = item.is_system_event and "[SUB-SESSION " in item.text
        return (messages, system_prompt, _memory_query, pool, pool_cfg,
                is_sub_session_result, prompt_mode, _memory_results,
                _prediction_results)

    async def _save_user_message(
        self, item: _QueueItem, pool_cfg: "ProviderConfig", is_sub_session_result: bool,
    ) -> None:
        """Persist the incoming user/system message to DB and emit events."""
        await self._store.save_user_message(
            text=item.text, thread_id=item.thread_id,
            is_system_event=item.is_system_event,
            is_sub_session_result=is_sub_session_result,
            convergence_depth=item.convergence_depth,
            content=item.content, model=pool_cfg.model,
        )

    async def _save_inference_result(
        self, item: _QueueItem, reply: LLMReply,
        pool: "BackendPool", pool_cfg: "ProviderConfig",
        inference_duration: float, memory_query: str | None,
    ) -> None:
        """Log inference results, save assistant message, emit events, run post-inference tasks."""
        thread_id = item.thread_id

        # Determine action type for interaction log.
        if item.convergence_depth > 0:
            _action = "convergence_response"
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
                pool.last_used,
                item.text, reply.text, "ok",
                raw_output=json.dumps(raw_output_data),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to save interaction log entry", exc_info=True)

        # Emit inference.completed event for self-model metrics.
        if self._event_bus:
            self._event_bus.emit(
                "inference.completed",
                thread_id=thread_id,
                duration_s=round(inference_duration, 2),
                tool_calls=len(reply.tool_call_details) if reply.tool_call_details else 0,
                model=pool.last_used,
            )
        try:
            await database.async_call(
                database.save_interaction_log,
                _time.time(), "inference_completed", thread_id,
                pool.last_used,
                f"duration={inference_duration:.2f}s",
                f"tool_calls={len(reply.tool_call_details) if reply.tool_call_details else 0}",
                "ok",
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to log inference_completed", exc_info=True)

        await self._store.save_assistant_message(
            reply.text, thread_id, pool_cfg.model,
        )

        await self._compactor.maybe_summarise_components(
            thread_id, _from_system_event=item.is_system_event,
        )

        # Attach timing/backend metadata so callers can use it.
        reply.duration_seconds = round(inference_duration, 2)
        reply.backend_used = pool.last_used

    async def _run_inference_with_retry(
        self,
        item: _QueueItem,
        system_prompt: str,
        messages: list[dict],
        pool: "BackendPool",
        memory_query: "str | None",
        memory_results: "list | None",
        prompt_mode: str,
        prediction_results: "list | None" = None,
    ) -> "LLMReply":
        """Run the inference loop, retrying once after compaction on context overflow."""
        thread_id = item.thread_id
        try:
            return await self._inference_loop(
                system_prompt, messages, thread_id,
                pool=pool,
            )
        except ContextTooLargeError:
            logger.warning("Context too large for thread %s — forcing compaction", thread_id)
            await self._compactor.compact(thread_id)
            messages = await self._store.build_messages(
                item.text, item.is_system_event, thread_id, item.content,
            )
            nl_enabled = self._nl_translation_config.get("enabled", False)
            nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
            summary = self._store.compaction_summaries.get(thread_id)
            system_prompt = prompt_assembler.assemble(
                extra_summary=summary, query=memory_query,
                memory_results=memory_results, prompt_mode=prompt_mode,
                tool_profiles=self._tool_deps.tool_profiles if self._tool_deps else None,
                nl_tools=nl_tools,
                prediction_results=prediction_results,
            )
            return await self._inference_loop(
                system_prompt, messages, thread_id,
                pool=pool,
            )

    async def _process(self, item: _QueueItem) -> LLMReply:
        thread_id = item.thread_id

        # 1. Prepare context: config, messages, memory, system prompt, pre-compaction.
        (messages, system_prompt, _memory_query, pool, pool_cfg,
         is_sub_session_result, _prompt_mode,
         _memory_results, _prediction_results) = await self._prepare_inference_context(item)

        # 2. Save the incoming message to DB.
        await self._save_user_message(item, pool_cfg, is_sub_session_result)

        # 3. Run inference (with compaction retry on context overflow).
        _inference_start = _time.time()
        reply = await self._run_inference_with_retry(
            item, system_prompt, messages, pool,
            _memory_query, _memory_results, _prompt_mode,
            prediction_results=_prediction_results,
        )
        _inference_duration = _time.time() - _inference_start

        # 4. Save results, log, emit events, run post-inference tasks.
        await self._save_inference_result(
            item, reply, pool, pool_cfg, _inference_duration, _memory_query,
        )

        return reply

    # ------------------------------------------------------------------
    # OpenAI inference loop (handles tool-use rounds)
    # ------------------------------------------------------------------

    async def _inference_loop(self, system_prompt: str, messages: list[dict],
                              thread_id: str = "default",
                              disable_tools: bool = False,
                              pool: "Optional[BackendPool]" = None) -> LLMReply:
        """
        Repeatedly call the API until finish_reason is not 'tool_calls'.
        The system prompt is prepended as a role=system message each call
        (not stored in the DB so it stays fresh on every inference).

        *pool* overrides the default main_pool when a per-thread backend
        override is active.

        Convergence Protocol hooks are fired at three phases:
          - pre_execution:  before each tool call (can block execution)
          - post_execution: after each tool call (can flag results)
          - post_inference:  handled by the caller (_run_convergence_check)
        """
        active_pool = pool or self._main_pool
        active_cfg = active_pool.primary if active_pool.enabled else self._cfg
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        _profiles = self._tool_deps.tool_profiles if self._tool_deps else None
        if disable_tools:
            tools = None
        elif nl_tools or _profiles:
            tools = tool_module.get_tool_schemas(nl_tools=nl_tools, tool_profiles=_profiles)
        else:
            tools = tool_module.TOOL_SCHEMAS
        token_budget = active_cfg.context_size - active_cfg.max_tokens
        reasoning_parts: list[str] = []
        tool_calls_made: list[str] = []
        tool_call_details: list[dict] = []

        cp_enabled = self._convergence_protocol_pool.enabled

        # Build a shared context for tool-call processing.
        async def _cp_check_main(phase, *, tool_name=None, tool_args=None,
                                 tool_result=None, assistant_response="",
                                 tool_calls_made=None, nl_tools=None):
            return await self._run_phase_check(
                phase=phase, thread_id=thread_id,
                tool_calls_made=tool_calls_made or [],
                assistant_response=assistant_response,
                tool_name=tool_name, tool_args=tool_args,
                tool_result=tool_result, nl_tools=nl_tools,
            )

        tc_ctx = make_tool_context(
            thread_id=thread_id,
            scope="main",
            pool_last_used=active_pool.last_used,
            event_bus=self._event_bus,
            nl_enabled=nl_enabled,
            nl_tools=nl_tools,
            nl_translation_pool=getattr(self, "_nl_translation_pool", None),
            timezone_str=prompt_assembler.get_timezone(),
            cp_enabled=cp_enabled,
            cp_check=_cp_check_main if cp_enabled else None,
            max_tool_output_chars=active_cfg.context_size * 4,  # tokens → approx chars
            tool_deps=self._tool_deps,
        )

        empty_retries = 0
        while True:
            # Trim oldest tool results if accumulated context exceeds budget.
            self._compactor.trim_tool_results(full_messages, token_budget, active_cfg.model)

            try:
                response = await asyncio.wait_for(
                    active_pool.call(
                        messages=full_messages,
                        tools=tools,
                    ),
                    timeout=300.0,  # 5 min hard ceiling per LLM call
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "LLM API call timed out after 300 seconds — backend may be "
                    "unresponsive. Aborting inference loop."
                )

            if response.content is None and not response.tool_calls:
                empty_retries += 1
                if empty_retries >= MAX_EMPTY_RETRIES:
                    raise RuntimeError(
                        f"LLM returned empty response {empty_retries} times in a row; aborting"
                    )
                logger.warning("LLM returned empty response, retrying (%d/%d)", empty_retries, MAX_EMPTY_RETRIES)
                logger.debug("Empty response: %s", response)
                full_messages.append({"role": "assistant", "content": ""})
                full_messages.append({"role": "user", "content": "Continue."})
                continue
            empty_retries = 0

            # Collect reasoning tokens from every round (including tool-use rounds).
            if active_cfg.reasoning and response.reasoning_content:
                reasoning_parts.append(response.reasoning_content)

            # Convert LLMResponse to a plain dict for conversation history.
            msg = response.to_message_dict()
            msg_tool_calls = msg.get("tool_calls")
            msg_content = extract_content_text(msg)

            if msg_tool_calls:
                _tc_names = [tc["function"]["name"] for tc in msg_tool_calls]
                logger.debug(
                    "Tool calls detected (finish_reason=%s): %s",
                    response.finish_reason, _tc_names,
                )

                # Detect truncated tool calls (LLM hit max_tokens mid-JSON).
                # The last tool call's arguments are likely incomplete.
                if response.finish_reason == "length":
                    logger.warning(
                        "Tool calls truncated (finish_reason=length) — "
                        "discarding and asking model to retry with smaller output"
                    )
                    full_messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was cut off (output token limit reached) "
                            "while generating tool call arguments. Please retry — if the "
                            "content is very large, split it into smaller parts using "
                            "multiple sequential tool calls."
                        ),
                    })
                    continue

                # Log this inference round (intermediate, tool-use round).
                try:
                    await database.async_call(
                        database.save_interaction_log,
                        _time.time(), "inference_round", thread_id,
                        active_pool.last_used,
                        msg_content[:500] or f"[requesting {len(_tc_names)} tool call(s)]",
                        f"[tool_calls: {', '.join(_tc_names)}]",
                        "ok",
                        raw_output=json.dumps({
                            "tool_calls": [
                                {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                                for tc in msg_tool_calls
                            ],
                            "content": msg_content,
                        }),
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to log inference round", exc_info=True)
                # Append the assistant's tool-call message (already a dict).
                full_messages.append(msg)

                # Execute each tool via the shared pipeline.
                tc_ctx.pool_last_used = active_pool.last_used
                for tc in msg_tool_calls:
                    outcome = await process_tool_call(
                        tc, tc_ctx, tool_calls_made,
                        assistant_response=msg_content,
                    )
                    tool_call_details.extend(outcome.call_details)
                    full_messages.append({
                        "role":         "tool",
                        "tool_call_id": tc["id"],
                        "content":      outcome.content,
                    })
                continue  # next round

            # -- Rescue XML/text-encoded tool calls -------------------
            if msg_content and tools:
                _known_names = {
                    s["function"]["name"] for s in tools
                }
                _rescued = rescue_tool_calls(msg_content, _known_names)
                if _rescued:
                    # Log this as an inference round with rescued tool calls so
                    # the interaction log reflects actual tool activity.
                    try:
                        _rescued_names = [tc.function.name for tc in _rescued]
                        await database.async_call(
                            database.save_interaction_log,
                            _time.time(), "inference_round", thread_id,
                            active_pool.last_used,
                            msg_content[:500],
                            f"[rescued_tool_calls: {', '.join(_rescued_names)}]",
                            "ok",
                            raw_output=json.dumps({
                                "tool_calls": [
                                    {"name": tc.function.name, "arguments": tc.function.arguments}
                                    for tc in _rescued
                                ],
                                "content": msg_content,
                                "rescue": True,
                            }),
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to log rescued tool calls", exc_info=True)
                    # Synthesise an assistant message with the rescued calls
                    # and inject tool-result messages, then loop again.
                    full_messages.append({
                        "role": "assistant",
                        "content": msg_content,
                        "tool_calls": [
                            {"id": tc.id, "type": tc.type,
                             "function": {"name": tc.function.name,
                                          "arguments": tc.function.arguments}}
                            for tc in _rescued
                        ],
                    })
                    tc_ctx.pool_last_used = active_pool.last_used
                    for tc in _rescued:
                        outcome = await process_tool_call(
                            tc, tc_ctx, tool_calls_made,
                            assistant_response=msg_content,
                        )
                        tool_call_details.extend(outcome.call_details)
                        full_messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      outcome.content,
                        })
                    continue  # next round

            # Terminal response.
            content = msg_content
            reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
            return LLMReply(text=content, reasoning=reasoning,
                            tool_calls_made=tool_calls_made,
                            tool_call_details=tool_call_details)

    async def _run_phase_check(
        self,
        phase: str,
        thread_id: str,
        tool_calls_made: list[str],
        assistant_response: str = "",
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
        nl_tools: "set[str] | None" = None,
    ) -> Optional["convergence_protocol_module.ConvergenceResult"]:
        """Run Convergence Protocol hooks for a specific phase.

        Delegates to the bound ``ConvergenceProtocolRunner`` (scope is fixed
        at construction time to ``"main"``).
        """
        return await self._cp_runner.run_phase(
            phase,
            thread_id=thread_id,
            tool_calls_made=tool_calls_made,
            assistant_response=assistant_response,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            nl_tools=nl_tools,
        )

