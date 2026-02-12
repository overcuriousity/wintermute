"""
Agentic Sub-sessions

An isolated, ephemeral worker that runs a multi-step tool loop in the
background without touching any user-facing thread's history.

Lifecycle
---------
1. Orchestrator calls spawn_sub_session tool (from within its normal inference
   loop).  SubSessionManager.spawn() creates an asyncio.Task and immediately
   returns the session_id to the orchestrator.
2. Worker runs _worker_loop(): its own inference + tool-call loop with a
   focused system prompt and an in-memory message list (never persisted).
3. On completion the worker calls back via enqueue_system_event so the result
   enters the parent thread as a system event.  The orchestrator then
   formulates the final user-facing reply.
4. If parent_thread_id is None (fire-and-forget mode, used by global heartbeat
   and system reminders) the result is only logged.

System prompt modes
-------------------
  "minimal"   – lightweight execution agent (default)
  "full"      – full assembled prompt (BASE + MEMORIES + HEARTBEATS + SKILLS)
  "base_only" – BASE_PROMPT.txt only
  "none"      – no system prompt (bare tool-use loop, e.g. pure script runner)

Tool filtering by mode
----------------------
  "minimal", "base_only", "none" → execution + research tools only
  "full"                          → all tools including orchestration

Nesting
-------
  "full"-mode workers may spawn sub-sessions up to MAX_NESTING_DEPTH (2).
  Other modes have no spawn_sub_session in their tool set at all.

Continuation on timeout
-----------------------
When a worker times out, its full message history is stored on the state
object (state.messages is updated after every tool call, not just at the
end).  The timeout handler auto-spawns a continuation sub-session that
receives the prior messages and appends a resumption note, so the new worker
picks up exactly where the old one left off.  Up to MAX_CONTINUATION_DEPTH
hops are allowed before the chain gives up and reports partial progress.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Coroutine, Optional

from openai import AsyncOpenAI

from ganglion import prompt_assembler
from ganglion import tools as tool_module

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300       # seconds per hop
MAX_CONTINUATION_DEPTH = 3  # max auto-continuation hops before giving up

# Tool categories available per system_prompt_mode.
_MODE_TOOL_CATEGORIES: dict[str, set[str]] = {
    "minimal":   {"execution", "research"},
    "base_only": {"execution", "research"},
    "none":      {"execution", "research"},
    "full":      {"execution", "research", "orchestration"},
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SubSessionState:
    session_id: str
    objective: str
    parent_thread_id: Optional[str]      # None = fire-and-forget
    system_prompt_mode: str              # "full" | "base_only" | "minimal" | "none"
    status: str                          # "running" | "completed" | "failed" | "timeout"
    created_at: str                      # ISO-8601
    nesting_depth: int = 1               # 1 = direct child, 2 = grandchild
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    tool_calls_log: list = field(default_factory=list)  # [(tool_name, summary), ...]
    # Full in-flight message history — updated after every tool call so it
    # survives asyncio.wait_for cancellation and can be handed to a continuation.
    messages: list = field(default_factory=list)
    continuation_depth: int = 0          # how many hops deep this session is
    continued_from: Optional[str] = None # session_id of predecessor, if any


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SubSessionManager:
    """
    Manages background worker sub-sessions.

    Injected dependencies
    ---------------------
    client             – shared AsyncOpenAI instance (no extra connections)
    llm_config         – LLMConfig from the main LLMThread (model, max_tokens, …)
    enqueue_system_event – async callable(text: str, thread_id: str) that injects
                           a result back into a parent thread's queue
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        llm_config,                          # ganglion.llm_thread.LLMConfig
        enqueue_system_event: Callable[..., Coroutine],
    ) -> None:
        self._client = client
        self._cfg = llm_config
        self._enqueue = enqueue_system_event
        self._states: dict[str, SubSessionState] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn(
        self,
        objective: str,
        context_blobs: Optional[list[str]] = None,
        parent_thread_id: Optional[str] = None,
        system_prompt_mode: str = "minimal",
        timeout: int = DEFAULT_TIMEOUT,
        nesting_depth: int = 1,
        prior_messages: Optional[list[dict]] = None,
        continuation_depth: int = 0,
        continued_from: Optional[str] = None,
    ) -> str:
        """
        Start a worker sub-session and return its session_id immediately.

        The caller (orchestrator) is not blocked.  Results are delivered via
        enqueue_system_event when the worker finishes.

        Pass prior_messages to resume from a previous session's message history
        (used internally by the auto-continuation logic on timeout).
        """
        session_id = f"sub_{uuid.uuid4().hex[:8]}"
        state = SubSessionState(
            session_id=session_id,
            objective=objective,
            parent_thread_id=parent_thread_id,
            system_prompt_mode=system_prompt_mode,
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
            nesting_depth=nesting_depth,
            messages=list(prior_messages) if prior_messages else [],
            continuation_depth=continuation_depth,
            continued_from=continued_from,
        )
        self._states[session_id] = state

        task = asyncio.create_task(
            self._run(state, context_blobs or [], timeout),
            name=f"sub_session_{session_id}",
        )
        self._tasks[session_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(session_id, None))

        logger.info(
            "Sub-session %s spawned (parent=%s mode=%s timeout=%ds depth=%d nest=%d)",
            session_id, parent_thread_id, system_prompt_mode, timeout,
            continuation_depth, nesting_depth,
        )
        return session_id

    def cancel_for_thread(self, thread_id: str) -> int:
        """
        Cancel all running sub-sessions whose parent is thread_id.
        Returns the number of sessions cancelled.
        """
        cancelled = 0
        for sid, state in list(self._states.items()):
            if state.parent_thread_id == thread_id and state.status == "running":
                task = self._tasks.get(sid)
                if task and not task.done():
                    task.cancel()
                    logger.info("Cancelled sub-session %s (thread reset)", sid)
                    cancelled += 1
        return cancelled

    def list_active(self) -> list[dict]:
        """Return serialisable state dicts for all non-completed sub-sessions."""
        return [
            self._serialise(state)
            for state in self._states.values()
            if state.status == "running"
        ]

    def list_all(self) -> list[dict]:
        """Return serialisable state dicts for all known sub-sessions, newest first."""
        return sorted(
            [self._serialise(state) for state in self._states.values()],
            key=lambda s: s["created_at"],
            reverse=True,
        )

    @staticmethod
    def _serialise(state: SubSessionState) -> dict:
        """Return state as a dict, omitting the (potentially large) messages list."""
        return {k: v for k, v in state.__dict__.items() if k != "messages"}

    # ------------------------------------------------------------------
    # Worker execution
    # ------------------------------------------------------------------

    async def _run(
        self,
        state: SubSessionState,
        context_blobs: list[str],
        timeout: int,
    ) -> None:
        try:
            result = await asyncio.wait_for(
                self._worker_loop(state, context_blobs),
                timeout=timeout,
            )
            state.status = "completed"
            state.result = result
            state.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info("Sub-session %s completed (%d chars)", state.session_id, len(result or ""))
            await self._report(state, f"[SUB-SESSION {state.session_id} RESULT]\n\n{result}")

        except asyncio.TimeoutError:
            state.status = "timeout"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "Sub-session %s timed out after %ds (depth=%d, tool_calls=%d)",
                state.session_id, timeout, state.continuation_depth,
                len(state.tool_calls_log),
            )

            if state.continuation_depth < MAX_CONTINUATION_DEPTH and state.messages:
                # Auto-continue: spawn a new worker with the full message history.
                # It will append a resumption note and pick up where we left off.
                cont_id = self.spawn(
                    objective=state.objective,
                    context_blobs=[],
                    parent_thread_id=state.parent_thread_id,
                    system_prompt_mode=state.system_prompt_mode,
                    timeout=timeout,
                    nesting_depth=state.nesting_depth,
                    prior_messages=state.messages,
                    continuation_depth=state.continuation_depth + 1,
                    continued_from=state.session_id,
                )
                msg = (
                    f"[SUB-SESSION {state.session_id} TIMEOUT → CONTINUING as {cont_id}]\n"
                    f"Exceeded {timeout}s after {len(state.tool_calls_log)} tool call(s). "
                    f"Automatically resuming from where it left off in {cont_id} "
                    f"(hop {state.continuation_depth + 1}/{MAX_CONTINUATION_DEPTH})."
                )
            else:
                # Terminal timeout — report what was accomplished across the chain.
                if state.continuation_depth > 0:
                    chain_note = (
                        f"Task chain exhausted all {MAX_CONTINUATION_DEPTH} continuation(s) "
                        f"and still did not complete. "
                    )
                else:
                    chain_note = ""

                if state.tool_calls_log:
                    steps = "\n".join(
                        f"  {i+1}. {name}: {summary}"
                        for i, (name, summary) in enumerate(state.tool_calls_log)
                    )
                    msg = (
                        f"[SUB-SESSION {state.session_id} TIMEOUT — GIVING UP]\n"
                        f"{chain_note}"
                        f"Last session completed {len(state.tool_calls_log)} tool call(s) "
                        f"in {timeout}s:\n{steps}\n\n"
                        f"The task may need to be broken into smaller steps."
                    )
                else:
                    msg = (
                        f"[SUB-SESSION {state.session_id} TIMEOUT — GIVING UP]\n"
                        f"{chain_note}"
                        f"Timed out after {timeout}s with no tool calls completed. "
                        f"The worker may have been stuck waiting for the first LLM response. "
                        f"Consider retrying or checking model availability."
                    )
            await self._report(state, msg)

        except asyncio.CancelledError:
            state.status = "failed"
            state.error = "Cancelled"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info("Sub-session %s cancelled", state.session_id)
            # Don't report back on explicit cancel (e.g. /new command) — the
            # user already knows they reset the session.

        except Exception as exc:  # noqa: BLE001
            state.status = "failed"
            state.error = str(exc)
            state.completed_at = datetime.now(timezone.utc).isoformat()
            msg = f"[SUB-SESSION {state.session_id} FAILED] {exc}"
            logger.exception("Sub-session %s failed", state.session_id)
            await self._report(state, msg)

    async def _report(self, state: SubSessionState, text: str) -> None:
        """Deliver result to parent thread, or log if fire-and-forget."""
        if state.parent_thread_id:
            try:
                await self._enqueue(text, state.parent_thread_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sub-session %s failed to report back: %s", state.session_id, exc)
        else:
            logger.info("Sub-session %s (fire-and-forget): %s", state.session_id, text[:300])

    async def _worker_loop(self, state: SubSessionState, context_blobs: list[str]) -> str:
        """
        Run a full tool-use inference loop in isolation.

        The worker has its own in-memory message list — nothing is read from
        or written to the conversation DB.  Tool schemas are filtered by the
        session's system_prompt_mode (e.g. "minimal" workers only get
        execution + research tools).

        state.messages is used directly (not a local copy) so that the timeout
        handler always has access to the latest message history for continuation.
        """
        # Build the tool set for this worker based on its mode.
        categories = _MODE_TOOL_CATEGORIES.get(
            state.system_prompt_mode, {"execution", "research"}
        )
        tool_schemas = tool_module.get_tool_schemas(categories)

        if state.messages:
            # Resuming from a prior timed-out session.  The full prior history
            # is already in state.messages; just append a resumption note.
            state.messages.append({
                "role":    "user",
                "content": (
                    f"You were interrupted by a timeout "
                    f"(this is continuation {state.continuation_depth}). "
                    "Continue the task from exactly where you left off."
                ),
            })
        else:
            # Fresh start — build the initial conversation.
            system_prompt = self._build_system_prompt(
                state.system_prompt_mode, state.objective, context_blobs
            )
            state.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": state.objective},
            ]

        while True:
            response = await self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                tools=tool_schemas,
                tool_choice="auto",
                messages=state.messages,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                state.messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    try:
                        inputs = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        inputs = {}

                    result = tool_module.execute_tool(
                        tc.function.name,
                        inputs,
                        thread_id=state.session_id,
                        nesting_depth=state.nesting_depth,
                    )
                    # Track progress on state so the timeout handler can report
                    # and continue from it.
                    result_preview = result[:120].replace("\n", " ")
                    state.tool_calls_log.append((tc.function.name, result_preview))
                    logger.debug("Sub-session %s tool %s -> %s",
                                 state.session_id, tc.function.name, result[:200])
                    state.messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue

            return (choice.message.content or "").strip()

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        mode: str,
        objective: str,
        context_blobs: list[str],
    ) -> str:
        if mode == "none":
            base = ""
        elif mode == "minimal":
            base = (
                "You are a background worker executing a specific task. "
                "Use the available tools to complete the objective. "
                "Be concise and report only what was done and any important results."
            )
        elif mode == "full":
            base = prompt_assembler.assemble()
        else:  # "base_only"
            raw = prompt_assembler._read(prompt_assembler.BASE_PROMPT_FILE)
            base = f"# Core Instructions\n\n{raw}" if raw else ""

        parts = [base] if base else []

        if context_blobs:
            blobs_text = "\n\n".join(context_blobs)
            parts.append(f"# Task Context\n\n{blobs_text}")

        parts.append(
            f"# Current Objective\n\n{objective}\n\n"
            "Work autonomously using the available tools. "
            "When you have completed the objective, provide a concise summary of "
            "what was done and any important results or file paths."
        )

        return "\n\n---\n\n".join(parts)
