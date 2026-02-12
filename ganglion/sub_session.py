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
  "full"      – full assembled prompt (BASE + MEMORIES + HEARTBEATS + SKILLS)
  "base_only" – BASE_PROMPT.txt only  (default for most worker tasks)
  "none"      – no system prompt (bare tool-use loop, e.g. pure script runner)
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

DEFAULT_TIMEOUT = 300  # seconds


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SubSessionState:
    session_id: str
    objective: str
    parent_thread_id: Optional[str]      # None = fire-and-forget
    system_prompt_mode: str              # "full" | "base_only" | "none"
    status: str                          # "running" | "completed" | "failed" | "timeout"
    created_at: str                      # ISO-8601
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


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
        system_prompt_mode: str = "base_only",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """
        Start a worker sub-session and return its session_id immediately.

        The caller (orchestrator) is not blocked.  Results are delivered via
        enqueue_system_event when the worker finishes.
        """
        session_id = f"sub_{uuid.uuid4().hex[:8]}"
        state = SubSessionState(
            session_id=session_id,
            objective=objective,
            parent_thread_id=parent_thread_id,
            system_prompt_mode=system_prompt_mode,
            status="running",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._states[session_id] = state

        task = asyncio.create_task(
            self._run(state, context_blobs or [], timeout),
            name=f"sub_session_{session_id}",
        )
        self._tasks[session_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(session_id, None))

        logger.info(
            "Sub-session %s spawned (parent=%s mode=%s timeout=%ds)",
            session_id, parent_thread_id, system_prompt_mode, timeout,
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
            {k: v for k, v in state.__dict__.items()}
            for state in self._states.values()
            if state.status == "running"
        ]

    def list_all(self) -> list[dict]:
        """Return serialisable state dicts for all known sub-sessions, newest first."""
        return sorted(
            [{k: v for k, v in state.__dict__.items()} for state in self._states.values()],
            key=lambda s: s["created_at"],
            reverse=True,
        )

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
            msg = f"[SUB-SESSION {state.session_id} TIMEOUT] Task exceeded {timeout}s and was cancelled."
            logger.warning("Sub-session %s timed out after %ds", state.session_id, timeout)
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
        or written to the conversation DB.  It has access to all tools except
        spawn_sub_session (depth limit of 1).
        """
        system_prompt = self._build_system_prompt(
            state.system_prompt_mode, state.objective, context_blobs
        )

        # Seed the conversation with the objective as the first user message.
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": state.objective},
        ]

        while True:
            response = await self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                tools=tool_module.TOOL_SCHEMAS,
                tool_choice="auto",
                messages=messages,
            )

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    try:
                        inputs = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        inputs = {}

                    result = tool_module.execute_tool(
                        tc.function.name,
                        inputs,
                        thread_id=state.session_id,
                        in_sub_session=True,          # blocks recursive spawning
                    )
                    logger.debug("Sub-session %s tool %s -> %s",
                                 state.session_id, tc.function.name, result[:200])
                    messages.append({
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
        elif mode == "full":
            base = prompt_assembler.assemble()
        else:  # "base_only" (default)
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
