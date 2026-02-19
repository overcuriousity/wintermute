"""
Agentic Sub-sessions with Workflow DAG

Isolated, ephemeral workers that run multi-step tool loops in the background
without touching any user-facing thread's history.  Every sub-session is
tracked as a node in a lightweight workflow DAG so that multi-step tasks
(research A + B → upload both) execute deterministically without relying on
the LLM to remember follow-up steps.

Lifecycle
---------
1. Orchestrator calls spawn_sub_session (optionally with depends_on).
   SubSessionManager.spawn() registers a TaskNode in a Workflow DAG and
   either starts the worker immediately (no deps / all deps done) or
   defers it as "pending".
2. Worker runs _worker_loop(): its own inference + tool-call loop with a
   focused system prompt and an in-memory message list (never persisted).
3. On completion, _resolve_dependents() checks if any pending nodes can
   now start.  Dependency results are passed as context_blobs.
4. The result enters the parent thread via enqueue_system_event.
   If parent_thread_id is None (fire-and-forget) the result is only logged.

Workflow DAG
------------
  - Every sub-session is a node in a Workflow (auto-created if needed).
  - depends_on: list of session_ids that must complete first.
  - Fan-in: task C depends_on=[A, B] — auto-starts when both finish.
  - Failure propagation: if a dependency fails, all transitive dependents
    are marked failed and reported.
  - Resolution is event-driven (no polling): each completion triggers a check.

System prompt modes
-------------------
  "minimal"   – lightweight execution agent (default)
  "full"      – full assembled prompt (BASE + MEMORIES + AGENDA + SKILLS)
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
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Callable, Coroutine, Optional

from wintermute import database
from wintermute import prompt_assembler
from wintermute import prompt_loader
from wintermute import turing_protocol as turing_protocol_module
from wintermute import nl_translator
from wintermute import tools as tool_module
from wintermute.llm_thread import BackendPool

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
    root_thread_id: Optional[str] = None # original user-facing thread (for nested routing)
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
    tool_names: Optional[list[str]] = None  # explicit tool whitelist (bypasses category filter)


@dataclass
class TaskNode:
    """A node in a workflow DAG."""
    node_id: str                          # == session_id
    objective: str
    context_blobs: list[str]
    system_prompt_mode: str
    timeout: int
    depends_on: list[str]                 # node_ids that must complete first
    nesting_depth: int = 1
    parent_thread_id: Optional[str] = None
    root_thread_id: Optional[str] = None  # original user-facing thread
    status: str = "pending"               # pending | running | completed | failed
    result: Optional[str] = None
    error: Optional[str] = None
    not_before: Optional[datetime] = None # time gate: don't start before this
    tool_names: Optional[list[str]] = None  # explicit tool whitelist (bypasses category filter)


@dataclass
class Workflow:
    """A DAG of TaskNodes."""
    workflow_id: str
    parent_thread_id: Optional[str]
    nodes: dict[str, TaskNode] = field(default_factory=dict)
    created_at: str = ""
    status: str = "running"               # running | completed | failed


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SubSessionManager:
    """
    Manages background worker sub-sessions and workflow DAGs.

    Every sub-session is tracked as a node in a Workflow.  When a node
    specifies depends_on, it stays pending until all dependencies finish,
    then auto-starts with their results as context.

    Injected dependencies
    ---------------------
    pool                       – BackendPool for the sub_sessions role (handles failover)
    enqueue_system_event       – async callable(text: str, thread_id: str) that injects
                                 a result back into a parent thread's queue
    turing_protocol_pool       – BackendPool for the Turing Protocol's own LLM calls
    turing_protocol_validators – per-hook enable/disable overrides from config
    """

    def __init__(
        self,
        pool: BackendPool,
        enqueue_system_event: Callable[..., Coroutine],
        turing_protocol_pool: Optional[BackendPool] = None,
        turing_protocol_validators: Optional[dict] = None,
        nl_translation_pool: Optional[BackendPool] = None,
        nl_translation_config: Optional[dict] = None,
    ) -> None:
        self._pool = pool
        self._cfg = pool.primary
        self._enqueue = enqueue_system_event
        self._tp_pool = turing_protocol_pool
        self._tp_validators = turing_protocol_validators
        self._nl_translation_pool = nl_translation_pool
        self._nl_translation_config = nl_translation_config or {}
        self._states: dict[str, SubSessionState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # DAG workflow tracking
        self._workflows: dict[str, Workflow] = {}
        self._session_to_workflow: dict[str, str] = {}  # session_id -> workflow_id
        # Tracks which sessions each worker has spawned (worker_session_id -> [child_ids]).
        # Used to resolve depends_on_previous without the LLM needing to track IDs.
        self._worker_spawned: dict[str, list[str]] = {}
        # Tracks parent sub-session IDs whose children have already been aggregated,
        # preventing duplicate delivery when multiple children complete near-simultaneously.
        self._aggregated_parents: set[str] = set()

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
        depends_on: Optional[list[str]] = None,
        depends_on_previous: bool = False,
        not_before: Optional[str] = None,
        tool_names: Optional[list[str]] = None,
    ) -> str:
        """
        Register a sub-session and start it immediately (or defer if deps pending).

        Every sub-session is tracked as a node in a workflow DAG.  When
        *depends_on* lists session_ids that haven't completed yet, the node
        stays pending and is auto-started once all dependencies finish.
        Results from completed dependencies are prepended to *context_blobs*.

        If *depends_on_previous* is True, the dependency list is automatically
        populated with all session IDs previously spawned by the same parent
        worker.  This eliminates the need for the LLM to track and reference
        session IDs, preventing hallucinated-ID deadlocks.

        Pass prior_messages to resume from a previous session's message history
        (used internally by the auto-continuation logic on timeout).
        """
        session_id = f"sub_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Resolve depends_on_previous: automatically depend on all sessions
        # the calling worker has spawned so far.
        if depends_on_previous and parent_thread_id:
            previous = list(self._worker_spawned.get(parent_thread_id, []))
            raw_deps = previous + (depends_on or [])
            if previous:
                logger.info(
                    "Sub-session %s: depends_on_previous resolved to %s",
                    session_id, previous,
                )
        else:
            raw_deps = depends_on or []

        # Track this session as a child of its parent worker.
        if parent_thread_id:
            self._worker_spawned.setdefault(parent_thread_id, []).append(session_id)

        # Validate dependency IDs — strip any that don't correspond to a known
        # session.  This prevents permanent deadlocks caused by hallucinated or
        # mistyped session IDs in depends_on.
        deps = []
        for dep_id in raw_deps:
            if dep_id in self._states:
                deps.append(dep_id)
            else:
                logger.warning(
                    "Sub-session %s: dropping unknown dependency '%s' "
                    "(session does not exist — possibly hallucinated by LLM)",
                    session_id, dep_id,
                )

        # Derive root_thread_id: the original user-facing thread.
        # When a sub-session spawns children, parent_thread_id is "sub_xxxx".
        # We resolve through to the original chat thread so aggregated results
        # can be delivered there deterministically.
        if parent_thread_id and parent_thread_id.startswith("sub_"):
            parent_state = self._states.get(parent_thread_id)
            root_thread_id = (parent_state.root_thread_id or parent_state.parent_thread_id) if parent_state else None
        else:
            root_thread_id = parent_thread_id

        # -- Parse not_before time gate --
        not_before_dt: Optional[datetime] = None
        if not_before:
            not_before_dt = self._parse_not_before(not_before)
            if not_before_dt:
                logger.info("Sub-session %s has time gate: not_before=%s", session_id, not_before_dt.isoformat())

        # -- Register the node in a workflow --
        node = TaskNode(
            node_id=session_id,
            objective=objective,
            context_blobs=list(context_blobs or []),
            system_prompt_mode=system_prompt_mode,
            timeout=timeout,
            depends_on=list(deps),
            nesting_depth=nesting_depth,
            parent_thread_id=parent_thread_id,
            root_thread_id=root_thread_id,
            not_before=not_before_dt,
            tool_names=tool_names,
        )

        if deps:
            # Collect all distinct workflows that the dependencies belong to.
            dep_wf_ids: set[str] = set()
            for dep_id in deps:
                wf_id = self._session_to_workflow.get(dep_id)
                if wf_id:
                    dep_wf_ids.add(wf_id)

            if not dep_wf_ids:
                # No deps are in a workflow yet — create a fresh one and
                # retroactively register the dependency sessions.
                workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
                wf = Workflow(
                    workflow_id=workflow_id,
                    parent_thread_id=parent_thread_id,
                    created_at=now,
                )
                self._workflows[workflow_id] = wf
                for dep_id in deps:
                    dep_state = self._states.get(dep_id)
                    if dep_state:
                        dep_node = TaskNode(
                            node_id=dep_id,
                            objective=dep_state.objective,
                            context_blobs=[],
                            system_prompt_mode=dep_state.system_prompt_mode,
                            timeout=DEFAULT_TIMEOUT,
                            depends_on=[],
                            nesting_depth=dep_state.nesting_depth,
                            parent_thread_id=dep_state.parent_thread_id,
                            status=dep_state.status,
                            result=dep_state.result,
                            error=dep_state.error,
                        )
                        wf.nodes[dep_id] = dep_node
                        self._session_to_workflow[dep_id] = workflow_id
            elif len(dep_wf_ids) == 1:
                # All deps share one workflow — just use it.
                workflow_id = next(iter(dep_wf_ids))
            else:
                # Dependencies span multiple workflows — merge them all
                # into the first one.
                wf_list = sorted(dep_wf_ids)
                workflow_id = wf_list[0]
                target_wf = self._workflows[workflow_id]
                for other_id in wf_list[1:]:
                    other_wf = self._workflows.pop(other_id, None)
                    if other_wf is None:
                        continue
                    # Move all nodes from the other workflow into target.
                    for nid, n in other_wf.nodes.items():
                        target_wf.nodes[nid] = n
                        self._session_to_workflow[nid] = workflow_id
                    logger.info(
                        "Merged workflow %s into %s (%d nodes)",
                        other_id, workflow_id, len(other_wf.nodes),
                    )

            # Also adopt any deps that weren't in any workflow yet
            # (possible when some deps had workflows and others didn't).
            wf = self._workflows[workflow_id]
            for dep_id in deps:
                if dep_id not in self._session_to_workflow:
                    dep_state = self._states.get(dep_id)
                    if dep_state:
                        dep_node = TaskNode(
                            node_id=dep_id,
                            objective=dep_state.objective,
                            context_blobs=[],
                            system_prompt_mode=dep_state.system_prompt_mode,
                            timeout=DEFAULT_TIMEOUT,
                            depends_on=[],
                            nesting_depth=dep_state.nesting_depth,
                            parent_thread_id=dep_state.parent_thread_id,
                            status=dep_state.status,
                            result=dep_state.result,
                            error=dep_state.error,
                        )
                        wf.nodes[dep_id] = dep_node
                        self._session_to_workflow[dep_id] = workflow_id

            wf.nodes[session_id] = node
            self._session_to_workflow[session_id] = workflow_id

            # Check if all deps are already done.
            all_done = all(
                self._states.get(d) and self._states[d].status in ("completed", "timeout")
                for d in deps
            )
            any_failed = any(
                self._states.get(d) and self._states[d].status == "failed"
                for d in deps
            )

            if any_failed:
                node.status = "failed"
                node.error = "dependency failed before task was started"
                logger.info("Sub-session %s skipped (dependency failed)", session_id)
                # Create a minimal state so _report and list_all work.
                state = SubSessionState(
                    session_id=session_id, objective=objective,
                    parent_thread_id=parent_thread_id,
                    root_thread_id=root_thread_id,
                    system_prompt_mode=system_prompt_mode,
                    status="failed", created_at=now,
                    error=node.error,
                )
                self._states[session_id] = state
                asyncio.get_running_loop().create_task(
                    self._report(state, f"[SUB-SESSION {session_id} FAILED] dependency failed")
                )
                return session_id

            if not all_done:
                node.status = "pending"
                # Create a placeholder state so list_all/cancel work.
                state = SubSessionState(
                    session_id=session_id, objective=objective,
                    parent_thread_id=parent_thread_id,
                    root_thread_id=root_thread_id,
                    system_prompt_mode=system_prompt_mode,
                    status="pending", created_at=now,
                    nesting_depth=nesting_depth,
                )
                self._states[session_id] = state
                logger.info(
                    "Sub-session %s registered as pending (deps=%s, workflow=%s)",
                    session_id, deps, workflow_id,
                )
                return session_id

            # All deps done — collect their results as context.
            node.context_blobs = self._collect_dep_results(deps) + node.context_blobs
        else:
            # No dependencies — create a single-node workflow.
            workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
            wf = Workflow(
                workflow_id=workflow_id,
                parent_thread_id=parent_thread_id,
                created_at=now,
            )
            wf.nodes[session_id] = node
            self._workflows[workflow_id] = wf
            self._session_to_workflow[session_id] = workflow_id

        # -- Check time gate before starting --
        if node.not_before and not self._time_gate_met(node.not_before):
            node.status = "pending"
            state = SubSessionState(
                session_id=session_id, objective=objective,
                parent_thread_id=parent_thread_id,
                root_thread_id=root_thread_id,
                system_prompt_mode=system_prompt_mode,
                status="pending", created_at=now,
                nesting_depth=nesting_depth,
            )
            self._states[session_id] = state
            self._schedule_time_gate(session_id, node.not_before)
            logger.info(
                "Sub-session %s pending on time gate (not_before=%s)",
                session_id, node.not_before.isoformat(),
            )
            return session_id

        # -- Spawn immediately --
        return self._start_node(node, prior_messages, continuation_depth, continued_from)

    def _collect_dep_results(self, dep_ids: list[str]) -> list[str]:
        """Gather result texts from completed dependency sessions."""
        blobs = []
        for dep_id in dep_ids:
            dep_state = self._states.get(dep_id)
            if dep_state and dep_state.result:
                blobs.append(
                    f"[Result from {dep_id} ({dep_state.objective[:80]})]\n"
                    f"{dep_state.result}"
                )
        return blobs

    @staticmethod
    def _time_gate_met(not_before: datetime) -> bool:
        """Return True if the current time is at or past *not_before*."""
        now = datetime.now(timezone.utc)
        # Ensure comparison is tz-aware.
        if not_before.tzinfo is None:
            not_before = not_before.replace(tzinfo=timezone.utc)
        return now >= not_before

    def _schedule_time_gate(self, session_id: str, not_before: datetime) -> None:
        """Schedule an asyncio callback to re-check a pending node at *not_before*."""
        now = datetime.now(timezone.utc)
        if not_before.tzinfo is None:
            not_before = not_before.replace(tzinfo=timezone.utc)
        delay = max((not_before - now).total_seconds(), 0.1)
        logger.info(
            "Scheduling time-gate wakeup for %s in %.0fs",
            session_id, delay,
        )
        asyncio.get_running_loop().call_later(
            delay,
            lambda: asyncio.ensure_future(self._time_gate_wakeup(session_id)),
        )

    async def _time_gate_wakeup(self, session_id: str) -> None:
        """Called when a time gate expires — trigger dependency resolution."""
        logger.info("Time gate expired for %s — resolving", session_id)
        # Re-run resolution which will now pass the time check.
        await self._resolve_dependents(session_id)
        # Also check the node itself if it has no deps (standalone time gate).
        wf_id = self._session_to_workflow.get(session_id)
        if not wf_id:
            return
        wf = self._workflows.get(wf_id)
        if not wf:
            return
        node = wf.nodes.get(session_id)
        if node and node.status == "pending":
            # Standalone time-gated node (no deps, or all deps already done).
            deps_ok = all(
                self._states.get(d) and self._states[d].status in ("completed", "timeout")
                for d in node.depends_on
            ) if node.depends_on else True
            if deps_ok and self._time_gate_met(node.not_before):
                if node.depends_on:
                    node.context_blobs = self._collect_dep_results(node.depends_on) + node.context_blobs
                self._start_node(node)

    @staticmethod
    def _parse_not_before(value: str) -> Optional[datetime]:
        """Parse a not_before string (ISO-8601) into a tz-aware datetime."""
        from dateutil import parser as dateutil_parser
        try:
            dt = dateutil_parser.parse(value)
            if dt.tzinfo is None:
                # Assume the configured timezone from prompt_assembler.
                from wintermute.prompt_assembler import _timezone
                try:
                    tz = ZoneInfo(_timezone)
                except Exception:
                    tz = timezone.utc
                dt = dt.replace(tzinfo=tz)
            return dt
        except (ValueError, OverflowError) as exc:
            logger.warning("Could not parse not_before '%s': %s", value, exc)
            return None

    def _start_node(
        self,
        node: TaskNode,
        prior_messages: Optional[list[dict]] = None,
        continuation_depth: int = 0,
        continued_from: Optional[str] = None,
    ) -> str:
        """Create the SubSessionState + asyncio.Task and start the worker."""
        session_id = node.node_id
        node.status = "running"

        existing = self._states.get(session_id)
        state = SubSessionState(
            session_id=session_id,
            objective=node.objective,
            parent_thread_id=node.parent_thread_id,
            root_thread_id=node.root_thread_id,
            system_prompt_mode=node.system_prompt_mode,
            status="running",
            created_at=existing.created_at if existing else datetime.now(timezone.utc).isoformat(),
            nesting_depth=node.nesting_depth,
            messages=list(prior_messages) if prior_messages else [],
            continuation_depth=continuation_depth,
            continued_from=continued_from,
            tool_names=node.tool_names,
        )
        self._states[session_id] = state

        task = asyncio.create_task(
            self._run(state, node.context_blobs, node.timeout),
            name=f"sub_session_{session_id}",
        )
        self._tasks[session_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(session_id, None))

        logger.info(
            "Sub-session %s spawned (parent=%s mode=%s timeout=%ds depth=%d nest=%d)",
            session_id, node.parent_thread_id, node.system_prompt_mode,
            node.timeout, continuation_depth, node.nesting_depth,
        )
        return session_id

    async def _resolve_dependents(self, session_id: str) -> None:
        """Check if any pending nodes can now be started after *session_id* finished."""
        workflow_id = self._session_to_workflow.get(session_id)
        if not workflow_id:
            return
        wf = self._workflows.get(workflow_id)
        if not wf:
            return

        # Sync the node's status from SubSessionState.
        state = self._states.get(session_id)
        node = wf.nodes.get(session_id)
        if node and state:
            node.status = state.status
            node.result = state.result
            node.error = state.error

        completed = sum(1 for n in wf.nodes.values() if n.status == "completed")
        total = len(wf.nodes)

        for nid, n in list(wf.nodes.items()):
            if n.status != "pending":
                continue

            dep_states = {d: self._states.get(d) for d in n.depends_on}
            any_failed = any(
                s and s.status == "failed" for s in dep_states.values()
            )
            all_done = all(
                s and s.status in ("completed", "timeout") for s in dep_states.values()
            )

            if any_failed:
                n.status = "failed"
                n.error = "dependency failed"
                # Update the placeholder SubSessionState too.
                placeholder = self._states.get(nid)
                if placeholder:
                    placeholder.status = "failed"
                    placeholder.error = n.error
                    placeholder.completed_at = datetime.now(timezone.utc).isoformat()
                await self._report(
                    placeholder or SubSessionState(
                        session_id=nid, objective=n.objective,
                        parent_thread_id=n.parent_thread_id,
                        system_prompt_mode=n.system_prompt_mode,
                        status="failed", created_at="",
                        error=n.error,
                    ),
                    f"[SUB-SESSION {nid} FAILED] dependency failed",
                )
                # Recursively resolve anything depending on this now-failed node.
                await self._resolve_dependents(nid)

            elif all_done:
                # Check time gate before starting.
                if n.not_before and not self._time_gate_met(n.not_before):
                    self._schedule_time_gate(nid, n.not_before)
                    logger.info(
                        "Deps ready for %s but time gate not met (not_before=%s)",
                        nid, n.not_before.isoformat(),
                    )
                    continue
                n.context_blobs = self._collect_dep_results(n.depends_on) + n.context_blobs
                logger.info(
                    "All deps ready for %s — auto-starting (workflow=%s, %d/%d done)",
                    nid, workflow_id, completed, total,
                )
                self._start_node(n)

        # Check if the whole workflow is terminal.
        all_terminal = all(
            n.status in ("completed", "failed", "timeout") for n in wf.nodes.values()
        )
        if all_terminal:
            any_fail = any(n.status == "failed" for n in wf.nodes.values())
            wf.status = "failed" if any_fail else "completed"
            logger.info("Workflow %s %s (%d nodes)", workflow_id, wf.status, total)
            self._cleanup_workflow(workflow_id)

    def cancel_for_thread(self, thread_id: str) -> int:
        """
        Cancel all running/pending sub-sessions whose parent is thread_id.
        Returns the number of sessions cancelled.
        """
        cancelled = 0
        needs_resolve: list[str] = []
        for sid, state in list(self._states.items()):
            if state.parent_thread_id != thread_id:
                continue
            if state.status == "running":
                task = self._tasks.get(sid)
                if task and not task.done():
                    # Mark failed before cancelling so _resolve_dependents sees
                    # the correct state immediately (CancelledError handler would
                    # otherwise race with ensure_future(_resolve_dependents)).
                    state.status = "failed"
                    state.error = "Cancelled"
                    state.completed_at = datetime.now(timezone.utc).isoformat()
                    task.cancel()
                    logger.info("Cancelled sub-session %s (thread reset)", sid)
                    cancelled += 1
                    needs_resolve.append(sid)
            elif state.status == "pending":
                state.status = "failed"
                state.error = "Cancelled"
                state.completed_at = datetime.now(timezone.utc).isoformat()
                logger.info("Cancelled pending sub-session %s (thread reset)", sid)
                cancelled += 1
                needs_resolve.append(sid)
        # Resolve dependents so nothing stays deadlocked.
        for sid in needs_resolve:
            asyncio.ensure_future(self._resolve_dependents(sid))
        return cancelled

    def _cleanup_workflow(self, workflow_id: str) -> None:
        """Remove a completed/failed workflow and its sessions from tracking dicts.

        Keeps the last 50 completed workflows for the debug UI (list_workflows /
        list_all) and purges the rest to prevent unbounded memory growth.
        """
        _MAX_COMPLETED = 50
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return

        # Count how many completed workflows already exist (excluding this one).
        completed_wfs = [
            wid for wid, w in self._workflows.items()
            if w.status in ("completed", "failed") and wid != workflow_id
        ]
        if len(completed_wfs) >= _MAX_COMPLETED:
            # Remove the oldest completed workflows to stay within budget.
            to_purge = completed_wfs[:len(completed_wfs) - _MAX_COMPLETED + 1]
            for old_wid in to_purge:
                old_wf = self._workflows.pop(old_wid, None)
                if old_wf:
                    for nid in old_wf.nodes:
                        self._states.pop(nid, None)
                        self._tasks.pop(nid, None)
                        self._session_to_workflow.pop(nid, None)
                        self._worker_spawned.pop(nid, None)
                        self._aggregated_parents.discard(nid)
                    logger.debug("Purged completed workflow %s from memory", old_wid)

    def list_active(self) -> list[dict]:
        """Return serialisable state dicts for all non-completed sub-sessions."""
        return [
            self._serialise(state)
            for state in self._states.values()
            if state.status in ("running", "pending")
        ]

    def list_all(self) -> list[dict]:
        """Return serialisable state dicts for all known sub-sessions, newest first."""
        return sorted(
            [self._serialise(state) for state in self._states.values()],
            key=lambda s: s["created_at"],
            reverse=True,
        )

    def get_messages(self, session_id: str) -> list:
        """Return the in-memory message list for a sub-session, or empty list."""
        state = self._states.get(session_id)
        if state is None:
            return []
        return list(state.messages)

    def _serialise(self, state: SubSessionState) -> dict:
        """Return state as a dict, omitting the (potentially large) messages list."""
        d = {k: v for k, v in state.__dict__.items() if k != "messages"}
        d["tool_call_count"] = len(state.tool_calls_log)
        # Enrich with workflow metadata.
        wf_id = self._session_to_workflow.get(state.session_id)
        d["workflow_id"] = wf_id
        if wf_id:
            wf = self._workflows.get(wf_id)
            node = wf.nodes.get(state.session_id) if wf else None
            d["depends_on"] = node.depends_on if node else []
            d["not_before"] = node.not_before.isoformat() if (node and node.not_before) else None
        else:
            d["depends_on"] = []
        return d

    def list_workflows(self) -> list[dict]:
        """Return serialisable workflow dicts, newest first."""
        result = []
        for wf in self._workflows.values():
            nodes = []
            for n in wf.nodes.values():
                nodes.append({
                    "node_id": n.node_id,
                    "objective": n.objective,
                    "status": n.status,
                    "depends_on": n.depends_on,
                    "not_before": n.not_before.isoformat() if n.not_before else None,
                    "result_preview": (n.result or "")[:120] if n.result else None,
                    "error": n.error,
                })
            result.append({
                "workflow_id": wf.workflow_id,
                "parent_thread_id": wf.parent_thread_id,
                "status": wf.status,
                "created_at": wf.created_at,
                "node_count": len(wf.nodes),
                "nodes": nodes,
            })
        result.sort(key=lambda w: w["created_at"], reverse=True)
        return result

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
            await self._resolve_dependents(state.session_id)

        except asyncio.TimeoutError:
            state.status = "timeout"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            try:
                database.save_interaction_log(
                    _time.time(), "sub_session", state.session_id,
                    self._pool.last_used,
                    state.objective[:500], "timeout", "timeout",
                )
            except Exception:
                pass
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
                    tool_names=state.tool_names,
                )
                # Move the continuation into the original workflow so it
                # doesn't create an orphaned single-node workflow.
                orig_wf_id = self._session_to_workflow.get(state.session_id)
                cont_wf_id = self._session_to_workflow.get(cont_id)
                if orig_wf_id and cont_wf_id and orig_wf_id != cont_wf_id:
                    orig_wf = self._workflows.get(orig_wf_id)
                    cont_wf = self._workflows.pop(cont_wf_id, None)
                    if orig_wf and cont_wf and cont_id in cont_wf.nodes:
                        orig_wf.nodes[cont_id] = cont_wf.nodes[cont_id]
                        self._session_to_workflow[cont_id] = orig_wf_id
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
            await self._resolve_dependents(state.session_id)

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
            try:
                database.save_interaction_log(
                    _time.time(), "sub_session", state.session_id,
                    self._pool.last_used,
                    state.objective[:500], str(exc)[:500], "error",
                )
            except Exception:
                pass
            msg = f"[SUB-SESSION {state.session_id} FAILED] {exc}"
            logger.exception("Sub-session %s failed", state.session_id)
            await self._report(state, msg)
            await self._resolve_dependents(state.session_id)

    _AGENDA_NO_ACTION = "[NO_ACTION]"

    async def _report(self, state: SubSessionState, text: str) -> None:
        """Deliver result to parent thread, or log if fire-and-forget.

        For nested sub-sessions (parent is another sub-session), individual
        reports are suppressed.  Instead, when all children of a parent
        sub-session are terminal, an aggregated result is delivered to the
        root (user-facing) thread.  This is fully deterministic — no LLM
        inference is involved in the routing or aggregation.
        """
        # Agenda reviews that need no action — suppress delivery entirely.
        if self._AGENDA_NO_ACTION in text:
            logger.info(
                "Sub-session %s: suppressing report (agenda no-action)",
                state.session_id,
            )
            return

        # Nested sub-session: suppress individual report, check aggregation.
        if state.parent_thread_id and state.parent_thread_id.startswith("sub_"):
            logger.info(
                "Nested sub-session %s completed (parent=%s) — deferring to aggregated delivery",
                state.session_id, state.parent_thread_id,
            )
            await self._check_nested_aggregation(state.session_id)
            return

        # Append workflow progress if this session is part of a DAG.
        wf_id = self._session_to_workflow.get(state.session_id)
        if wf_id:
            wf = self._workflows.get(wf_id)
            if wf and len(wf.nodes) > 1:
                done = sum(
                    1 for n in wf.nodes.values()
                    if n.status in ("completed", "failed", "timeout")
                )
                total = len(wf.nodes)
                text += f"\n\n(workflow {wf_id}: {done}/{total} nodes complete)"
                if done == total:
                    any_fail = any(n.status == "failed" for n in wf.nodes.values())
                    if any_fail:
                        text += f"\n[WORKFLOW {wf_id} FINISHED WITH ERRORS]"
                    else:
                        text += f"\n[WORKFLOW {wf_id} COMPLETE]"

        if state.parent_thread_id:
            try:
                await self._enqueue(text, state.parent_thread_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sub-session %s failed to report back: %s", state.session_id, exc)
        else:
            logger.info("Sub-session %s (fire-and-forget): %s", state.session_id, text[:300])

    async def _check_nested_aggregation(self, completed_session_id: str) -> None:
        """Check if all children of a parent sub-session are terminal.

        When a sub-session (depth 1) spawns children (depth 2), each child
        has parent_thread_id = the parent's session_id.  This method checks
        whether ALL such siblings have reached a terminal state.  If so, it
        concatenates their results and delivers one aggregated message to the
        root thread (the original user-facing chat thread).

        The _aggregated_parents set prevents duplicate delivery when multiple
        children complete near-simultaneously (safe because asyncio is
        cooperative and there is no await between the membership check and
        the set.add).
        """
        state = self._states.get(completed_session_id)
        if not state or not state.parent_thread_id:
            return
        parent_sid = state.parent_thread_id
        root_tid = state.root_thread_id
        if not root_tid:
            logger.warning(
                "Nested sub-session %s has no root_thread_id — cannot aggregate",
                completed_session_id,
            )
            return

        # Already delivered for this parent?
        if parent_sid in self._aggregated_parents:
            return

        # Find all children of the same parent sub-session.
        siblings = [
            s for s in self._states.values()
            if s.parent_thread_id == parent_sid
        ]

        all_terminal = all(
            s.status in ("completed", "failed", "timeout") for s in siblings
        )
        if not all_terminal:
            logger.debug(
                "Nested aggregation: %d/%d children of %s are terminal",
                sum(1 for s in siblings if s.status in ("completed", "failed", "timeout")),
                len(siblings), parent_sid,
            )
            return

        # Mark as aggregated BEFORE the await to prevent duplicate delivery.
        self._aggregated_parents.add(parent_sid)

        # Build aggregated report.
        parts = []
        any_fail = False
        for s in siblings:
            header = f"**[{s.session_id}]** {s.objective[:100]}"
            if s.status == "completed" and s.result:
                parts.append(f"{header}\n{s.result}")
            elif s.status == "failed":
                parts.append(f"{header}\n[FAILED: {s.error}]")
                any_fail = True
            elif s.status == "timeout":
                parts.append(f"{header}\n[TIMEOUT — partial progress in logs]")
            else:
                parts.append(f"{header}\n[{s.status}]")

        parent_state = self._states.get(parent_sid)
        parent_obj = parent_state.objective[:80] if parent_state else parent_sid

        status_word = "FINISHED WITH ERRORS" if any_fail else "COMPLETE"
        text = (
            f"[NESTED TASKS {status_word} — {len(siblings)} task(s) for: {parent_obj}]\n\n"
            + "\n\n---\n\n".join(parts)
        )

        logger.info(
            "Delivering aggregated results (%d children of %s) to root thread %s",
            len(siblings), parent_sid, root_tid,
        )
        try:
            await self._enqueue(text, root_tid)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to deliver aggregated nested results to %s: %s",
                root_tid, exc,
            )

    async def _worker_loop(self, state: SubSessionState, context_blobs: list[str]) -> str:
        """
        Run a full tool-use inference loop in isolation.

        The worker has its own in-memory message list — nothing is read from
        or written to the conversation DB.  Tool schemas are filtered by the
        session's system_prompt_mode (e.g. "minimal" workers only get
        execution + research tools).

        state.messages is used directly (not a local copy) so that the timeout
        handler always has access to the latest message history for continuation.

        Turing Protocol hooks are fired at three phases:
          - post_inference:  after the model produces a text-only (final)
            response — the ``objective_completion`` validator decides whether
            the sub-session may exit or must keep working.
          - pre_execution:   before each tool call (can block execution).
          - post_execution:  after each tool call (can annotate results).
        """
        # Build the tool set for this worker based on its mode.
        nl_enabled = self._nl_translation_config.get("enabled", False)
        nl_tools = self._nl_translation_config.get("tools", set()) if nl_enabled else None
        if state.tool_names:
            # Explicit tool whitelist — bypass category system entirely.
            allowed = set(state.tool_names)
            tool_schemas = [
                s for s in tool_module.TOOL_SCHEMAS
                if s["function"]["name"] in allowed
            ]
        else:
            categories = _MODE_TOOL_CATEGORIES.get(
                state.system_prompt_mode, {"execution", "research"}
            )
            tool_schemas = tool_module.get_tool_schemas(categories, nl_tools=nl_tools)

        tp_enabled = bool(self._tp_pool and self._tp_pool.enabled)
        tp_corrected = False  # single-shot: only one objective correction allowed
        tool_calls_made: list[str] = []  # accumulated across the session

        if state.messages:
            # Resuming from a prior timed-out session.  The full prior history
            # is already in state.messages; just append a resumption note.
            state.messages.append({
                "role":    "user",
                "content": prompt_loader.load(
                    "WORKER_CONTINUATION.txt",
                    continuation_depth=state.continuation_depth,
                ),
            })
        else:
            # Fresh start — build the initial conversation.
            system_prompt = self._build_system_prompt(
                state.system_prompt_mode, state.objective, context_blobs,
                thread_id=state.parent_thread_id,
            )
            state.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": state.objective},
            ]

        while True:
            response = await self._pool.call(
                messages=state.messages,
                tools=tool_schemas,
            )

            choice = response.choices[0]

            # Log this inference round
            try:
                content_text = (choice.message.content or "").strip()
                if choice.message.tool_calls:
                    tc_names = [tc.function.name for tc in choice.message.tool_calls]
                    output_summary = f"[tool_calls: {', '.join(tc_names)}]"
                else:
                    output_summary = content_text[:500]
                raw_output_data = json.dumps({
                    "content": content_text[:500],
                    "tool_calls": [tc.function.name for tc in (choice.message.tool_calls or [])],
                })
                if tool_calls_made:
                    _recent = tool_calls_made[-3:]
                    _input_summary = f"[round {len(tool_calls_made)+1}, after: {', '.join(_recent)}]"
                else:
                    _input_summary = state.objective[:500]
                database.save_interaction_log(
                    _time.time(), "sub_session", state.session_id,
                    self._pool.last_used,
                    _input_summary, output_summary, "ok",
                    raw_output=raw_output_data,
                )
            except Exception:
                pass

            if choice.message.tool_calls:
                state.messages.append(choice.message)

                for tc in choice.message.tool_calls:
                    try:
                        inputs = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        inputs = {}

                    name = tc.function.name
                    nl_was_translated = False

                    # -- NL Translation: expand description to structured args --
                    if nl_enabled and nl_translator.is_nl_tool_call(name, inputs):
                        from wintermute.prompt_assembler import _timezone as _pa_tz
                        translated = await nl_translator.translate_nl_tool_call(
                            self._nl_translation_pool, name, inputs["description"],
                            thread_id=state.session_id,
                            timezone_str=_pa_tz,
                        )
                        # Log the NL translation call.
                        try:
                            database.save_interaction_log(
                                _time.time(), "nl_translation", state.session_id,
                                self._nl_translation_pool.last_used,
                                inputs["description"],
                                json.dumps(translated) if translated is not None else "null",
                                "ok" if translated is not None else "error",
                            )
                        except Exception:
                            pass
                        if translated is None:
                            state.messages.append({
                                "role":         "tool",
                                "tool_call_id": tc.id,
                                "content":      "[TRANSLATION ERROR] Failed to translate natural-language tool call. Please try rephrasing or use structured arguments.",
                            })
                            continue
                        if isinstance(translated, dict) and "error" in translated:
                            state.messages.append({
                                "role":         "tool",
                                "tool_call_id": tc.id,
                                "content":      f"[CLARIFICATION NEEDED] {translated.get('clarification_needed', translated['error'])}",
                            })
                            continue
                        if isinstance(translated, list):
                            combined_results = []
                            for i, item_args in enumerate(translated):
                                item_result = await asyncio.get_running_loop().run_in_executor(
                                    None, lambda _n=name, _a=item_args: tool_module.execute_tool(
                                        _n, _a,
                                        thread_id=state.session_id,
                                        nesting_depth=state.nesting_depth,
                                        parent_thread_id=state.parent_thread_id,
                                    )
                                )
                                summary = ", ".join(f"{k}={v!r}" for k, v in item_args.items() if k != "description")
                                combined_results.append(f"[{i+1}] [Translated to: {summary}] {item_result}")
                                tool_calls_made.append(name)
                                state.tool_calls_log.append((name, item_result[:120].replace("\n", " ")))
                                try:
                                    database.save_interaction_log(
                                        _time.time(), "tool_call", state.session_id,
                                        self._pool.last_used,
                                        json.dumps({"tool": name, "arguments": json.dumps(item_args)}),
                                        item_result[:500], "ok",
                                    )
                                except Exception:
                                    pass
                            state.messages.append({
                                "role":         "tool",
                                "tool_call_id": tc.id,
                                "content":      "\n\n".join(combined_results),
                            })
                            continue
                        inputs = translated
                        nl_was_translated = True

                    # -- Turing Protocol: pre_execution phase --
                    if tp_enabled:
                        pre_result = await self._run_tp_phase(
                            phase="pre_execution",
                            state=state,
                            tool_calls_made=tool_calls_made,
                            assistant_response=(choice.message.content or ""),
                            tool_name=name,
                            tool_args=inputs,
                        )
                        if pre_result and pre_result.correction:
                            logger.warning(
                                "Sub-session %s: pre_execution hook blocked tool %s: %s",
                                state.session_id, name,
                                pre_result.correction[:200],
                            )
                            state.messages.append({
                                "role":         "tool",
                                "tool_call_id": tc.id,
                                "content":      f"[BLOCKED BY TURING PROTOCOL] {pre_result.correction}",
                            })
                            continue

                    result = await asyncio.get_running_loop().run_in_executor(
                        None, lambda _n=name, _i=inputs: tool_module.execute_tool(
                            _n, _i,
                            thread_id=state.session_id,
                            nesting_depth=state.nesting_depth,
                            parent_thread_id=state.parent_thread_id,
                        )
                    )
                    tool_calls_made.append(name)

                    # Prepend translation summary if NL translation was used.
                    if nl_was_translated:
                        summary = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
                        result = f"[Translated to: {summary}] {result}"

                    # -- Turing Protocol: post_execution phase --
                    if tp_enabled:
                        post_result = await self._run_tp_phase(
                            phase="post_execution",
                            state=state,
                            tool_calls_made=tool_calls_made,
                            assistant_response=(choice.message.content or ""),
                            tool_name=name,
                            tool_result=result,
                        )
                        if post_result and post_result.correction:
                            result += f"\n\n[TURING PROTOCOL WARNING] {post_result.correction}"

                    # Log the tool call
                    try:
                        database.save_interaction_log(
                            _time.time(), "tool_call", state.session_id,
                            self._pool.last_used,
                            json.dumps({"tool": name, "arguments": tc.function.arguments}),
                            result[:500], "ok",
                        )
                    except Exception:
                        pass
                    # Track progress on state so the timeout handler can report
                    # and continue from it.
                    result_preview = result[:120].replace("\n", " ")
                    state.tool_calls_log.append((name, result_preview))
                    logger.debug("Sub-session %s tool %s -> %s",
                                 state.session_id, name, result[:200])
                    state.messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue

            # -- Terminal response: model produced text without tool calls --
            final_text = (choice.message.content or "").strip()

            # -- Turing Protocol: post_inference phase (objective gatekeeper) --
            # Single-shot: one correction at most, no re-checking.
            if tp_enabled and not tp_corrected:
                pi_result = await self._run_tp_phase(
                    phase="post_inference",
                    state=state,
                    tool_calls_made=tool_calls_made,
                    assistant_response=final_text,
                )
                if pi_result and pi_result.correction:
                    tp_corrected = True
                    logger.info(
                        "Sub-session %s: objective not met — injecting "
                        "correction and resuming loop",
                        state.session_id,
                    )
                    # Append the model's response and the correction as a
                    # system message, then re-enter the inference loop.
                    state.messages.append({
                        "role": "assistant",
                        "content": final_text,
                    })
                    state.messages.append({
                        "role": "user",
                        "content": pi_result.correction,
                    })
                    continue  # back to while True → next inference call

            return final_text

    # ------------------------------------------------------------------
    # Turing Protocol helper for sub-session phases
    # ------------------------------------------------------------------

    async def _run_tp_phase(
        self,
        phase: str,
        state: SubSessionState,
        tool_calls_made: list[str],
        assistant_response: str = "",
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_result: Optional[str] = None,
    ) -> Optional[turing_protocol_module.TuringResult]:
        """Run Turing Protocol hooks for a specific phase in sub-session scope.

        Returns TuringResult if violations confirmed, None otherwise.
        """
        if not self._tp_pool or not self._tp_pool.enabled:
            return None

        # Quick check: any hooks for this phase+scope?
        hooks = turing_protocol_module.get_hooks(
            self._tp_validators,
            phase_filter=phase,
            scope_filter="sub_session",
        )
        if not hooks:
            return None

        try:
            return await turing_protocol_module.run_turing_protocol(
                pool=self._tp_pool,
                user_message=state.objective,
                assistant_response=assistant_response,
                tool_calls_made=tool_calls_made,
                active_sessions=[],
                enabled_validators=self._tp_validators,
                thread_id=state.session_id,
                phase=phase,
                scope="sub_session",
                objective=state.objective,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Sub-session %s: Turing Protocol %s check raised (non-fatal)",
                state.session_id, phase,
            )
            return None

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(
        mode: str,
        objective: str,
        context_blobs: list[str],
        thread_id: Optional[str] = None,
    ) -> str:
        if mode == "none":
            base = ""
        elif mode == "minimal":
            base = prompt_loader.load("WORKER_MINIMAL.txt")
        elif mode == "full":
            base = prompt_assembler.assemble(thread_id=thread_id)
        else:  # "base_only"
            raw = prompt_loader.load("BASE_PROMPT.txt")
            base = f"# Core Instructions\n\n{raw}" if raw else ""

        parts = [base] if base else []

        if context_blobs:
            blobs_text = "\n\n".join(context_blobs)
            parts.append(f"# Task Context\n\n{blobs_text}")

        parts.append(prompt_loader.load("WORKER_OBJECTIVE.txt", objective=objective))

        return "\n\n---\n\n".join(parts)
