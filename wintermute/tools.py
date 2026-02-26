"""
Tool definitions and execution for the AI assistant.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).
The dispatcher ``execute_tool`` is the single entry point used by the LLM thread.

Tool categories
---------------
  "execution"     – shell, file I/O (available to all sub-session modes)
  "research"      – web search, URL fetching (available to all sub-session modes)
  "orchestration" – memory, tasks, skills, sub-session spawning (main agent
                    and "full"-mode sub-sessions only)
"""

import json
import logging
import os
import subprocess
import time
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from wintermute.infra import database
from wintermute.infra import prompt_assembler

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

SEARXNG_URL = os.environ.get("WINTERMUTE_SEARXNG_URL", "http://127.0.0.1:8888")

# ---------------------------------------------------------------------------
# In-memory tool call log (bounded ring buffer for the debug UI)
# ---------------------------------------------------------------------------

_TOOL_CALL_LOG: deque[dict] = deque(maxlen=500)


def get_tool_call_log() -> list[dict]:
    """Return a copy of the tool call log, newest first."""
    return list(reversed(_TOOL_CALL_LOG))

# Maximum nesting depth for sub-session spawning.
# 0 = main agent, 1 = sub-session, 2 = sub-sub-session (max).
MAX_NESTING_DEPTH = 2

# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, parameters: dict) -> dict:
    """Wrap a function schema in the OpenAI tool envelope."""
    return {
        "type": "function",
        "function": {
            "name":        name,
            "description": description,
            "parameters":  parameters,
        },
    }


TOOL_SCHEMAS = [
    _fn(
        "spawn_sub_session",
        (
            "Spawn an autonomous background worker. Returns a session_id immediately; "
            "result arrives as a [SYSTEM EVENT]."
        ),
        {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": (
                        "Task description. Worker has NO conversation access — "
                        "include ALL concrete values (URLs, tokens, credentials, IDs, "
                        "parameters) verbatim. Never say 'the provided token'; "
                        "paste the actual token into the objective or context_blobs."
                    ),
                },
                "context_blobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Context snippets passed verbatim to the worker. "
                        "Use this for large data: API tokens, request bodies, "
                        "file contents, credentials. Unneeded when using depends_on."
                    ),
                },
                "system_prompt_mode": {
                    "type": "string",
                    "enum": ["minimal", "full", "base_only", "none"],
                    "description": (
                        "'minimal' (default): lightweight agent, no memories/skills. "
                        "'full': complete context (memories + tasks + skills). "
                        "'base_only': core instructions only. "
                        "'none': no system prompt."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds before timeout (default: 300).",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Session IDs to wait for; results auto-passed as context. Prefer depends_on_previous.",
                },
                "depends_on_previous": {
                    "type": "boolean",
                    "description": "Depend on all sessions spawned in this context; avoids manually tracking IDs.",
                },
                "not_before": {
                    "type": "string",
                    "description": "Earliest start time (ISO-8601). Waits even if deps are satisfied.",
                },
                "profile": {
                    "type": "string",
                    "description": (
                        "Named tool profile (e.g. 'researcher', 'file_worker'). "
                        "Overrides system_prompt_mode and sets an optimised tool set. "
                        "See available profiles in config."
                    ),
                },
            },
            "required": ["objective"],
        },
    ),
    _fn(
        "task",
        (
            "Manage tracked tasks — goals, reminders, and scheduled actions. "
            "Use action 'add' to create, 'complete' to finish, 'pause'/'resume' to control schedules, "
            "'update' to modify, 'delete' to remove, 'list' to show."
        ),
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "complete", "pause", "resume", "delete", "list"],
                    "description": "Operation to perform.",
                },
                "content": {
                    "type": "string",
                    "description": "Task description (for add/update).",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (for update/complete/pause/resume/delete).",
                },
                "reason": {
                    "type": "string",
                    "description": "Required for complete: evidence why this task is truly finished.",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "1 (urgent) to 10 (low), default 5.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "all"],
                    "description": "For list: filter (default: active).",
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly", "monthly", "interval"],
                    "description": "Omit for unscheduled tasks.",
                },
                "at": {
                    "type": "string",
                    "description": "Time spec. For once: ISO-8601. For recurring: HH:MM.",
                },
                "day_of_week": {
                    "type": "string",
                    "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                    "description": "Required for weekly.",
                },
                "day_of_month": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 31,
                    "description": "Required for monthly.",
                },
                "interval_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Required for interval.",
                },
                "window_start": {
                    "type": "string",
                    "description": "For interval: earliest fire time, HH:MM.",
                },
                "window_end": {
                    "type": "string",
                    "description": "For interval: latest fire time, HH:MM.",
                },
                "ai_prompt": {
                    "type": "string",
                    "description": "AI action to run when schedule fires. Write as a complete task instruction.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Silent execution — no chat delivery. Only valid with ai_prompt.",
                },
            },
            "required": ["action"],
        },
    ),
    _fn(
        "append_memory",
        "Append a fact to MEMORIES.txt. One entry per call",
        {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "Fact or note to append.",
                },
                "source": {
                    "type": "string",
                    "description": "Origin tag for this memory (e.g. 'user_explicit', 'harvest'). Default: 'user_explicit'.",
                },
            },
            "required": ["entry"],
        },
    ),
    _fn(
        "add_skill",
        "Create or overwrite a skill in data/skills/. A summary appears in the system prompt; full content is loaded on demand via read_file.",
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Filename stem without extension (e.g. 'calendar').",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary for the skills index (max 80 chars).",
                },
                "documentation": {
                    "type": "string",
                    "description": "Markdown documentation for the skill. Be concise, max 500 chars.",
                },
            },
            "required": ["skill_name", "summary", "documentation"],
        },
    ),
    _fn(
        "execute_shell",
        "Run a shell command. Returns stdout, stderr, and exit_code.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30).",
                },
            },
            "required": ["command"],
        },
    ),
    _fn(
        "read_file",
        "Read a file and return its contents.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                }
            },
            "required": ["path"],
        },
    ),
    _fn(
        "write_file",
        "Write content to a file, creating parent directories as needed.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write.",
                },
            },
            "required": ["path", "content"],
        },
    ),
    _fn(
        "search_web",
        "Search the web via SearXNG. Returns titles, URLs, and snippets.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    ),
    _fn(
        "fetch_url",
        "Fetch a web page and return it as plain text (HTML stripped).",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of content to return (default: 20000).",
                },
            },
            "required": ["url"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool categories — controls which tools sub-sessions receive
# ---------------------------------------------------------------------------

TOOL_CATEGORIES: dict[str, str] = {
    "execute_shell":      "execution",
    "read_file":          "execution",
    "write_file":         "execution",
    "search_web":         "research",
    "fetch_url":          "research",
    "spawn_sub_session":  "orchestration",
    "task":               "orchestration",
    "append_memory":      "orchestration",
    "add_skill":          "orchestration",
}


NL_TOOL_SCHEMAS = [
    _fn(
        "task",
        (
            "Manage tasks — tracked goals, reminders, and scheduled actions. "
            "Describe what you want in plain English — the system will translate "
            "it into structured arguments."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the task operation. "
                        "Examples: 'add a high-priority task to fix the login bug', "
                        "'remind me daily at 9am to check email', "
                        "'every morning at 8 search for news and summarize', "
                        "'complete task_abc because the server is fixed', "
                        "'pause task_def', 'list all tasks'."
                    ),
                },
            },
            "required": ["description"],
        },
    ),
    _fn(
        "spawn_sub_session",
        (
            "Spawn one or more autonomous background workers as a DAG pipeline. "
            "Describe all tasks in plain English, including sequencing and parallel "
            "branches — the system will translate it into a dependency graph. "
            "Examples: 'research X then summarise it' (sequential), "
            "'check weather in Berlin and Tokyo' (parallel), "
            "'fetch A and B, then combine results' (fan-in)."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the full DAG pipeline: what "
                        "each worker should do, which tasks must run before others, "
                        "and which can run in parallel. Be explicit about sequencing "
                        "(e.g. 'then', 'after that', 'at the same time as')."
                    ),
                },
            },
            "required": ["description"],
        },
    ),
    _fn(
        "add_skill",
        (
            "Save a learned procedure as a reusable skill. Describe what skill "
            "to save and the procedure details in plain English — the system "
            "will format it into a properly structured skill file."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What skill to save: a name and the full procedure. "
                        "Example: 'save a skill called deploy-docker about "
                        "deploying containers: run docker compose up -d ...'"
                    ),
                },
            },
            "required": ["description"],
        },
    ),
]

_NL_SCHEMA_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in NL_TOOL_SCHEMAS
}


def get_tool_schemas(categories: set[str] | None = None,
                     nl_tools: set[str] | None = None) -> list[dict]:
    """Return tool schemas filtered by category, with optional NL substitution.

    If *categories* is None, return all schemas (used by the main agent).
    Otherwise return only schemas whose tool name maps to one of the
    requested categories.

    When *nl_tools* is provided, any tool whose name is in the set gets its
    schema replaced with the simplified single-field NL variant.
    """
    if categories is None:
        schemas = TOOL_SCHEMAS
    else:
        schemas = [
            schema for schema in TOOL_SCHEMAS
            if TOOL_CATEGORIES.get(schema["function"]["name"]) in categories
        ]
    if not nl_tools:
        return schemas
    result = []
    for schema in schemas:
        name = schema["function"]["name"]
        if name in nl_tools and name in _NL_SCHEMA_MAP:
            result.append(_NL_SCHEMA_MAP[name])
        else:
            result.append(schema)
    return result


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# These will be injected by the scheduler thread at startup.
_task_scheduler_ensure = None   # Callable[[str, dict, str|None, str|None, bool], None]
_task_scheduler_remove = None   # Callable[[str], None]
_task_scheduler_list = None     # Callable[[], list[dict]]

# This will be injected by the SubSessionManager at startup.
_sub_session_spawn = None  # Callable[[dict, str], str]

# Event bus — injected by main.py at startup.
_event_bus = None  # Optional[EventBus]


def register_task_scheduler(ensure_fn, remove_fn, list_fn) -> None:
    """Called once by the scheduler thread to provide task schedule management."""
    global _task_scheduler_ensure, _task_scheduler_remove, _task_scheduler_list
    _task_scheduler_ensure = ensure_fn
    _task_scheduler_remove = remove_fn
    _task_scheduler_list = list_fn


def register_sub_session_manager(fn) -> None:
    """Called once by SubSessionManager to provide the spawn hook."""
    global _sub_session_spawn
    _sub_session_spawn = fn


def register_event_bus(bus) -> None:
    """Called once by main.py to provide the event bus."""
    global _event_bus
    _event_bus = bus


def _tool_spawn_sub_session(inputs: dict, thread_id: Optional[str] = None,
                            nesting_depth: int = 0, **_kw) -> str:
    if nesting_depth >= MAX_NESTING_DEPTH:
        return json.dumps({
            "error": (
                f"Maximum nesting depth ({MAX_NESTING_DEPTH}) reached. "
                "Cannot spawn further sub-sessions."
            )
        })
    if _sub_session_spawn is None:
        return json.dumps({"error": "Sub-session manager not ready yet."})
    try:
        context_blobs = list(inputs.get("context_blobs") or [])

        # Auto-inject last user message from parent thread so the worker
        # has access to concrete values (tokens, URLs, IDs) the user provided
        # even if the LLM forgot to copy them into the objective/context_blobs.
        if thread_id:
            try:
                last_user_msg = database.get_last_user_message(thread_id)
                if last_user_msg:
                    context_blobs.append(
                        f"[Parent conversation — last user message]\n{last_user_msg}"
                    )
            except Exception:
                logger.debug("Failed to fetch parent user message", exc_info=True)

        # Query historical outcomes for similar objectives.
        try:
            similar = database.get_similar_outcomes(inputs["objective"], limit=5)
            if similar:
                lines = []
                total_dur = 0
                dur_count = 0
                success_count = 0
                for o in similar:
                    dur = o.get("duration_seconds")
                    tov = o.get("timeout_value")
                    tc = o.get("tool_call_count", 0)
                    st = o.get("status", "?")
                    obj_short = (o.get("objective") or "")[:60]
                    dur_str = f"{dur:.0f}s" if dur is not None else "?"
                    tov_str = f"{tov}s timeout" if tov is not None else "no timeout"
                    cont = o.get("continuation_count", 0)
                    extra = f", continued {cont}x" if cont else ""
                    lines.append(f"- \"{obj_short}\" ({tov_str}): {st} in {dur_str}, {tc} tool calls{extra}")
                    if dur is not None:
                        total_dur += dur
                        dur_count += 1
                    if st == "completed":
                        success_count += 1
                avg_dur = f"{total_dur / dur_count:.0f}s" if dur_count else "N/A"
                rate = f"{success_count * 100 / len(similar):.0f}%"
                blob = (
                    "[Historical Feedback] Similar past sub-sessions:\n"
                    + "\n".join(lines)
                    + f"\nAverage duration: {avg_dur} | Success rate: {rate}"
                )
                context_blobs.insert(0, blob)
        except Exception as exc:
            logger.debug("Historical feedback lookup failed: %s", exc, exc_info=True)

        kwargs = dict(
            objective=inputs["objective"],
            context_blobs=context_blobs,
            parent_thread_id=thread_id,
            system_prompt_mode=inputs.get("system_prompt_mode", "minimal"),
            nesting_depth=nesting_depth + 1,
        )
        if "timeout" in inputs:
            kwargs["timeout"] = int(inputs["timeout"])
        if "depends_on" in inputs:
            kwargs["depends_on"] = inputs["depends_on"]
        if inputs.get("depends_on_previous"):
            kwargs["depends_on_previous"] = True
        if "not_before" in inputs:
            kwargs["not_before"] = inputs["not_before"]
        if "profile" in inputs:
            kwargs["profile"] = inputs["profile"]
        session_id = _sub_session_spawn(**kwargs)
        has_deps = bool(inputs.get("depends_on")) or bool(inputs.get("depends_on_previous"))
        has_gate = bool(inputs.get("not_before"))
        if has_deps or has_gate:
            reasons = []
            if has_deps:
                reasons.append("dependencies")
            if has_gate:
                reasons.append(f"time gate ({inputs['not_before']})")
            status = f"pending (waiting for {' and '.join(reasons)})"
        else:
            status = "started"
        return json.dumps({
            "status": status,
            "session_id": session_id,
            "IMPORTANT": (
                "The worker is now running in the background. "
                "You do NOT have its results yet. "
                "Tell the user you started the task and STOP. "
                "Do NOT fabricate, guess, or anticipate what the worker will find. "
                "The results will be delivered to you later as a [SYSTEM EVENT]. "
                "Only then should you report them to the user."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("spawn_sub_session failed")
        return json.dumps({"error": str(exc)})


def _tool_task(inputs: dict, thread_id: Optional[str] = None,
               parent_thread_id: Optional[str] = None, **_kw) -> str:
    """Unified task tool — handles add/update/complete/pause/resume/delete/list."""
    effective_scope = parent_thread_id or thread_id
    try:
        action = inputs.get("action", "list")

        if action == "add":
            content = inputs.get("content")
            if not content:
                return json.dumps({"error": "content is required for add action"})
            add_thread = inputs.get("thread_id") or effective_scope
            schedule_type = inputs.get("schedule_type")
            ai_prompt = inputs.get("ai_prompt")
            background = bool(inputs.get("background", False))

            # Build schedule config and description for DB storage.
            schedule_config = None
            schedule_desc = None
            if schedule_type:
                from wintermute.workers.scheduler_thread import _describe_schedule
                sched_inputs = {k: inputs[k] for k in
                    ("schedule_type", "at", "day_of_week", "day_of_month",
                     "interval_seconds", "window_start", "window_end")
                    if k in inputs}
                schedule_config = json.dumps(sched_inputs)
                schedule_desc = _describe_schedule(sched_inputs)

            task_id = database.add_task(
                content=content,
                priority=int(inputs.get("priority", 5)),
                thread_id=add_thread,
                schedule_type=schedule_type,
                schedule_desc=schedule_desc,
                schedule_config=schedule_config,
                ai_prompt=ai_prompt,
                background=background,
            )

            # If scheduled, register with APScheduler.
            if schedule_type and _task_scheduler_ensure is not None:
                _task_scheduler_ensure(
                    task_id, json.loads(schedule_config),
                    ai_prompt, add_thread, background,
                )
                database.update_task(task_id, apscheduler_job_id=task_id)

            if _event_bus:
                _event_bus.emit("task.created", task_id=task_id,
                                content=content[:200],
                                schedule_type=schedule_type)
            result = {"status": "ok", "task_id": task_id}
            if schedule_desc:
                result["schedule"] = schedule_desc
            return json.dumps(result)

        elif action == "complete":
            task_id = inputs.get("task_id")
            if not task_id:
                return json.dumps({"error": "task_id is required for complete action"})
            reason = (inputs.get("reason") or "").strip()
            if not reason:
                return json.dumps({"error": "reason is required for complete action — explain why this task is finished"})
            # Remove schedule if any.
            task = database.get_task(task_id)
            if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
                _task_scheduler_remove(task_id)
            ok = database.complete_task(task_id, reason=reason, thread_id=effective_scope)
            if ok and _event_bus:
                _event_bus.emit("task.completed", task_id=task_id, reason=reason[:200])
            return json.dumps({"status": "ok" if ok else "not_found", "reason": reason})

        elif action == "pause":
            task_id = inputs.get("task_id")
            if not task_id:
                return json.dumps({"error": "task_id is required for pause action"})
            task = database.get_task(task_id)
            if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
                _task_scheduler_remove(task_id)
            ok = database.pause_task(task_id)
            return json.dumps({"status": "ok" if ok else "not_found"})

        elif action == "resume":
            task_id = inputs.get("task_id")
            if not task_id:
                return json.dumps({"error": "task_id is required for resume action"})
            ok = database.resume_task(task_id)
            if ok:
                # Re-register schedule with APScheduler.
                task = database.get_task(task_id)
                if task and task.get("schedule_config") and _task_scheduler_ensure:
                    sched = json.loads(task["schedule_config"])
                    _task_scheduler_ensure(
                        task_id, sched,
                        task.get("ai_prompt"), task.get("thread_id"),
                        bool(task.get("background")),
                    )
            return json.dumps({"status": "ok" if ok else "not_found"})

        elif action == "delete":
            task_id = inputs.get("task_id")
            if not task_id:
                return json.dumps({"error": "task_id is required for delete action"})
            task = database.get_task(task_id)
            if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
                _task_scheduler_remove(task_id)
            ok = database.delete_task(task_id)
            return json.dumps({"status": "ok" if ok else "not_found"})

        elif action == "update":
            task_id = inputs.get("task_id")
            if not task_id:
                return json.dumps({"error": "task_id is required for update action"})
            kwargs = {}
            if "content" in inputs:
                kwargs["content"] = inputs["content"]
            if "priority" in inputs:
                kwargs["priority"] = int(inputs["priority"])
            ok = database.update_task(task_id, thread_id=effective_scope, **kwargs)
            return json.dumps({"status": "ok" if ok else "not_found"})

        elif action == "list":
            status = inputs.get("status", "active")
            items = database.list_tasks(status, thread_id=effective_scope)
            # Format for readability
            formatted = []
            for it in items:
                entry = {
                    "id": it["id"],
                    "content": it["content"],
                    "priority": it["priority"],
                    "status": it["status"],
                }
                if it.get("schedule_desc"):
                    entry["schedule"] = it["schedule_desc"]
                if it.get("ai_prompt"):
                    entry["ai_prompt"] = it["ai_prompt"][:100]
                if it.get("last_run_at"):
                    entry["last_run_at"] = it["last_run_at"]
                    entry["run_count"] = it.get("run_count", 0)
                if it.get("last_result_summary"):
                    entry["last_result"] = it["last_result_summary"][:200]
                formatted.append(entry)
            return json.dumps({"tasks": formatted, "count": len(formatted)})

        else:
            return json.dumps({"error": f"Unknown action: {action}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("task tool failed")
        return json.dumps({"error": str(exc)})


def _tool_append_memory(inputs: dict, **_kw) -> str:
    try:
        source = inputs.get("source", "user_explicit")
        total_len = prompt_assembler.append_memory(inputs["entry"], source=source)
        if _event_bus:
            _event_bus.emit("memory.appended", entry=inputs["entry"][:200])
        return json.dumps({"status": "ok", "total_chars": total_len})
    except Exception as exc:  # noqa: BLE001
        logger.exception("append_memory failed")
        return json.dumps({"error": str(exc)})




def _tool_add_skill(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.add_skill(
            inputs["skill_name"],
            inputs["documentation"],
            summary=inputs.get("summary"),
        )
        if _event_bus:
            _event_bus.emit("skill.added", skill_name=inputs["skill_name"])
        try:
            from wintermute.workers import skill_stats
            skill_stats.record_skill_written(inputs["skill_name"])
        except Exception:
            pass
        return json.dumps({"status": "ok", "skill": inputs["skill_name"]})
    except Exception as exc:  # noqa: BLE001
        logger.exception("add_skill failed")
        return json.dumps({"error": str(exc)})


def _tool_execute_shell(inputs: dict, **_kw) -> str:
    command = inputs["command"]
    timeout = int(inputs.get("timeout", 30))
    logger.info("execute_shell: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps({
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("execute_shell failed")
        return json.dumps({"error": str(exc)})


def _tool_read_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        result = json.dumps({"content": path.read_text(encoding="utf-8")})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except OSError as exc:
        return json.dumps({"error": str(exc)})
    # Track skill reads for skill_stats.
    try:
        if path.parent.name == "skills" and path.suffix == ".md" and "data/skills" in str(path):
            from wintermute.workers import skill_stats
            skill_stats.record_read(path.stem)
    except Exception:
        pass
    return result


def _tool_write_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inputs["content"], encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path)})
    except OSError as exc:
        return json.dumps({"error": str(exc)})




# ---------------------------------------------------------------------------
# HTML-to-text helper (stdlib only, no extra dependencies)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that strips tags and scripts."""

    _IGNORE_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
        elif tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        import re
        text = "".join(self._pieces)
        # Collapse runs of whitespace but preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Research tool implementations
# ---------------------------------------------------------------------------

def _tool_search_web(inputs: dict, **_kw) -> str:
    import urllib.parse
    query = inputs["query"]
    max_results = int(inputs.get("max_results", 5))
    logger.info("search_web: %s", query)

    # --- Try SearXNG first ---
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "categories": "general"})
        req = Request(f"{SEARXNG_URL}/search?{params}", headers={"User-Agent": "wintermute/0.1"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
            for item in data.get("results", [])[:max_results]
        ]
        return json.dumps({"query": query, "source": "searxng", "results": results, "count": len(results)})
    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        if not any(s in reason for s in ("Connection refused", "No route to host", "timed out")):
            return json.dumps({"error": f"SearXNG request failed: {reason}"})
        logger.warning("SearXNG unreachable (%s), falling back to curl", reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SearXNG error (%s), falling back to curl", exc)

    # --- Fallback: DuckDuckGo Instant Answer API via curl (no auth required) ---
    try:
        safe_q = urllib.parse.quote_plus(query)
        proc = subprocess.run(
            f'curl -s --max-time 15 -A "wintermute/0.1" '
            f'"https://api.duckduckgo.com/?q={safe_q}&format=json&no_html=1&skip_disambig=1"',
            shell=True, capture_output=True, text=True, timeout=20,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"curl exited {proc.returncode}: {proc.stderr[:200]}")
        data = json.loads(proc.stdout)
        results = []
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({"title": data.get("Heading", query), "url": data["AbstractURL"], "snippet": data["AbstractText"]})
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if "FirstURL" in topic:
                results.append({"title": topic.get("Text", "")[:80], "url": topic["FirstURL"], "snippet": topic.get("Text", "")})
            elif "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    if "FirstURL" in sub:
                        results.append({"title": sub.get("Text", "")[:80], "url": sub["FirstURL"], "snippet": sub.get("Text", "")})
        return json.dumps({
            "query": query, "source": "duckduckgo_fallback",
            "warning": "SearXNG unavailable. Start it with: cd ~/searxng-test && ./start-searxng.sh",
            "results": results[:max_results], "count": len(results[:max_results]),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_web fallback failed")
        return json.dumps({"error": f"Both SearXNG and curl fallback failed: {exc}"})


def _tool_fetch_url(inputs: dict, **_kw) -> str:
    url = inputs["url"]
    max_chars = int(inputs.get("max_chars", 20000))
    logger.info("fetch_url: %s", url)

    try:
        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; wintermute/0.1; "
                "+https://github.com/wintermute)"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
        })
        with urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

            # Detect encoding from Content-Type header, fall back to utf-8
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            body = raw.decode(charset, errors="replace")

    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        return json.dumps({"error": f"Failed to fetch URL: {reason}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_url failed")
        return json.dumps({"error": str(exc)})

    # Strip HTML if it looks like an HTML document.
    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        body = _html_to_text(body)

    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"

    return json.dumps({
        "url": url,
        "content_type": content_type,
        "length": len(body),
        "content": body,
    })


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "spawn_sub_session":  _tool_spawn_sub_session,
    "task":               _tool_task,
    "append_memory":      _tool_append_memory,
    "add_skill":          _tool_add_skill,
    "execute_shell":      _tool_execute_shell,
    "read_file":          _tool_read_file,
    "write_file":         _tool_write_file,
    "search_web":         _tool_search_web,
    "fetch_url":          _tool_fetch_url,
}


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 nesting_depth: int = 0, parent_thread_id: Optional[str] = None) -> str:
    """Execute a tool by name and return its JSON-string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    logger.debug("Executing tool '%s' with inputs: %s", name, inputs)
    t0 = time.monotonic()
    try:
        result = fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth,
                    parent_thread_id=parent_thread_id)
        error_flag = None
    except (KeyError, TypeError, ValueError) as exc:
        result = json.dumps({"error": f"Tool '{name}' called with invalid arguments: {exc}"})
        error_flag = str(exc)
        logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
    duration_ms = round((time.monotonic() - t0) * 1000, 1)
    _TOOL_CALL_LOG.append({
        "ts": time.time(),
        "tool": name,
        "inputs": inputs,
        "thread_id": thread_id or "unknown",
        "result_preview": result[:300] if result else "",
        "duration_ms": duration_ms,
        "error": error_flag,
    })
    return result
