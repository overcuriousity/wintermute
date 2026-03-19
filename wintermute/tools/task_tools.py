"""Task management tool implementations."""

import json
import logging
from typing import Any, Optional

from wintermute.core.tool_deps import ToolDeps
from wintermute.infra import database

logger = logging.getLogger(__name__)


_EXECUTION_MODES = {"reminder", "autonomous_notify", "autonomous_silent"}


def _resolve_execution_mode(schedule_type: Optional[str], ai_prompt: Optional[str],
                            execution_mode: Optional[str], background: bool,
                            background_provided: bool = False) -> tuple[Optional[str], bool]:
    """Resolve explicit/legacy execution semantics for scheduled tasks."""
    mode = (execution_mode or "").strip() or None
    if mode is not None and mode not in _EXECUTION_MODES:
        raise ValueError(
            "execution_mode must be one of: reminder, autonomous_notify, autonomous_silent"
        )

    if not schedule_type:
        if mode is not None:
            raise ValueError("execution_mode is only valid for scheduled tasks")
        return None, bool(background)

    if mode == "reminder":
        if ai_prompt:
            raise ValueError("ai_prompt is not allowed when execution_mode is reminder")
        return mode, False

    if mode in {"autonomous_notify", "autonomous_silent"}:
        if not ai_prompt:
            raise ValueError(
                f"ai_prompt is required when execution_mode is {mode}"
            )
        return mode, True

    # Backward compatibility for existing callers without execution_mode:
    # - ai_prompt + background=true  -> autonomous_notify
    # - ai_prompt + background=false -> autonomous_silent
    # - ai_prompt + background omitted -> autonomous_notify (legacy default)
    # - no ai_prompt                -> reminder
    if ai_prompt:
        if background_provided:
            inferred = "autonomous_notify" if background else "autonomous_silent"
        else:
            inferred = "autonomous_notify"
        return inferred, True

    return "reminder", False


def _describe_schedule(inputs: dict) -> str:
    """Build a human-readable schedule string from structured inputs."""
    t = inputs.get("schedule_type", "once")
    if t == "once":
        return f"once at {inputs.get('at', '?')}"
    if t == "daily":
        return f"daily at {inputs.get('at', '?')}"
    if t == "weekly":
        return f"weekly on {inputs.get('day_of_week', '?')} at {inputs.get('at', '?')}"
    if t == "monthly":
        return f"monthly on day {inputs.get('day_of_month', '?')} at {inputs.get('at', '?')}"
    if t == "interval":
        secs = inputs.get("interval_seconds", "?")
        desc = f"every {secs}s"
        ws, we = inputs.get("window_start"), inputs.get("window_end")
        if ws and we:
            desc += f" from {ws} to {we}"
        return desc
    return str(inputs)


def _task_add(inputs: dict, effective_scope: Optional[str],
              tool_deps: Optional[ToolDeps] = None) -> str:
    content = inputs.get("content")
    if not content:
        return json.dumps({"error": "content is required for add action"})
    add_thread = inputs.get("thread_id") or effective_scope
    schedule_type = inputs.get("schedule_type")
    ai_prompt = (inputs.get("ai_prompt") or "").strip() or None
    execution_mode = (inputs.get("execution_mode") or "").strip() or None
    background_provided = "background" in inputs
    background = bool(inputs.get("background", False))

    try:
        execution_mode, background = _resolve_execution_mode(
            schedule_type=schedule_type,
            ai_prompt=ai_prompt,
            execution_mode=execution_mode,
            background=background,
            background_provided=background_provided,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    schedule_config = None
    schedule_desc = None
    if schedule_type:
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
        execution_mode=execution_mode,
    )

    deps = tool_deps or ToolDeps()
    if schedule_type and deps.task_scheduler is not None:
        deps.task_scheduler.ensure_job(
            task_id, json.loads(schedule_config),
            ai_prompt, add_thread, background, execution_mode,
        )
        database.update_task(task_id, apscheduler_job_id=task_id)

    if deps.event_bus:
        deps.event_bus.emit("task.created", task_id=task_id,
                        content=content[:200],
                        schedule_type=schedule_type)
    result = {"status": "ok", "task_id": task_id}
    if schedule_desc:
        result["schedule"] = schedule_desc
    if schedule_type:
        result["execution_mode"] = execution_mode
    return json.dumps(result)


def _task_complete(inputs: dict, effective_scope: Optional[str],
                   tool_deps: Optional[ToolDeps] = None) -> str:
    deps = tool_deps or ToolDeps()
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for complete action"})
    reason = (inputs.get("reason") or "").strip()
    if not reason:
        return json.dumps({"error": "reason is required for complete action — explain why this task is finished"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and deps.task_scheduler:
        deps.task_scheduler.remove_job(task_id)
    ok = database.complete_task(task_id, reason=reason, thread_id=effective_scope)
    if ok and deps.event_bus:
        deps.event_bus.emit("task.completed", task_id=task_id, reason=reason[:200])
    return json.dumps({"status": "ok" if ok else "not_found", "reason": reason})


def _task_pause(inputs: dict, effective_scope: Optional[str],
                tool_deps: Optional[ToolDeps] = None) -> str:
    deps = tool_deps or ToolDeps()
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for pause action"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and deps.task_scheduler:
        deps.task_scheduler.remove_job(task_id)
    ok = database.pause_task(task_id)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_resume(inputs: dict, effective_scope: Optional[str],
                 tool_deps: Optional[ToolDeps] = None) -> str:
    deps = tool_deps or ToolDeps()
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for resume action"})
    ok = database.resume_task(task_id)
    if ok:
        task = database.get_task(task_id)
        if task and task.get("schedule_config") and deps.task_scheduler:
            sched = json.loads(task["schedule_config"])
            deps.task_scheduler.ensure_job(
                task_id, sched,
                task.get("ai_prompt"), task.get("thread_id"),
                bool(task.get("background")),
                task.get("execution_mode"),
            )
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_delete(inputs: dict, effective_scope: Optional[str],
                 tool_deps: Optional[ToolDeps] = None) -> str:
    deps = tool_deps or ToolDeps()
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for delete action"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and deps.task_scheduler:
        deps.task_scheduler.remove_job(task_id)
    ok = database.delete_task(task_id)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_update(inputs: dict, effective_scope: Optional[str],
                 tool_deps: Optional[ToolDeps] = None) -> str:
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


def _task_list(inputs: dict, effective_scope: Optional[str],
               tool_deps: Optional[ToolDeps] = None) -> str:
    status = inputs.get("status", "active")
    items = database.list_tasks(status, thread_id=effective_scope)
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
        if it.get("execution_mode"):
            entry["execution_mode"] = it["execution_mode"]
        if it.get("last_run_at"):
            entry["last_run_at"] = it["last_run_at"]
            entry["run_count"] = it.get("run_count", 0)
        if it.get("last_result_summary"):
            entry["last_result"] = it["last_result_summary"][:200]
        formatted.append(entry)
    return json.dumps({"tasks": formatted, "count": len(formatted)})


TASK_ACTIONS: dict[str, Any] = {
    "add":      _task_add,
    "complete": _task_complete,
    "pause":    _task_pause,
    "resume":   _task_resume,
    "delete":   _task_delete,
    "update":   _task_update,
    "list":     _task_list,
}


def tool_task(inputs: dict, thread_id: Optional[str] = None,
              parent_thread_id: Optional[str] = None,
              tool_deps: Optional[ToolDeps] = None, **_kw) -> str:
    """Unified task tool — handles add/update/complete/pause/resume/delete/list."""
    effective_scope = parent_thread_id or thread_id
    try:
        action = inputs.get("action", "list")
        handler = TASK_ACTIONS.get(action)
        if handler is None:
            return json.dumps({"error": f"Unknown action: {action}"})
        return handler(inputs, effective_scope, tool_deps=tool_deps)
    except Exception as exc:  # noqa: BLE001
        logger.exception("task tool failed")
        return json.dumps({"error": str(exc)})
