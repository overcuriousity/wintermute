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


_SCHEDULE_KEYS = ("schedule_type", "at", "day_of_week", "day_of_month",
                  "interval_seconds", "window_start", "window_end")

_SCHEDULE_TYPES = ("once", "daily", "weekly", "monthly", "interval")

_SCHEDULE_REQUIRED: dict[str, tuple[str, ...]] = {
    "once": ("at",),
    "weekly": ("at",),
    "monthly": ("at",),
    "interval": ("interval_seconds",),
}

# Defaults the scheduler applies in _parse_trigger. Mirrored into the stored
# config so schedule_desc describes the trigger that actually runs.
_SCHEDULE_DEFAULTS: dict[str, dict] = {
    "daily": {"at": "09:00"},
    "weekly": {"day_of_week": "mon"},
    "monthly": {"day_of_month": 1},
}


def _build_schedule(inputs: dict) -> tuple[dict, str]:
    """Validate schedule inputs and return ``(schedule_config, description)``.

    Raises ValueError for an unknown schedule_type or missing required fields.
    """
    schedule_type = inputs.get("schedule_type")
    if schedule_type not in _SCHEDULE_TYPES:
        raise ValueError(
            "schedule_type must be one of: " + ", ".join(_SCHEDULE_TYPES)
        )
    sched = {k: inputs[k] for k in _SCHEDULE_KEYS if k in inputs}
    missing = [f for f in _SCHEDULE_REQUIRED.get(schedule_type, ())
               if sched.get(f) in (None, "")]
    if missing:
        raise ValueError(
            f"Missing required field(s) for {schedule_type!r} schedule: "
            + ", ".join(sorted(missing))
        )
    # Fields the scheduler parses with int() — validate them here so a bad
    # value is rejected before the task row is inserted.
    for key in ("interval_seconds", "day_of_month"):
        if sched.get(key) in (None, ""):
            continue
        try:
            sched[key] = int(sched[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer, got {sched[key]!r}") from None
    if schedule_type == "interval" and sched["interval_seconds"] <= 0:
        raise ValueError("interval_seconds must be a positive integer")
    for key, value in _SCHEDULE_DEFAULTS.get(schedule_type, {}).items():
        sched.setdefault(key, value)
    return sched, _describe_schedule(sched)


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
        try:
            sched_inputs, schedule_desc = _build_schedule(inputs)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        schedule_config = json.dumps(sched_inputs)

    task_id = database.add_task(
        content=content,
        thread_id=add_thread,
        schedule_type=schedule_type,
        schedule_desc=schedule_desc,
        schedule_config=schedule_config,
        ai_prompt=ai_prompt,
        background=background,
        execution_mode=execution_mode,
    )

    deps = tool_deps or ToolDeps()
    scheduled = False
    if schedule_type and deps.task_scheduler is not None:
        try:
            deps.task_scheduler.ensure_job(
                task_id, json.loads(schedule_config),
                ai_prompt, add_thread, background, execution_mode,
            )
            database.update_task(task_id, apscheduler_job_id=task_id)
            scheduled = True
        except Exception:
            logger.warning("Could not schedule APScheduler job for new task %s",
                           task_id, exc_info=True)

    if deps.event_bus:
        deps.event_bus.emit("task.created", task_id=task_id,
                        content=content[:200],
                        schedule_type=schedule_type)
    result = {"status": "ok", "task_id": task_id}
    if schedule_desc:
        result["schedule"] = schedule_desc
    if schedule_type:
        result["execution_mode"] = execution_mode
        result["scheduled"] = scheduled
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
    # Mutate the DB first — removing the scheduler job before a failed DB
    # write would leave an active task that silently never fires.
    ok = database.complete_task(task_id, reason=reason, thread_id=effective_scope)
    if ok and task and task.get("apscheduler_job_id") and deps.task_scheduler:
        try:
            deps.task_scheduler.remove_job(task["apscheduler_job_id"])
        except Exception:
            logger.warning("Could not remove APScheduler job for completed task %s",
                           task_id, exc_info=True)
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
    ok = database.pause_task(task_id)
    if ok and task and task.get("apscheduler_job_id") and deps.task_scheduler:
        try:
            deps.task_scheduler.remove_job(task["apscheduler_job_id"])
        except Exception:
            logger.warning("Could not remove APScheduler job for paused task %s",
                           task_id, exc_info=True)
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
    ok = database.delete_task(task_id)
    if ok and task and task.get("apscheduler_job_id") and deps.task_scheduler:
        try:
            deps.task_scheduler.remove_job(task["apscheduler_job_id"])
        except Exception:
            logger.warning("Could not remove APScheduler job for deleted task %s",
                           task_id, exc_info=True)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_update(inputs: dict, effective_scope: Optional[str],
                 tool_deps: Optional[ToolDeps] = None) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for update action"})
    kwargs = {}
    if "content" in inputs:
        kwargs["content"] = inputs["content"]
    task = database.get_task(task_id)
    # Web-created tasks have no thread scope (thread_id NULL) and are editable
    # from any scope; scoped tasks are only editable from their own thread.
    task_thread = task.get("thread_id") if task else None
    if not task or (task_thread is not None and task_thread != effective_scope):
        return json.dumps({"status": "not_found"})
    new_ai_prompt = (task.get("ai_prompt") or "").strip() or None
    new_execution_mode = (task.get("execution_mode") or "").strip() or None
    new_background = bool(task.get("background"))
    if "ai_prompt" in inputs or "execution_mode" in inputs:
        if "ai_prompt" in inputs:
            raw_ai_prompt = (inputs.get("ai_prompt") or "").strip()
            new_ai_prompt = raw_ai_prompt or None
            # Clearing ai_prompt converts an autonomous task back into a plain
            # reminder — reset the stored mode unless explicitly overridden.
            if new_ai_prompt is None and "execution_mode" not in inputs:
                new_execution_mode = None
        if "execution_mode" in inputs:
            raw_execution_mode = (inputs.get("execution_mode") or "").strip()
            new_execution_mode = raw_execution_mode or None
        try:
            resolved_mode, new_background = _resolve_execution_mode(
                schedule_type=task.get("schedule_type"),
                ai_prompt=new_ai_prompt,
                execution_mode=new_execution_mode,
                background=bool(task.get("background")),
                background_provided=True,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        new_execution_mode = resolved_mode
        if "ai_prompt" in inputs:
            kwargs["ai_prompt"] = new_ai_prompt
        # Persist the canonical resolved mode (as the web endpoint does), so
        # the stored mode and the delivery semantics cannot diverge.
        if task.get("schedule_type"):
            kwargs["execution_mode"] = resolved_mode
        kwargs["background"] = int(new_background)
    ok = database.update_task(task_id, thread_id=task_thread, **kwargs)
    if not ok:
        return json.dumps({"status": "not_found"})

    # The APScheduler job caches content/ai_prompt/background/execution_mode in
    # its persisted kwargs, so re-register it after an edit.
    deps = tool_deps or ToolDeps()
    if (task.get("schedule_config") and task.get("status") == "active"
            and deps.task_scheduler is not None):
        try:
            deps.task_scheduler.ensure_job(
                task_id, json.loads(task["schedule_config"]),
                new_ai_prompt, task.get("thread_id"),
                new_background, new_execution_mode,
            )
        except Exception:
            logger.warning("Could not re-schedule APScheduler job for updated task %s",
                           task_id, exc_info=True)
    return json.dumps({"status": "ok"})


def _task_list(inputs: dict, effective_scope: Optional[str],
               tool_deps: Optional[ToolDeps] = None) -> str:
    status = inputs.get("status", "active")
    items = database.list_tasks(status, thread_id=effective_scope)
    formatted = []
    for it in items:
        entry = {
            "id": it["id"],
            "content": it["content"],
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
