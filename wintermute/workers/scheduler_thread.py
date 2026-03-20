"""
Task Schedule Engine

Uses APScheduler with a persistent SQLite job store so scheduled tasks survive
restarts.  At startup it detects missed executions and runs them immediately.

The ``ensure_job``, ``remove_job``, and ``list_jobs`` functions are injected
into the tools module so the unified ``task`` tool can manage schedules.

APScheduler's SQLite store is the single source of truth for active jobs.
Inline execution tracking (last_run_at, run_count, last_result_summary) is
stored in the tasks table in conversation.db.

Natural-language time parsing is handled by a simple heuristic + dateutil.
"""

import asyncio
import json
import logging
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dateutil import parser as dateutil_parser

from wintermute.infra import database
from wintermute.infra.paths import DATA_DIR, SCHEDULER_DB
from wintermute import tools as tool_module

if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager
    from wintermute.core.session_manager import SessionManager
    from wintermute.infra.event_bus import EventBus

logger = logging.getLogger(__name__)

# Module-level reference so the job function below can be pickled by APScheduler.
# Set by TaskScheduler.start().
_instance: Optional["TaskScheduler"] = None


async def _fire_task_job(task_id: str, message: str, ai_prompt: Optional[str],
                          thread_id: Optional[str] = None,
                          background: bool = False,
                          execution_mode: Optional[str] = None,
                          **_extra) -> None:
    """
    Module-level coroutine used as the APScheduler job callable.
    Must be at module level so pickle can serialize it by reference.

    **_extra absorbs metadata kwargs stored alongside the job.
    """
    if _instance is not None:
        await _instance._fire_task(
            task_id, message, ai_prompt, thread_id, background, execution_mode
        )


# Backward-compat aliases: jobs persisted in scheduler.db before the
# routine→task rename reference these names. APScheduler resolves
# callables by dotted name at load time, so the aliases must exist.
async def _fire_routine_job(job_id: str, message: str, ai_prompt: Optional[str],
                             thread_id: Optional[str] = None,
                             background: bool = False,
                             execution_mode: Optional[str] = None,
                             **_extra) -> None:
    if _instance is not None:
        await _instance._fire_task(
            job_id, message, ai_prompt, thread_id, background, execution_mode
        )

_fire_reminder_job = _fire_routine_job


async def _check_predictions_job(**_extra) -> None:
    """Module-level coroutine for the hourly prediction check.

    Must be at module level so APScheduler's SQLAlchemy pickle can
    serialize it by reference.
    """
    if _instance is not None:
        await _instance._check_predictions()


async def _check_session_timeouts_job(**_extra) -> None:
    """Module-level coroutine for periodic session timeout enforcement."""
    if _instance is not None:
        await _instance._enforce_session_timeouts()


@dataclass
class SchedulerConfig:
    timezone: str = "UTC"
    prediction_proactive_scheduling: bool = True
    prediction_proactive_cooldown_hours: int = 4
    proactive_target_thread_id: str = "default"


class TaskScheduler:
    """Wraps APScheduler and manages task schedules.

    APScheduler's persistent SQLite store is the single source of truth for
    active jobs.  Metadata is stored in the job's kwargs so it can be
    retrieved without a second store.
    """

    def __init__(self, config: SchedulerConfig, broadcast_fn, llm_enqueue_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None,
                 event_bus: "Optional[EventBus]" = None) -> None:
        self._cfg = config
        self._broadcast = broadcast_fn
        self._llm_enqueue = llm_enqueue_fn
        self._sub_sessions = sub_session_manager
        self._event_bus = event_bus
        self._event_bus_subs: list[str] = []
        self._scheduler: Optional[AsyncIOScheduler] = None
        # Track last proactive fire time per prediction ID to enforce cooldowns.
        self._prediction_last_fired: dict[str, float] = {}
        self._session_manager: Optional["SessionManager"] = None

    def set_session_manager(self, mgr: "SessionManager") -> None:
        """Wire a session manager for timeout enforcement."""
        self._session_manager = mgr

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        jobstores = {
            "default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB.as_posix()}")
        }
        executors = {"default": AsyncIOExecutor()}
        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            timezone=self._cfg.timezone,
        )
        self._scheduler.start()

        global _instance
        _instance = self

        self._recover_missed()

        # Subscribe to task.created events to schedule new jobs immediately.
        if self._event_bus:
            sub_id = self._event_bus.subscribe("task.created", self._on_task_created)
            self._event_bus_subs.append(sub_id)

        # Restore prediction cooldowns from interaction_log.
        # Run synchronously on the loop thread to avoid cross-thread races
        # on self._prediction_last_fired (read/written by async jobs too).
        try:
            self._restore_prediction_cooldowns()
        except Exception:
            logger.exception("Error restoring prediction cooldowns")

        logger.info("[scheduler] started (timezone=%s)", self._cfg.timezone)

        # Schedule periodic prediction-based proactive checks.
        if self._cfg.prediction_proactive_scheduling:
            self._scheduler.add_job(
                _check_predictions_job,
                trigger=IntervalTrigger(hours=1),
                id="_prediction_check",
                replace_existing=True,
                misfire_grace_time=3600,
            )
            logger.info("[scheduler] Prediction proactive check scheduled (hourly)")

        # Schedule periodic session timeout enforcement.
        if self._session_manager is not None:
            self._scheduler.add_job(
                _check_session_timeouts_job,
                trigger=IntervalTrigger(minutes=5),
                id="_session_timeout_check",
                replace_existing=True,
                misfire_grace_time=600,
            )
            logger.info("[scheduler] Session timeout check scheduled (every 5 min)")

    async def _on_task_created(self, event) -> None:
        """React to task.created events — schedule the job if it has schedule_config."""
        task_id = event.data.get("task_id")
        if not task_id:
            return
        try:
            task = await database.async_call(database.get_task, task_id)
            if not task:
                return
            # Skip if already scheduled (task_tools._task_add calls ensure_job before emitting).
            if task.get("apscheduler_job_id"):
                logger.info("[scheduler] Task %s already has apscheduler_job_id — skipping", task_id)
                return
            raw_config = task.get("schedule_config")
            if not raw_config:
                logger.info("[scheduler] New task %s has no schedule_config — skipping job creation", task_id)
                return
            schedule_config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
            ai_prompt = task.get("ai_prompt")
            thread_id = task.get("thread_id")
            background = bool(task.get("background"))
            execution_mode = task.get("execution_mode")
            self.ensure_job(task_id, schedule_config, ai_prompt=ai_prompt,
                            thread_id=thread_id, background=background,
                            execution_mode=execution_mode)
            # Persist job ID so pause/complete/delete can manage it later.
            await database.async_call(database.update_task, task_id, apscheduler_job_id=task_id)
            logger.info("[scheduler] Scheduled job for new task %s", task_id)
        except Exception:
            logger.exception("[scheduler] Failed to process task.created for %s", task_id)

    def _restore_prediction_cooldowns(self) -> None:
        """Pre-populate _prediction_last_fired from recent interaction_log entries."""
        cooldown_seconds = self._cfg.prediction_proactive_cooldown_hours * 3600
        cutoff = _time.time() - cooldown_seconds
        try:
            entries = database.get_interaction_log(limit=50, action_filter="prediction_fired")
            for e in entries:
                try:
                    ts = float(e.get("timestamp") or 0)
                except (TypeError, ValueError):
                    continue
                if ts < cutoff:
                    continue
                pred_id = (e.get("input") or "").strip()
                if pred_id:
                    existing = self._prediction_last_fired.get(pred_id, 0)
                    if ts > existing:
                        self._prediction_last_fired[pred_id] = ts
            if self._prediction_last_fired:
                logger.info("[scheduler] Restored %d prediction cooldown(s) from DB",
                            len(self._prediction_last_fired))
        except Exception:
            logger.debug("[scheduler] Failed to restore prediction cooldowns", exc_info=True)

    def stop(self) -> None:
        if self._event_bus:
            for sub_id in self._event_bus_subs:
                self._event_bus.unsubscribe(sub_id)
            self._event_bus_subs.clear()
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[scheduler] stopped")

    # ------------------------------------------------------------------
    # Job management (called by tools.py _tool_task)
    # ------------------------------------------------------------------

    def ensure_job(self, task_id: str, schedule_config: dict,
                   ai_prompt: Optional[str] = None,
                   thread_id: Optional[str] = None,
                   background: bool = False,
                   execution_mode: Optional[str] = None) -> None:
        """Create or update an APScheduler job for a task."""
        trigger = self._parse_trigger(schedule_config)
        message = database.get_task(task_id) or {}
        content = message.get("content", task_id)

        self._scheduler.add_job(
            _fire_task_job,
            trigger=trigger,
            id=task_id,
            kwargs={
                "task_id": task_id,
                "message": content,
                "ai_prompt": ai_prompt,
                "thread_id": thread_id,
                "background": background,
                "execution_mode": execution_mode,
                "schedule_type": schedule_config.get("schedule_type"),
                "schedule": tool_module._describe_schedule(schedule_config),
                "created": datetime.now(timezone.utc).isoformat(),
            },
            replace_existing=True,
            misfire_grace_time=3600,
        )

        next_run = self._scheduler.get_job(task_id)
        if next_run:
            logger.info("Task job scheduled: %s at %s (thread=%s)",
                        task_id, next_run.next_run_time, thread_id)

    def remove_job(self, task_id: str) -> None:
        """Remove an APScheduler job for a task."""
        if self._scheduler.get_job(task_id) is not None:
            self._scheduler.remove_job(task_id)
            logger.info("Task job removed: %s", task_id)

    def list_jobs(self) -> list[dict]:
        """Return serialisable info about all APScheduler jobs."""
        result = []
        for job in self._scheduler.get_jobs():
            result.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
                "kwargs": {k: str(v)[:300] for k, v in (job.kwargs or {}).items()},
            })
        return result

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _fire_task(self, task_id: str, message: str, ai_prompt: Optional[str],
                          thread_id: Optional[str] = None,
                          background: bool = False,
                          execution_mode: Optional[str] = None) -> None:
        raw_mode = execution_mode
        mode = (execution_mode or "").strip() or None
        if mode not in {"reminder", "autonomous_notify", "autonomous_silent"}:
            # An explicit but unexpected execution_mode was provided; log and fall back.
            if raw_mode is not None:
                logger.warning(
                    "Unexpected execution_mode '%s' for task %s; falling back to inferred "
                    "behavior (ai_prompt=%s, background=%s)",
                    raw_mode,
                    task_id,
                    bool(ai_prompt),
                    background,
                )
            if ai_prompt:
                mode = "autonomous_notify" if background else "autonomous_silent"
            else:
                mode = "reminder"

        logger.info("Firing task %s (thread=%s, background=%s, execution_mode=%s)",
                    task_id, thread_id, background, mode)
        if self._event_bus:
            self._event_bus.emit("task.fired", task_id=task_id, thread_id=thread_id)

        executed = False
        attempted = False
        requested_mode = mode
        delivery = f"execution_mode={mode}, delivery=skipped"

        try:
            if mode == "reminder":
                if thread_id:
                    attempted = True
                    await self._broadcast(f"⏰ Task: {message}", thread_id)
                    executed = True
                    delivery = "execution_mode=reminder, delivery=chat"
                else:
                    delivery = "execution_mode=reminder, delivery=not_delivered_no_thread"
                    logger.warning(
                        "Task %s is reminder mode but has no thread_id — "
                        "message was NOT delivered: %s", task_id, message
                    )
            else:
                if not ai_prompt:
                    delivery = f"execution_mode={mode}, delivery=not_executed_missing_ai_prompt"
                    logger.warning(
                        "Task %s execution_mode=%s requires ai_prompt but none provided — skipping",
                        task_id, mode,
                    )
                elif self._sub_sessions is not None:
                    attempted = True
                    pred_ctx = await self._get_prediction_context()
                    if ai_prompt == message:
                        objective = f"[TASK {task_id}] {ai_prompt}\n\n"
                    else:
                        objective = (
                            f"[TASK {task_id}] {ai_prompt}\n\n"
                            f"(Task: {message})\n\n"
                        )
                    if pred_ctx:
                        capped = pred_ctx[:800]
                        objective += (
                            f"## Relevant predictions about the user\n"
                            f"{capped}\n\n"
                        )
                    # Inject run history so recurring tasks can build on prior runs.
                    task_record = await database.async_call(database.get_task, task_id)
                    if task_record:
                        run_count = task_record.get("run_count") or 0
                        last_run_at = task_record.get("last_run_at")
                        last_summary = task_record.get("last_result_summary")
                        if run_count > 0 and last_run_at:
                            try:
                                last_dt = datetime.fromtimestamp(last_run_at, tz=timezone.utc)
                                last_run_fmt = last_dt.strftime("%Y-%m-%d %H:%M UTC")
                            except (OSError, ValueError):
                                last_run_fmt = "unknown"
                            objective += f"## Prior run context\nRun #{run_count + 1}. Last run: {last_run_fmt}.\n"
                            if last_summary:
                                objective += f"Last result: {last_summary[:1500]}\n\n"
                            else:
                                objective += "\n"

                    objective += (
                        "If you have nothing actionable to report, "
                        "respond with exactly: [NO_ACTION]"
                    )
                    parent_thread_id = None
                    if mode == "autonomous_notify":
                        if thread_id is None:
                            logger.warning(
                                "Task %s execution_mode=autonomous_notify but no thread_id provided — "
                                "running as autonomous_silent (no notifications will be delivered)",
                                task_id,
                            )
                            mode = "autonomous_silent"
                        else:
                            parent_thread_id = thread_id
                    self._sub_sessions.spawn(
                        objective=objective,
                        parent_thread_id=parent_thread_id,
                        system_prompt_mode="full",
                        task_id=task_id,
                    )
                    executed = True
                    if mode == "autonomous_notify":
                        delivery = "execution_mode=autonomous_notify, delivery=chat"
                    elif requested_mode == "autonomous_notify":
                        delivery = "execution_mode=autonomous_silent, delivery=silent_downgraded_missing_thread"
                    else:
                        delivery = "execution_mode=autonomous_silent, delivery=silent"
                else:
                    delivery = f"execution_mode={mode}, delivery=not_executed_no_sub_session_manager"
                    logger.warning(
                        "Task %s execution_mode=%s has ai_prompt but SubSessionManager "
                        "is not available — skipping", task_id, mode
                    )

            if executed:
                try:
                    summary = f"executed (execution_mode={mode})"
                    database.record_task_run(task_id, summary=summary)
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to record task run for %s", task_id)

            try:
                mode_info = (
                    f"requested_mode={requested_mode}, execution_mode={mode}"
                    if requested_mode != mode
                    else f"execution_mode={mode}"
                )
                status = "ok" if executed else "skipped"
                await database.async_call(
                    database.save_interaction_log,
                    _time.time(),
                    "task_fired",
                    thread_id or "system:scheduler",
                    "scheduler",
                    f"task_id={task_id}, {mode_info}, content={message[:200]}",
                    delivery,
                    status,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to log task firing for %s", task_id)

            if self._scheduler.get_job(task_id) is None:
                logger.info("One-time task %s completed", task_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Task %s failed", task_id)
            if attempted:
                try:
                    database.record_task_run(task_id, summary=f"failed: {exc}")
                except Exception:  # noqa: BLE001
                    pass
            try:
                # In autonomous_silent mode, suppress chat broadcasts even on failure.
                if thread_id and mode != "autonomous_silent":
                    await self._broadcast(f"❌ Task {task_id} failed: {exc}", thread_id)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Prediction-based proactive scheduling
    # ------------------------------------------------------------------

    async def _check_predictions(self) -> None:
        """Check temporal predictions and spawn proactive sub-sessions.

        Runs hourly.  For each ``[prediction:temporal]`` entry, parses the
        predicted active time window.  If the current time falls within a
        predicted window and the cooldown has elapsed, spawns a full
        sub-session to check for anything proactive to surface.
        """
        if self._sub_sessions is None:
            return

        from wintermute.infra import memory_store

        try:
            predictions = await asyncio.to_thread(
                memory_store.get_by_source, "dreaming_prediction", 50, bump_access=False
            )
        except Exception:
            logger.debug("[scheduler] Failed to fetch predictions", exc_info=True)
            return

        if not predictions:
            return

        now = _time.time()
        cooldown_seconds = self._cfg.prediction_proactive_cooldown_hours * 3600

        # Use UTC to match dreaming phase which generates predictions in UTC.
        current = datetime.now(timezone.utc)
        current_hour = current.hour
        current_day = current.strftime("%A").lower()

        for pred in predictions:
            raw_pred_id = pred.get("id")
            pred_id = str(raw_pred_id).strip() if raw_pred_id is not None else ""
            if not pred_id:
                # Skip predictions without a stable ID to avoid
                # collapsing cooldown tracking under an empty key.
                continue
            text = pred.get("text", "")
            text_lower = text.lower()

            # Only process temporal predictions.
            if "[prediction:temporal]" not in text_lower and "most active" not in text_lower:
                continue

            # Enforce cooldown per prediction.
            last_fired = self._prediction_last_fired.get(pred_id, 0)
            if now - last_fired < cooldown_seconds:
                continue

            # Try structured ||key=val|| suffix first, fall back to regex.
            structured_hours = re.search(r'\|\|hours=(\d{1,2})-(\d{1,2})\|\|', text, re.IGNORECASE)
            structured_days = re.search(r'\|\|days=([\w,]+)\|\|', text, re.IGNORECASE)

            in_time_window = False
            used_structured = bool(structured_hours or structured_days)

            if used_structured:
                # Default: in window unless constrained out by hours/days.
                in_time_window = True

                if structured_hours:
                    try:
                        sh, eh = int(structured_hours.group(1)), int(structured_hours.group(2))
                        if sh <= 23 and eh <= 23:
                            if sh <= eh:
                                in_time_window = sh <= current_hour <= eh
                            else:
                                in_time_window = current_hour >= sh or current_hour <= eh
                        else:
                            in_time_window = False
                    except (ValueError, TypeError):
                        in_time_window = False

                if in_time_window and structured_days:
                    days_list = {d.strip().lower()[:3] for d in structured_days.group(1).split(",")}
                    current_day_abbr = current_day[:3]
                    if current_day_abbr not in days_list:
                        in_time_window = False

            if not used_structured:
                # Legacy regex parsing for predictions without structured suffix.
                hour_matches = re.findall(
                    r'(\d{1,2})(?::\d{2})?\s*(am|pm)?\s*(?:-|to)\s*(\d{1,2})(?::\d{2})?\s*(am|pm)?',
                    text_lower,
                )
                day_matches = re.findall(
                    r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?',
                    text_lower,
                )

                if hour_matches:
                    for start_h, start_ampm, end_h, end_ampm in hour_matches:
                        try:
                            sh, eh = int(start_h), int(end_h)
                            s_ap = start_ampm or end_ampm
                            e_ap = end_ampm or start_ampm
                            if s_ap == "pm" and sh < 12:
                                sh += 12
                            elif s_ap == "am" and sh == 12:
                                sh = 0
                            if e_ap == "pm" and eh < 12:
                                eh += 12
                            elif e_ap == "am" and eh == 12:
                                eh = 0
                            if sh > 23 or eh > 23:
                                continue
                            if sh <= eh:
                                in_time_window = sh <= current_hour <= eh
                            else:
                                in_time_window = current_hour >= sh or current_hour <= eh
                        except (ValueError, TypeError):
                            pass
                        if in_time_window:
                            break

                if day_matches:
                    if current_day not in {d.lower() for d in day_matches}:
                        in_time_window = False
                elif not hour_matches:
                    # No parseable time info — skip.
                    continue

            if not in_time_window:
                continue

            # Spawn a proactive sub-session.
            logger.info(
                "[scheduler] Proactive prediction trigger: %s", text[:120]
            )
            self._prediction_last_fired[pred_id] = now
            # Persist cooldown to interaction_log so it survives restarts.
            try:
                await database.async_call(
                    database.save_interaction_log,
                    now, "prediction_fired", "system:scheduler",
                    "scheduler", pred_id, text[:200], "ok",
                )
            except Exception:
                logger.debug("[scheduler] Failed to log prediction firing", exc_info=True)
            # Bump access count — this prediction was actually used.
            try:
                from wintermute.infra import memory_store
                await asyncio.to_thread(memory_store.track_access, [pred_id])
            except Exception:
                pass  # Best-effort.
            self._sub_sessions.spawn(
                objective=(
                    f"[PROACTIVE] User is predicted to be active now based on:\n"
                    f"{text}\n\n"
                    f"Check for pending tasks, recent reminders, upcoming "
                    f"deadlines, or anything useful to surface proactively.\n\n"
                    f"If you have nothing actionable to report, respond with "
                    f"exactly: [NO_ACTION]"
                ),
                system_prompt_mode="full",
                parent_thread_id=self._cfg.proactive_target_thread_id,
            )

            if self._event_bus:
                self._event_bus.emit(
                    "scheduler.proactive_fired",
                    prediction_id=pred_id,
                    prediction_text=text[:200],
                )

    # ------------------------------------------------------------------
    # Session timeout enforcement
    # ------------------------------------------------------------------

    async def _enforce_session_timeouts(self) -> None:
        """Check for expired sessions and reset them."""
        if self._session_manager is None:
            return
        expired = self._session_manager.check_session_timeouts()
        for tid in expired:
            # Re-check: last_activity may have been updated since the snapshot.
            last = self._session_manager.last_activity.get(tid)
            if last is None:
                continue  # Already cleaned up.
            resolved = self._session_manager.resolve_config(tid)
            timeout = resolved.session_timeout_minutes if resolved else None
            if timeout is None or (_time.time() - last) <= timeout * 60:
                continue
            logger.info("[scheduler] Session timeout — resetting thread %s", tid)
            try:
                await self._session_manager.reset_session(tid)
                # Clear the last-activity marker so this thread is not
                # repeatedly treated as expired on subsequent checks.
                self._session_manager.last_activity.pop(tid, None)
                # Clear any per-thread tool-call history so cross-turn CP
                # checks don't see pre-reset tool usage after a timeout reset.
                prior_tool_calls = getattr(self._session_manager, "prior_tool_calls", None)
                if isinstance(prior_tool_calls, dict):
                    prior_tool_calls.pop(tid, None)
                if self._event_bus:
                    self._event_bus.emit("session.timeout_reset", thread_id=tid)
            except Exception:
                logger.exception("[scheduler] Failed to reset timed-out session %s", tid)

    # ------------------------------------------------------------------
    # Prediction context injection for task sub-sessions
    # ------------------------------------------------------------------

    async def _get_prediction_context(self) -> str:
        """Return behavioral/preference prediction text for task context."""
        from wintermute.infra import memory_store

        lines: list[str] = []
        try:
            predictions = await asyncio.to_thread(
                memory_store.get_by_source, "dreaming_prediction", 50, bump_access=False
            )
            used_ids: list[str] = []
            for pred in predictions:
                text = pred.get("text", "")
                if "[prediction:behavioral]" in text.lower() or "[prediction:preference]" in text.lower():
                    lines.append(text.strip())
                    pred_id = pred.get("id")
                    if pred_id is not None:
                        used_ids.append(str(pred_id))
            if used_ids:
                try:
                    await asyncio.to_thread(memory_store.track_access, used_ids)
                except Exception:
                    pass  # Best-effort access bumping.
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    _MAX_MISSED_AGE_HOURS = 24  # Don't recover jobs missed more than 24h ago

    def _recover_missed(self) -> None:
        """Detect and re-fire jobs that were missed during downtime.

        APScheduler's misfire_grace_time handles some of this, but jobs whose
        fire time fell entirely within the downtime window may not re-fire.
        We explicitly check for jobs with a next_run_time in the past and
        whose associated task has an ai_prompt, then fire them immediately.
        One-time (DateTrigger) jobs that already ran are excluded because
        APScheduler removes them after execution.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self._MAX_MISSED_AGE_HOURS)
        recovered = 0

        for job in self._scheduler.get_jobs():
            if job.next_run_time is None:
                continue
            # Normalise to UTC for comparison.
            next_run = job.next_run_time
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
            else:
                next_run = next_run.astimezone(timezone.utc)

            if next_run >= now:
                logger.debug("Loaded job %s, next_run=%s (upcoming)", job.id, next_run)
                continue

            if next_run < cutoff:
                logger.info(
                    "[scheduler] Skipping stale missed job %s (next_run=%s, older than %dh)",
                    job.id, next_run, self._MAX_MISSED_AGE_HOURS,
                )
                continue

            # This job's next_run_time is in the past but within the recovery window.
            kw = job.kwargs or {}
            ai_prompt = kw.get("ai_prompt")
            task_id = kw.get("task_id", job.id)
            thread_id = kw.get("thread_id")
            background = kw.get("background", False)
            execution_mode = kw.get("execution_mode")
            message = kw.get("message", task_id)

            if not ai_prompt and not thread_id:
                logger.debug(
                    "[scheduler] Missed job %s has no ai_prompt and no thread_id — skipping",
                    job.id,
                )
                continue

            logger.info(
                "[scheduler] Recovering missed job %s (next_run=%s, missed by %.0fs)",
                job.id, next_run, (now - next_run).total_seconds(),
            )
            # Schedule immediate async execution via the event loop.
            self._loop_call_soon(
                task_id, message, ai_prompt, thread_id, background, execution_mode,
            )
            recovered += 1

        if recovered:
            logger.info("[scheduler] Recovered %d missed job(s)", recovered)

    def _loop_call_soon(self, task_id: str, message: str,
                        ai_prompt: str | None, thread_id: str | None,
                        background: bool, execution_mode: str | None) -> None:
        """Schedule _fire_task on the event loop from the synchronous start() context."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._fire_task(task_id, message, ai_prompt, thread_id, background, execution_mode),
                name=f"recover_{task_id}",
            )
        except RuntimeError:
            # No running loop yet — defer via call_soon_threadsafe if possible.
            logger.warning(
                "[scheduler] No running event loop during recovery — job %s will "
                "fire at its next scheduled time", task_id,
            )

    # ------------------------------------------------------------------
    # Time parsing
    # ------------------------------------------------------------------

    def _parse_trigger(self, inputs: dict):
        """Return an APScheduler trigger from structured schedule inputs."""
        schedule_type = inputs.get("schedule_type", "once")

        if schedule_type == "daily":
            h, m = _parse_hhmm(inputs.get("at", "09:00"))
            return CronTrigger(hour=h, minute=m)

        if schedule_type == "weekly":
            h, m = _parse_hhmm(inputs.get("at", "09:00"))
            day = inputs.get("day_of_week", "mon")
            return CronTrigger(day_of_week=day, hour=h, minute=m)

        if schedule_type == "monthly":
            h, m = _parse_hhmm(inputs.get("at", "09:00"))
            day = int(inputs.get("day_of_month", 1))
            return CronTrigger(day=day, hour=h, minute=m)

        if schedule_type == "interval":
            interval_seconds = int(inputs["interval_seconds"])
            window_start = inputs.get("window_start")
            window_end   = inputs.get("window_end")

            if window_start and window_end:
                sh, sm = _parse_hhmm(window_start)
                eh, _  = _parse_hhmm(window_end)

                if interval_seconds >= 3600 and interval_seconds % 3600 == 0:
                    interval_hours = interval_seconds // 3600
                    hours = list(range(sh, eh + 1, interval_hours))
                    return CronTrigger(
                        hour=",".join(str(h) for h in hours),
                        minute=sm,
                    )

                if interval_seconds >= 60 and interval_seconds % 60 == 0:
                    interval_minutes = interval_seconds // 60
                    minute_expr = f"{sm}/{interval_minutes}" if sm else f"*/{interval_minutes}"
                    return CronTrigger(hour=f"{sh}-{eh}", minute=minute_expr)

            return IntervalTrigger(seconds=interval_seconds)

        # Default: once.
        fire_at = _parse_once_at(inputs.get("at", ""), tz_name=self._cfg.timezone)
        return DateTrigger(run_date=fire_at)


# Backward-compat alias so existing imports keep working.
RoutineScheduler = TaskScheduler


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hhmm(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute). Defaults to (9, 0) on failure."""
    m = re.search(r"(\d{1,2}):(\d{2})", s.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return 9, 0


def _parse_once_at(spec: str, tz_name: str = "UTC") -> datetime:
    """Parse a one-time fire datetime from natural language or ISO-8601."""
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = ZoneInfo("UTC")
    now = datetime.now(local_tz)
    s = spec.strip().lower()

    m = re.match(r"in\s+(\d+)\s+(minute|hour|day)s?", s)
    if m:
        amount = int(m.group(1))
        unit   = m.group(2)
        delta  = {"minute": timedelta(minutes=amount),
                  "hour":   timedelta(hours=amount),
                  "day":    timedelta(days=amount)}[unit]
        return now + delta

    if s.startswith("tomorrow"):
        h, minute = _parse_hhmm(spec)
        base = now + timedelta(days=1)
        return base.replace(hour=h, minute=minute, second=0, microsecond=0)

    try:
        dt = dateutil_parser.parse(spec, default=now.replace(tzinfo=None))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
        return dt
    except Exception:  # noqa: BLE001
        logger.warning("Could not parse once_at '%s', defaulting to +1h", spec)
        return now + timedelta(hours=1)
