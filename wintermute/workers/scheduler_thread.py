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

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dateutil import parser as dateutil_parser

from wintermute import tools as tool_module
from wintermute.infra import database
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.core.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
SCHEDULER_DB = "data/scheduler.db"

# Module-level reference so the job function below can be pickled by APScheduler.
# Set by TaskScheduler.start().
_instance: Optional["TaskScheduler"] = None


async def _fire_task_job(task_id: str, message: str, ai_prompt: Optional[str],
                          thread_id: Optional[str] = None,
                          background: bool = False,
                          **_extra) -> None:
    """
    Module-level coroutine used as the APScheduler job callable.
    Must be at module level so pickle can serialize it by reference.

    **_extra absorbs metadata kwargs stored alongside the job.
    """
    if _instance is not None:
        await _instance._fire_task(task_id, message, ai_prompt, thread_id, background)


# Backward-compat aliases: jobs persisted in scheduler.db before the
# routine→task rename reference these names. APScheduler resolves
# callables by dotted name at load time, so the aliases must exist.
async def _fire_routine_job(job_id: str, message: str, ai_prompt: Optional[str],
                             thread_id: Optional[str] = None,
                             **_extra) -> None:
    if _instance is not None:
        await _instance._fire_task(job_id, message, ai_prompt, thread_id)

_fire_reminder_job = _fire_routine_job


from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    timezone: str = "UTC"


class TaskScheduler:
    """Wraps APScheduler and manages task schedules.

    APScheduler's persistent SQLite store is the single source of truth for
    active jobs.  Metadata is stored in the job's kwargs so it can be
    retrieved without a second store.
    """

    def __init__(self, config: SchedulerConfig, broadcast_fn, llm_enqueue_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._cfg = config
        self._broadcast = broadcast_fn
        self._llm_enqueue = llm_enqueue_fn
        self._sub_sessions = sub_session_manager
        self._scheduler: Optional[AsyncIOScheduler] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        jobstores = {
            "default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB}")
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

        # Register into the tools module.
        tool_module.register_task_scheduler(self.ensure_job, self.remove_job, self.list_jobs)

        self._recover_missed()
        logger.info("[scheduler] started (timezone=%s)", self._cfg.timezone)

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[scheduler] stopped")

    # ------------------------------------------------------------------
    # Job management (called by tools.py _tool_task)
    # ------------------------------------------------------------------

    def ensure_job(self, task_id: str, schedule_config: dict,
                   ai_prompt: Optional[str] = None,
                   thread_id: Optional[str] = None,
                   background: bool = False) -> None:
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
                "schedule_type": schedule_config.get("schedule_type"),
                "schedule": _describe_schedule(schedule_config),
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
                          background: bool = False) -> None:
        logger.info("Firing task %s (thread=%s, background=%s)", task_id, thread_id, background)
        try:
            if ai_prompt:
                if background:
                    # Background task: spawn an isolated sub-session with full
                    # orchestration tools.  Results are delivered back to the
                    # originating thread; [NO_ACTION] suppression prevents
                    # noise when there's nothing to report.
                    if self._sub_sessions is not None:
                        self._sub_sessions.spawn(
                            objective=(
                                f"[TASK {task_id}] {ai_prompt}\n\n"
                                f"(Task: {message})\n\n"
                                f"If you have nothing actionable to report, "
                                f"respond with exactly: [NO_ACTION]"
                            ),
                            parent_thread_id=thread_id,
                            system_prompt_mode="full",
                        )
                    else:
                        logger.warning(
                            "Task %s has ai_prompt but SubSessionManager "
                            "is not available — skipping", task_id
                        )
                elif thread_id:
                    # Foreground task: enqueue into the main LLM thread
                    # as if the user typed it.
                    await self._llm_enqueue(
                        f"[TASK {task_id}] {ai_prompt}\n\n"
                        f"(Task: {message})",
                        thread_id,
                    )
                else:
                    logger.warning(
                        "Task %s has ai_prompt but no thread_id and not "
                        "background — message was NOT delivered: %s",
                        task_id, message
                    )
            else:
                if thread_id:
                    await self._broadcast(f"\u23f0 Task: {message}", thread_id)
                else:
                    logger.warning(
                        "Task %s has no thread_id and no ai_prompt — "
                        "message was NOT delivered: %s", task_id, message
                    )

            # Record execution in the tasks table.
            try:
                database.record_task_run(task_id, summary="executed")
            except Exception:  # noqa: BLE001
                logger.debug("Failed to record task run for %s", task_id)

            # If the job is no longer in APScheduler (one-time, now done), log it.
            if self._scheduler.get_job(task_id) is None:
                logger.info("One-time task %s completed", task_id)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Task %s failed", task_id)
            try:
                database.record_task_run(task_id, summary=f"failed: {exc}")
            except Exception:  # noqa: BLE001
                pass
            try:
                if thread_id:
                    await self._broadcast(f"\u274c Task {task_id} failed: {exc}", thread_id)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def _recover_missed(self) -> None:
        """Log jobs that were loaded from the persistent store."""
        for job in self._scheduler.get_jobs():
            if job.next_run_time is None:
                continue
            logger.debug("Loaded job %s, next_run=%s", job.id, job.next_run_time)

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
    from zoneinfo import ZoneInfo
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
