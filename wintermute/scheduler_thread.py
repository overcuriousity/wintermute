"""
Reminder Scheduler

Uses APScheduler with a persistent SQLite job store so reminders survive
restarts.  At startup it detects missed executions and runs them immediately.

The ``set_reminder`` and ``list_reminders`` functions are injected into the
tools module so the LLM can schedule and query jobs through the normal tool
interface.

APScheduler's SQLite store is the single source of truth for active jobs.
A separate JSON file (data/reminder_history.json) keeps an append-only log
of completed and failed reminders for display purposes only.

Natural-language time parsing is handled by a simple heuristic + dateutil.
For production use you might replace this with a dedicated NLP parser.
"""

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "reminder_history.json"
SCHEDULER_DB = "data/scheduler.db"

# Module-level reference so the job function below can be pickled by APScheduler.
# Set by ReminderScheduler.start().
_instance: Optional["ReminderScheduler"] = None


async def _fire_reminder_job(job_id: str, message: str, ai_prompt: Optional[str],
                             thread_id: Optional[str] = None,
                             **_extra) -> None:
    """
    Module-level coroutine used as the APScheduler job callable.
    Must be at module level so pickle can serialize it by reference
    (pickle only stores the dotted name, not the function body).

    **_extra absorbs metadata kwargs (schedule, created, schedule_type)
    stored alongside the job for query purposes.
    """
    if _instance is not None:
        await _instance._fire_reminder(job_id, message, ai_prompt, thread_id)


@dataclass
class SchedulerConfig:
    timezone: str = "UTC"


class ReminderScheduler:
    """Wraps APScheduler and manages reminders.

    APScheduler's persistent SQLite store is the single source of truth for
    active reminders.  Metadata (schedule description, creation time) is
    stored in the job's kwargs so it can be retrieved without a second store.

    Completed/failed history is appended to a JSON log for display only.
    """

    def __init__(self, config: SchedulerConfig, broadcast_fn, llm_enqueue_fn,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._cfg = config
        self._broadcast = broadcast_fn     # async callable(text, thread_id=None)
        self._llm_enqueue = llm_enqueue_fn  # async callable(text, thread_id) for thread-bound events
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

        # Expose this instance so the module-level job function can reach it.
        global _instance
        _instance = self

        # Register tool callables into the tools module.
        tool_module.register_scheduler(self._schedule_reminder)
        tool_module.register_reminder_lister(self.list_reminders)
        tool_module.register_reminder_deleter(self.delete_reminder)

        self._recover_missed()
        self._migrate_legacy_registry()
        logger.info("Reminder scheduler started (timezone=%s)", self._cfg.timezone)

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Reminder scheduler stopped")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_reminders(self) -> dict:
        """Return a combined view: active from APScheduler + history from JSON."""
        active = []
        for job in self._scheduler.get_jobs():
            kw = job.kwargs or {}
            active.append({
                "id":        kw.get("job_id", job.id),
                "created":   kw.get("created"),
                "type":      kw.get("schedule_type"),
                "schedule":  kw.get("schedule"),
                "message":   kw.get("message"),
                "ai_prompt": kw.get("ai_prompt"),
                "thread_id": kw.get("thread_id"),
                "next_run":  job.next_run_time.isoformat() if job.next_run_time else None,
            })
        history = _load_history()
        return {
            "active":    active,
            "completed": history.get("completed", []),
            "failed":    history.get("failed", []),
        }

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

    def delete_reminder(self, job_id: str) -> bool:
        """Remove a reminder by job_id.  Returns True if found and removed."""
        if self._scheduler.get_job(job_id) is None:
            return False
        self._scheduler.remove_job(job_id)
        _append_history("cancelled", {
            "id": job_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_reminder(self, inputs: dict) -> str:
        """Called by the tools module. Parses inputs and creates the job."""
        message       = inputs["message"]
        ai_prompt     = inputs.get("ai_prompt")
        schedule_type = inputs.get("schedule_type", "once")
        thread_id     = inputs.get("thread_id")  # None for system reminders

        job_id = f"reminder_{uuid.uuid4().hex[:8]}"
        trigger = self._parse_trigger(inputs)

        self._scheduler.add_job(
            _fire_reminder_job,
            trigger=trigger,
            id=job_id,
            kwargs={
                "job_id":        job_id,
                "message":       message,
                "ai_prompt":     ai_prompt,
                "thread_id":     thread_id,
                "schedule_type": schedule_type,
                "schedule":      _describe_schedule(inputs),
                "created":       datetime.now(timezone.utc).isoformat(),
            },
            replace_existing=True,
            misfire_grace_time=3600,  # allow up to 1h late execution
        )

        next_run = self._scheduler.get_job(job_id).next_run_time
        logger.info("Reminder scheduled: %s at %s (thread=%s)", job_id, next_run, thread_id)
        return job_id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _fire_reminder(self, job_id: str, message: str, ai_prompt: Optional[str],
                             thread_id: Optional[str] = None) -> None:
        logger.info("Firing reminder %s (thread=%s)", job_id, thread_id)
        try:
            if ai_prompt:
                if thread_id:
                    await self._llm_enqueue(
                        f"[REMINDER {job_id}] {ai_prompt}\n\n"
                        f"(Original reminder message: {message})",
                        thread_id,
                    )
                else:
                    if self._sub_sessions is not None:
                        self._sub_sessions.spawn(
                            objective=(
                                f"[REMINDER {job_id}] {ai_prompt}\n\n"
                                f"(Original reminder message: {message})"
                            ),
                            parent_thread_id=None,
                            system_prompt_mode="base_only",
                        )
                    else:
                        logger.warning(
                            "System reminder %s has ai_prompt but SubSessionManager "
                            "is not available — skipping AI execution", job_id
                        )
            else:
                if thread_id:
                    await self._broadcast(f"\u23f0 Reminder: {message}", thread_id)
                else:
                    # No thread_id and no ai_prompt — should not happen after
                    # the tool-level fix that always injects thread_id at
                    # creation time.  Log loudly so it's visible in the journal.
                    logger.warning(
                        "Reminder %s has no thread_id and no ai_prompt — "
                        "message was NOT delivered: %s", job_id, message
                    )

            # If the job is no longer in APScheduler (one-time, now done), log it.
            if self._scheduler.get_job(job_id) is None:
                _append_history("completed", {
                    "id": job_id,
                    "message": message,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reminder %s failed", job_id)
            _append_history("failed", {
                "id": job_id,
                "message": message,
                "error": str(exc),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            try:
                if thread_id:
                    await self._broadcast(f"\u274c Reminder {job_id} failed: {exc}", thread_id)
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
    # Legacy migration
    # ------------------------------------------------------------------

    def _migrate_legacy_registry(self) -> None:
        """One-time migration from the old dual-write reminders.json.

        Moves completed/failed entries to the new history file and removes
        the legacy file.
        """
        legacy = DATA_DIR / "reminders.json"
        if not legacy.exists():
            return
        try:
            old = json.loads(legacy.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        migrated = False
        for bucket in ("completed", "failed", "cancelled"):
            entries = old.get(bucket, [])
            if entries:
                history = _load_history()
                history.setdefault(bucket, []).extend(entries)
                _save_history(history)
                migrated = True
        if migrated:
            legacy.unlink(missing_ok=True)
            logger.info("Migrated legacy reminders.json to reminder_history.json")

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


# ---------------------------------------------------------------------------
# History log (append-only, for completed/failed/cancelled)
# ---------------------------------------------------------------------------

def _load_history() -> dict:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"completed": [], "failed": []}


def _save_history(history: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


MAX_HISTORY_PER_BUCKET = 200

_history_lock = threading.Lock()


def _append_history(bucket: str, entry: dict) -> None:
    """Append an entry to the history log under the given bucket.

    Keeps only the most recent MAX_HISTORY_PER_BUCKET entries per bucket
    to prevent unbounded growth over long-running deployments.
    """
    with _history_lock:
        history = _load_history()
        items = history.setdefault(bucket, [])
        items.append(entry)
        if len(items) > MAX_HISTORY_PER_BUCKET:
            history[bucket] = items[-MAX_HISTORY_PER_BUCKET:]
        _save_history(history)
