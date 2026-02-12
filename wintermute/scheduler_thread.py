"""
Reminder Scheduler Thread

Uses APScheduler with a persistent SQLite job store so reminders survive
restarts.  At startup it detects missed executions and runs them immediately.

The ``set_reminder`` function is injected into the tools module so the LLM
can schedule jobs through the normal tool interface.

Natural-language time parsing is handled by a simple heuristic + dateutil.
For production use you might replace this with a dedicated NLP parser.
"""

import asyncio
import json
import logging
import time
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

from openai import AsyncOpenAI

from wintermute import tools as tool_module
from wintermute.dreaming import run_dream_cycle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wintermute.sub_session import SubSessionManager

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
REGISTRY_FILE = DATA_DIR / "reminders.json"
SCHEDULER_DB = "data/scheduler.db"

# Module-level reference so the job function below can be pickled by APScheduler.
# Set by ReminderScheduler.start().
_instance: Optional["ReminderScheduler"] = None


async def _fire_reminder_job(job_id: str, message: str, ai_prompt: Optional[str],
                             thread_id: Optional[str] = None) -> None:
    """
    Module-level coroutine used as the APScheduler job callable.
    Must be at module level so pickle can serialize it by reference
    (pickle only stores the dotted name, not the function body).
    """
    if _instance is not None:
        await _instance._fire_reminder(job_id, message, ai_prompt, thread_id)


DREAMING_JOB_ID = "system_dreaming"


async def _fire_dreaming_job() -> None:
    """Module-level coroutine for the nightly dreaming job."""
    if _instance is not None:
        await _instance._fire_dreaming()


@dataclass
class DreamingConfig:
    hour: int = 1
    minute: int = 0
    model: Optional[str] = None  # None = fall back to compaction_model, then main model


@dataclass
class SchedulerConfig:
    timezone: str = "UTC"
    dreaming: Optional[DreamingConfig] = None


class ReminderScheduler:
    """Wraps APScheduler and manages the reminder registry."""

    def __init__(self, config: SchedulerConfig, broadcast_fn, llm_enqueue_fn,
                 llm_client: Optional[AsyncOpenAI] = None,
                 llm_model: Optional[str] = None,
                 compaction_model: Optional[str] = None,
                 sub_session_manager: "Optional[SubSessionManager]" = None) -> None:
        self._cfg = config
        self._broadcast = broadcast_fn     # async callable(text, thread_id=None)
        self._llm_enqueue = llm_enqueue_fn  # async callable(text, thread_id) for thread-bound events
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._compaction_model = compaction_model
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

        # Register the set_reminder callable into the tools module.
        tool_module.register_scheduler(self._schedule_reminder)

        self._recover_missed()
        self._ensure_dreaming_job()
        logger.info("Reminder scheduler started (timezone=%s)", self._cfg.timezone)

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Reminder scheduler stopped")

    def delete_reminder(self, job_id: str) -> bool:
        """Remove a reminder by job_id from both APScheduler and the registry.

        Returns True if the job was found (in either place) and removed.
        """
        found_in_scheduler = False
        if self._scheduler.get_job(job_id) is not None:
            self._scheduler.remove_job(job_id)
            found_in_scheduler = True

        reg = _load_registry()
        entry = next((e for e in reg["active"] if e["id"] == job_id), None)
        if entry:
            reg["active"].remove(entry)
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
            reg.setdefault("cancelled", []).append(entry)
            _save_registry(reg)
            return True
        return found_in_scheduler

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
                "job_id":    job_id,
                "message":   message,
                "ai_prompt": ai_prompt,
                "thread_id": thread_id,
            },
            replace_existing=True,
            misfire_grace_time=3600,  # allow up to 1h late execution
        )

        next_run = self._scheduler.get_job(job_id).next_run_time
        self._registry_add({
            "id":        job_id,
            "created":   datetime.now(timezone.utc).isoformat(),
            "type":      schedule_type,
            "schedule":  _describe_schedule(inputs),
            "message":   message,
            "ai_prompt": ai_prompt,
            "thread_id": thread_id,
            "next_run":  next_run.isoformat() if next_run else None,
        })

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
                    # Thread-bound reminder with AI prompt: goes through the
                    # normal LLM queue so the result is delivered back to the
                    # originating room/tab in conversation context.
                    await self._llm_enqueue(
                        f"[REMINDER {job_id}] {ai_prompt}\n\n"
                        f"(Original reminder message: {message})",
                        thread_id,
                    )
                else:
                    # System reminder with AI prompt: run in an isolated
                    # sub-session — no thread history, no chat delivery.
                    if self._sub_sessions is not None:
                        self._sub_sessions.spawn(
                            objective=(
                                f"[REMINDER {job_id}] {ai_prompt}\n\n"
                                f"(Original reminder message: {message})"
                            ),
                            parent_thread_id=None,   # fire-and-forget
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
                    # System reminder without AI prompt — log only.
                    logger.info("System reminder %s: %s", job_id, message)

            # If the job is still scheduled (recurring), update next_run in registry.
            # Only move to "completed" for one-time jobs that are no longer in APScheduler.
            job = self._scheduler.get_job(job_id)
            if job is not None:
                self._registry_update_next_run(job_id, job.next_run_time)
            else:
                self._registry_move(job_id, "completed")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reminder %s failed", job_id)
            self._registry_move(job_id, "failed", error=str(exc))
            try:
                if thread_id:
                    await self._broadcast(f"\u274c Reminder {job_id} failed: {exc}", thread_id)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Dreaming (nightly memory consolidation)
    # ------------------------------------------------------------------

    def _ensure_dreaming_job(self) -> None:
        """Register the nightly dreaming job if not already present."""
        dreaming_cfg = self._cfg.dreaming
        if dreaming_cfg is None:
            # Dreaming not configured — use defaults
            dreaming_cfg = DreamingConfig()

        existing = self._scheduler.get_job(DREAMING_JOB_ID)
        if existing:
            logger.debug("Dreaming job already registered, next_run=%s", existing.next_run_time)
            return

        trigger = CronTrigger(hour=dreaming_cfg.hour, minute=dreaming_cfg.minute)
        self._scheduler.add_job(
            _fire_dreaming_job,
            trigger=trigger,
            id=DREAMING_JOB_ID,
            replace_existing=True,
            misfire_grace_time=7200,  # allow up to 2h late
        )
        next_run = self._scheduler.get_job(DREAMING_JOB_ID).next_run_time
        logger.info("Dreaming job registered at %02d:%02d (next_run=%s)",
                     dreaming_cfg.hour, dreaming_cfg.minute, next_run)

    async def _fire_dreaming(self) -> None:
        """Execute the nightly memory consolidation."""
        if self._llm_client is None:
            logger.error("Dreaming: no LLM client available, skipping")
            return

        # Resolve model: dreaming config override > compaction_model > main model
        dreaming_cfg = self._cfg.dreaming or DreamingConfig()
        model = dreaming_cfg.model or self._compaction_model or self._llm_model
        if not model:
            logger.error("Dreaming: no model configured, skipping")
            return

        logger.info("Dreaming: starting nightly consolidation (model=%s)", model)
        try:
            await run_dream_cycle(
                client=self._llm_client,
                model=model,
            )
            logger.info("Dreaming: nightly consolidation complete")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dreaming: nightly consolidation failed: %s", exc)

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def _recover_missed(self) -> None:
        """Execute any jobs that were missed during downtime."""
        now = datetime.now(timezone.utc)
        for job in self._scheduler.get_jobs():
            if job.next_run_time is None:
                continue
            # APScheduler already handles misfire_grace_time, but we log.
            logger.debug("Loaded job %s, next_run=%s", job.id, job.next_run_time)

    # ------------------------------------------------------------------
    # Time parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_trigger(inputs: dict):
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
                    # Whole-hour interval: enumerate firing hours explicitly.
                    interval_hours = interval_seconds // 3600
                    hours = list(range(sh, eh + 1, interval_hours))
                    return CronTrigger(
                        hour=",".join(str(h) for h in hours),
                        minute=sm,
                    )

                if interval_seconds >= 60 and interval_seconds % 60 == 0:
                    # Whole-minute interval: use */N within the hour window.
                    interval_minutes = interval_seconds // 60
                    minute_expr = f"{sm}/{interval_minutes}" if sm else f"*/{interval_minutes}"
                    return CronTrigger(hour=f"{sh}-{eh}", minute=minute_expr)

            return IntervalTrigger(seconds=interval_seconds)

        # Default: once.
        fire_at = _parse_once_at(inputs.get("at", ""))
        return DateTrigger(run_date=fire_at)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hhmm(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute). Defaults to (9, 0) on failure."""
    import re
    m = re.match(r"(\d{1,2}):(\d{2})", s.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return 9, 0


def _parse_once_at(spec: str) -> datetime:
    """Parse a one-time fire datetime from natural language or ISO-8601."""
    now = datetime.now(timezone.utc)
    s = spec.strip().lower()

    import re
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
            dt = dt.replace(tzinfo=timezone.utc)
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
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"active": [], "completed": [], "failed": []}


def _save_registry(reg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(
        json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _registry_add(self, entry: dict) -> None:
    reg = _load_registry()
    reg["active"].append(entry)
    _save_registry(reg)


def _registry_move(self, job_id: str, destination: str, error: Optional[str] = None) -> None:
    reg = _load_registry()
    entry = next((e for e in reg["active"] if e["id"] == job_id), None)
    if entry:
        reg["active"].remove(entry)
        if error:
            entry["error"] = error
        entry["completed_at"] = datetime.now(timezone.utc).isoformat()
        reg.setdefault(destination, []).append(entry)
        _save_registry(reg)


def _registry_update_next_run(self, job_id: str, next_run_time) -> None:
    """Update the next_run field for a recurring job that is still active."""
    reg = _load_registry()
    entry = next((e for e in reg["active"] if e["id"] == job_id), None)
    if entry:
        entry["next_run"] = next_run_time.isoformat() if next_run_time else None
        _save_registry(reg)


# Bind as methods (avoid self-less helper issue).
ReminderScheduler._registry_add = _registry_add
ReminderScheduler._registry_move = _registry_move
ReminderScheduler._registry_update_next_run = _registry_update_next_run
