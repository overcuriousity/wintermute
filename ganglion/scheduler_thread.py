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

from ganglion import tools as tool_module
from ganglion.dreaming import run_dream_cycle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ganglion.sub_session import SubSessionManager

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

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_reminder(self, inputs: dict) -> str:
        """Called by the tools module. Parses inputs and creates the job."""
        time_spec  = inputs["time_spec"]
        message    = inputs["message"]
        ai_prompt  = inputs.get("ai_prompt")
        recurring  = inputs.get("recurring", "none")
        thread_id  = inputs.get("thread_id")  # None for system reminders

        job_id = f"reminder_{uuid.uuid4().hex[:8]}"
        trigger = self._parse_trigger(time_spec, recurring)

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
            "type":      recurring if recurring not in ("none", "interval") else (
                             "one-time" if recurring == "none" else f"interval:{time_spec}"
                         ),
            "schedule":  time_spec,
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
    def _parse_trigger(time_spec: str, recurring: str):
        """
        Return an APScheduler trigger for the given time_spec + recurring type.
        """
        spec = time_spec.strip().lower()

        # Recurring types take priority if specified.
        if recurring == "daily":
            # Try to extract HH:MM from spec, default to 09:00.
            hour, minute = _extract_time(spec)
            return CronTrigger(hour=hour, minute=minute)

        if recurring == "weekly":
            hour, minute = _extract_time(spec)
            day = _extract_day_of_week(spec) or "mon"
            return CronTrigger(day_of_week=day, hour=hour, minute=minute)

        if recurring == "monthly":
            hour, minute = _extract_time(spec)
            day = _extract_day_of_month(spec) or 1
            return CronTrigger(day=day, hour=hour, minute=minute)

        if recurring == "interval":
            delta = _extract_interval(spec)
            return IntervalTrigger(seconds=int(delta.total_seconds()))

        # One-time: relative or absolute.
        fire_at = _parse_one_time(spec)
        return DateTrigger(run_date=fire_at)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_time(spec: str):
    """Return (hour, minute) from spec, defaulting to (9, 0)."""
    import re
    m = re.search(r"(\d{1,2}):(\d{2})", spec)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d{1,2})\s*(am|pm)", spec)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h != 12:
            h += 12
        if m.group(2) == "am" and h == 12:
            h = 0
        return h, 0
    return 9, 0


def _extract_day_of_week(spec: str) -> Optional[str]:
    days = {"monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun"}
    for name, abbr in days.items():
        if name in spec or abbr in spec:
            return abbr
    return None


def _extract_day_of_month(spec: str) -> Optional[int]:
    import re
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\b", spec)
    if m:
        return int(m.group(1))
    return None


def _extract_interval(spec: str) -> timedelta:
    """
    Parse an interval duration from natural language.
    Examples: 'every 5 minutes', '30 minutes', '2 hours', '1 day'.
    Defaults to 1 hour if nothing matches.
    """
    import re
    spec = spec.lower()
    m = re.search(r"(\d+)\s*(minute|min|hour|hr|day|second|sec)s?", spec)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        if unit in ("minute", "min"):
            return timedelta(minutes=amount)
        if unit in ("hour", "hr"):
            return timedelta(hours=amount)
        if unit == "day":
            return timedelta(days=amount)
        if unit in ("second", "sec"):
            return timedelta(seconds=amount)
    logger.warning("Could not parse interval from '%s', defaulting to 1 hour", spec)
    return timedelta(hours=1)


def _parse_one_time(spec: str) -> datetime:
    """Parse a one-time fire datetime from natural language or ISO-8601."""
    now = datetime.now(timezone.utc)

    import re
    # "in X minutes/hours/days"
    m = re.match(r"in\s+(\d+)\s+(minute|hour|day)s?", spec)
    if m:
        amount = int(m.group(1))
        unit   = m.group(2)
        delta  = {"minute": timedelta(minutes=amount),
                  "hour":   timedelta(hours=amount),
                  "day":    timedelta(days=amount)}[unit]
        return now + delta

    # "tomorrow [HH:MM]"
    if spec.startswith("tomorrow"):
        hour, minute = _extract_time(spec)
        base = now + timedelta(days=1)
        return base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # Try dateutil as fallback.
    try:
        dt = dateutil_parser.parse(spec, default=now.replace(tzinfo=None))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        # Last resort: 1 hour from now.
        logger.warning("Could not parse time spec '%s', defaulting to +1h", spec)
        return now + timedelta(hours=1)


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
