"""Tests for task tool schedule validation and scheduler re-registration."""

import json
from unittest.mock import Mock

import pytest

from wintermute.core.tool_deps import ToolDeps
from wintermute.infra import database
from wintermute.tools.task_tools import (
    _build_schedule,
    _resolve_execution_mode,
    tool_task,
)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point the database module at a throwaway SQLite file."""
    monkeypatch.setattr(database, "CONVERSATION_DB", tmp_path / "test.db")
    database.init_db()
    yield database
    database.close_all_connections()


@pytest.fixture()
def deps():
    d = ToolDeps()
    d.task_scheduler = Mock()
    return d


def _result(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# _build_schedule
# ---------------------------------------------------------------------------

class TestBuildSchedule:
    def test_daily_default_time(self):
        sched, desc = _build_schedule({"schedule_type": "daily"})
        assert sched["at"] == "09:00"
        assert desc == "daily at 09:00"

    def test_weekly_default_day(self):
        sched, _ = _build_schedule({"schedule_type": "weekly", "at": "10:30"})
        assert sched["day_of_week"] == "mon"

    def test_monthly_default_day(self):
        sched, _ = _build_schedule({"schedule_type": "monthly", "at": "08:00"})
        assert sched["day_of_month"] == 1

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="schedule_type"):
            _build_schedule({"schedule_type": "hourly"})

    def test_once_requires_at(self):
        with pytest.raises(ValueError, match="at"):
            _build_schedule({"schedule_type": "once"})

    def test_interval_requires_seconds(self):
        with pytest.raises(ValueError, match="interval_seconds"):
            _build_schedule({"schedule_type": "interval"})

    def test_interval_seconds_must_be_int(self):
        with pytest.raises(ValueError, match="interval_seconds must be an integer"):
            _build_schedule({"schedule_type": "interval", "interval_seconds": "abc"})

    def test_interval_seconds_coerced_to_int(self):
        sched, desc = _build_schedule({"schedule_type": "interval",
                                       "interval_seconds": "3600"})
        assert sched["interval_seconds"] == 3600
        assert desc == "every 3600s"

    def test_interval_seconds_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            _build_schedule({"schedule_type": "interval", "interval_seconds": 0})

    def test_day_of_month_must_be_int(self):
        with pytest.raises(ValueError, match="day_of_month must be an integer"):
            _build_schedule({"schedule_type": "monthly", "at": "08:00",
                             "day_of_month": "first"})


# ---------------------------------------------------------------------------
# _resolve_execution_mode
# ---------------------------------------------------------------------------

class TestResolveExecutionMode:
    def test_mode_requires_schedule(self):
        with pytest.raises(ValueError, match="only valid for scheduled"):
            _resolve_execution_mode(None, None, "reminder", False)

    def test_reminder_rejects_prompt(self):
        with pytest.raises(ValueError, match="not allowed"):
            _resolve_execution_mode("daily", "do x", "reminder", False)

    def test_autonomous_requires_prompt(self):
        with pytest.raises(ValueError, match="ai_prompt is required"):
            _resolve_execution_mode("daily", None, "autonomous_notify", True)

    def test_cleared_prompt_resolves_to_reminder(self):
        mode, background = _resolve_execution_mode("daily", None, None, True,
                                                   background_provided=True)
        assert mode == "reminder"
        assert background is False

    def test_legacy_inference_notify(self):
        mode, background = _resolve_execution_mode("daily", "do x", None, True,
                                                   background_provided=True)
        assert mode == "autonomous_notify"
        assert background is True

    def test_legacy_inference_silent(self):
        mode, background = _resolve_execution_mode("daily", "do x", None, False,
                                                   background_provided=True)
        assert mode == "autonomous_silent"
        assert background is True

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="execution_mode must be one of"):
            _resolve_execution_mode("daily", None, "bogus", False)


# ---------------------------------------------------------------------------
# Tool actions against a real database + mocked scheduler
# ---------------------------------------------------------------------------

class TestTaskAdd:
    def test_add_scheduled_registers_job(self, db, deps):
        res = _result(tool_task(
            {"action": "add", "content": "stand up", "schedule_type": "daily"},
            thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        assert res["scheduled"] is True
        deps.task_scheduler.ensure_job.assert_called_once()
        task = db.get_task(res["task_id"])
        assert task["apscheduler_job_id"] == res["task_id"]

    def test_invalid_interval_leaves_no_orphan_row(self, db, deps):
        res = _result(tool_task(
            {"action": "add", "content": "x", "schedule_type": "interval",
             "interval_seconds": "abc"},
            thread_id="t1", tool_deps=deps))
        assert "error" in res
        assert db.list_tasks("all") == []
        deps.task_scheduler.ensure_job.assert_not_called()

    def test_failing_scheduler_is_reported_not_raised(self, db, deps):
        deps.task_scheduler.ensure_job.side_effect = RuntimeError("boom")
        res = _result(tool_task(
            {"action": "add", "content": "x", "schedule_type": "daily"},
            thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        assert res["scheduled"] is False


class TestTaskUpdate:
    def _add_autonomous(self, deps) -> str:
        res = _result(tool_task(
            {"action": "add", "content": "check mail", "schedule_type": "daily",
             "ai_prompt": "summarise new mail"},
            thread_id="t1", tool_deps=deps))
        return res["task_id"]

    def test_update_reregisters_job(self, db, deps):
        task_id = self._add_autonomous(deps)
        deps.task_scheduler.ensure_job.reset_mock()
        res = _result(tool_task({"action": "update", "task_id": task_id,
                                 "content": "check inbox"},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        deps.task_scheduler.ensure_job.assert_called_once()

    def test_clearing_prompt_converts_to_reminder(self, db, deps):
        task_id = self._add_autonomous(deps)
        res = _result(tool_task({"action": "update", "task_id": task_id,
                                 "ai_prompt": ""},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        task = db.get_task(task_id)
        assert task["ai_prompt"] is None
        assert task["execution_mode"] == "reminder"
        assert task["background"] == 0

    def test_scoped_task_not_editable_from_other_thread(self, db, deps):
        task_id = self._add_autonomous(deps)
        res = _result(tool_task({"action": "update", "task_id": task_id,
                                 "content": "hijack"},
                                thread_id="t2", tool_deps=deps))
        assert res["status"] == "not_found"
        assert db.get_task(task_id)["content"] == "check mail"

    def test_unscoped_task_editable_from_any_thread(self, db, deps):
        # Simulates a web-created task (thread_id NULL).
        task_id = db.add_task(content="web task", thread_id=None)
        res = _result(tool_task({"action": "update", "task_id": task_id,
                                 "content": "edited via tool"},
                                thread_id="t9", tool_deps=deps))
        assert res["status"] == "ok"
        assert db.get_task(task_id)["content"] == "edited via tool"


class TestTaskLifecycle:
    def _add_scheduled(self, deps) -> str:
        res = _result(tool_task(
            {"action": "add", "content": "x", "schedule_type": "daily"},
            thread_id="t1", tool_deps=deps))
        return res["task_id"]

    def test_delete_survives_remove_job_failure(self, db, deps):
        task_id = self._add_scheduled(deps)
        deps.task_scheduler.remove_job.side_effect = RuntimeError("boom")
        res = _result(tool_task({"action": "delete", "task_id": task_id},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        assert db.get_task(task_id)["status"] == "deleted"

    def test_delete_missing_task_does_not_touch_scheduler(self, db, deps):
        res = _result(tool_task({"action": "delete", "task_id": "nope"},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "not_found"
        deps.task_scheduler.remove_job.assert_not_called()

    def test_pause_removes_job_after_db_write(self, db, deps):
        task_id = self._add_scheduled(deps)
        res = _result(tool_task({"action": "pause", "task_id": task_id},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        deps.task_scheduler.remove_job.assert_called_once_with(task_id)
        assert db.get_task(task_id)["status"] == "paused"

    def test_resume_recreates_job(self, db, deps):
        task_id = self._add_scheduled(deps)
        tool_task({"action": "pause", "task_id": task_id},
                  thread_id="t1", tool_deps=deps)
        deps.task_scheduler.ensure_job.reset_mock()
        res = _result(tool_task({"action": "resume", "task_id": task_id},
                                thread_id="t1", tool_deps=deps))
        assert res["status"] == "ok"
        deps.task_scheduler.ensure_job.assert_called_once()

    def test_complete_requires_reason(self, db, deps):
        task_id = self._add_scheduled(deps)
        res = _result(tool_task({"action": "complete", "task_id": task_id},
                                thread_id="t1", tool_deps=deps))
        assert "error" in res
        deps.task_scheduler.remove_job.assert_not_called()
        assert db.get_task(task_id)["status"] == "active"
