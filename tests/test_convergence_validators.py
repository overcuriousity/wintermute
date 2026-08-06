"""Tests for the Convergence Protocol stage-2 programmatic validators.

Each validator returns True when the violation is confirmed and False when
Stage 1's detection was a false positive.  The false-positive cases matter
most: an over-firing validator blocks legitimate agent actions.
"""

import pytest

from wintermute.core import convergence_protocol as cp
from wintermute.core.tool_schemas import TOOL_CATEGORIES

# ── workflow_spawn ──────────────────────────────────────────────────


def test_workflow_spawn_confirmed_when_not_called():
    assert cp.validate_workflow_spawn({"tool_calls_made": []}, {}) is True


def test_workflow_spawn_false_positive_called_this_turn():
    ctx = {"tool_calls_made": ["worker_delegation"]}
    assert cp.validate_workflow_spawn(ctx, {}) is False


def test_workflow_spawn_false_positive_called_prior_turn():
    ctx = {"tool_calls_made": [], "prior_tool_calls_made": ["worker_delegation"]}
    assert cp.validate_workflow_spawn(ctx, {}) is False


# ── phantom_tool_result ─────────────────────────────────────────────


def test_phantom_confirmed_when_claimed_tool_never_called():
    ctx = {"tool_calls_made": []}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is True


def test_phantom_false_positive_claimed_tool_was_called():
    ctx = {"tool_calls_made": ["read_file"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_execute_shell_substitutes():
    """execute_shell can read files, list dirs, fetch URLs — it covers for any
    file/data claim."""
    ctx = {"tool_calls_made": ["execute_shell"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_claimed_tool_in_prior_turn():
    ctx = {"tool_calls_made": [], "prior_tool_calls_made": ["read_file"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_no_tool_named_but_tools_ran():
    ctx = {"tool_calls_made": ["append_memory"]}
    assert cp.validate_phantom_tool_result(ctx, {}) is False


# ── empty_promise ───────────────────────────────────────────────────


def test_empty_promise_confirmed_when_no_tools_called():
    ctx = {
        "tool_calls_made": [],
        "assistant_response": "I will now read the config file and report back.",
    }
    assert cp.validate_empty_promise(ctx, {}) is True


def test_empty_promise_false_positive_when_tools_called():
    ctx = {"tool_calls_made": ["read_file"], "assistant_response": "Reading it now."}
    assert cp.validate_empty_promise(ctx, {}) is False


def test_empty_promise_false_positive_pure_confirmation_question():
    """A response that is only a question is waiting for approval, not
    promising anything."""
    ctx = {"tool_calls_made": [], "assistant_response": "Should I restart the service?"}
    assert cp.validate_empty_promise(ctx, {}) is False


def test_empty_promise_confirmed_despite_trailing_question():
    """A generic closer appended to a real commitment does not excuse it.
    The line before the question is substantive (>20 chars)."""
    ctx = {
        "tool_calls_made": [],
        "assistant_response": (
            "I will now read the config file and report the results back to you.\nAnything else?"
        ),
    }
    assert cp.validate_empty_promise(ctx, {}) is True


# ── task_complete ───────────────────────────────────────────────────


def test_task_complete_confirmed_on_short_reason():
    ctx = {"tool_name": "task", "tool_args": {"action": "complete", "reason": "ok"}}
    assert cp.validate_task_complete(ctx, {}) is True
    assert "insufficient reason" in ctx["_convergence_hook_reason"]


def test_task_complete_passes_with_substantive_reason():
    ctx = {
        "tool_name": "task",
        "tool_args": {"action": "complete", "reason": "verified all tests pass"},
    }
    assert cp.validate_task_complete(ctx, {}) is False


def test_task_complete_ignores_other_actions():
    ctx = {"tool_name": "task", "tool_args": {"action": "list"}}
    assert cp.validate_task_complete(ctx, {}) is False


def test_task_complete_ignores_other_tools():
    ctx = {"tool_name": "read_file", "tool_args": {"action": "complete"}}
    assert cp.validate_task_complete(ctx, {}) is False


# ── repetition_loop ─────────────────────────────────────────────────


def test_repetition_loop_confirmed_on_near_identical_response():
    text = "I checked the scheduler and it reports three pending jobs right now."
    ctx = {"assistant_response": text, "recent_assistant_messages": [text]}
    assert cp.validate_repetition_loop(ctx, {}) is True
    assert "similar" in ctx["_convergence_hook_reason"]


def test_repetition_loop_passes_on_different_response():
    ctx = {
        "assistant_response": "I checked the scheduler and it reports three pending jobs.",
        "recent_assistant_messages": [
            "The memory store contains four hundred entries after consolidation."
        ],
    }
    assert cp.validate_repetition_loop(ctx, {}) is False


def test_repetition_loop_ignores_short_responses():
    ctx = {"assistant_response": "ok", "recent_assistant_messages": ["ok"]}
    assert cp.validate_repetition_loop(ctx, {}) is False


def test_repetition_loop_passes_with_no_history():
    ctx = {
        "assistant_response": "I checked the scheduler and it reports three pending jobs.",
        "recent_assistant_messages": [],
    }
    assert cp.validate_repetition_loop(ctx, {}) is False


# ── inline_tool_limit ───────────────────────────────────────────────


def _exec_tools(n: int) -> list[str]:
    """Return *n* tool names whose category is execution or research."""
    names = [t for t, c in sorted(TOOL_CATEGORIES.items()) if c in ("execution", "research")]
    assert len(names) >= n, "not enough execution/research tools to build the fixture"
    return names[:n]


@pytest.fixture()
def inline_limit_two():
    """Set the module-level inline tool limit to 2 and restore it after."""
    original = cp._max_inline_tool_rounds
    cp.set_max_inline_tool_rounds(2)
    yield
    cp.set_max_inline_tool_rounds(original)


def test_inline_tool_limit_confirmed_at_limit(inline_limit_two):
    tools = _exec_tools(3)
    ctx = {"scope": "main", "tool_name": tools[2], "tool_calls_made": tools[:2]}
    assert cp.validate_inline_tool_limit(ctx, {}) is True


def test_inline_tool_limit_passes_below_limit(inline_limit_two):
    tools = _exec_tools(2)
    ctx = {"scope": "main", "tool_name": tools[1], "tool_calls_made": tools[:1]}
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_never_blocks_orchestration_tools(inline_limit_two):
    """The model must always be able to delegate, however many tools it ran."""
    ctx = {
        "scope": "main",
        "tool_name": "worker_delegation",
        "tool_calls_made": _exec_tools(3),
    }
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_scoped_to_main(inline_limit_two):
    tools = _exec_tools(3)
    ctx = {"scope": "sub_session", "tool_name": tools[2], "tool_calls_made": tools[:2]}
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_disabled_when_zero():
    original = cp._max_inline_tool_rounds
    cp.set_max_inline_tool_rounds(0)
    try:
        tools = _exec_tools(3)
        ctx = {"scope": "main", "tool_name": tools[2], "tool_calls_made": tools[:2]}
        assert cp.validate_inline_tool_limit(ctx, {}) is False
    finally:
        cp.set_max_inline_tool_rounds(original)


def test_inline_tool_limit_per_thread_override_wins(inline_limit_two):
    """extra_context.max_inline_tool_rounds takes precedence over the global."""
    tools = _exec_tools(3)
    ctx = {
        "scope": "main",
        "tool_name": tools[2],
        "tool_calls_made": tools[:2],
        "extra_context": {"max_inline_tool_rounds": 10},
    }
    assert cp.validate_inline_tool_limit(ctx, {}) is False


# ── credential_redaction ────────────────────────────────────────────


def test_credential_redaction_confirmed_when_placeholder_present():
    ctx = {"assistant_response": f"Your key is {cp._SECRET_PLACEHOLDER} now."}
    assert cp.validate_credential_redaction(ctx, {}) is True


def test_credential_redaction_passes_on_clean_response():
    ctx = {"assistant_response": "Everything looks fine."}
    assert cp.validate_credential_redaction(ctx, {}) is False


# ── tool_schema ─────────────────────────────────────────────────────


def test_tool_schema_skips_when_not_pre_execution():
    """tool_args is None outside the pre_execution phase."""
    assert cp.validate_tool_schema({"tool_name": "task", "tool_args": None}, {}) is False


def test_tool_schema_skips_unknown_tool():
    ctx = {"tool_name": "no_such_tool_exists", "tool_args": {}}
    assert cp.validate_tool_schema(ctx, {}) is False


def test_tool_schema_flags_invalid_args_and_records_reason():
    ctx = {"tool_name": "task", "tool_args": {"totally_bogus_field": 1}}
    assert cp.validate_tool_schema(ctx, {}) is True
    assert ctx["_convergence_hook_reason"]


def test_tool_schema_registry_covers_every_validator():
    """Guard against a validator being added without registration."""
    assert set(cp._PROGRAMMATIC_VALIDATORS) == {
        "validate_workflow_spawn",
        "validate_phantom_tool_result",
        "validate_empty_promise",
        "validate_tool_schema",
        "validate_task_complete",
        "validate_repetition_loop",
        "validate_inline_tool_limit",
        "validate_credential_redaction",
    }
