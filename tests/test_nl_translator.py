"""Tests for natural-language tool-call detection and JSON repair.

These cover the synchronous helpers only; translate_nl_tool_call itself
requires a backend and is out of scope.
"""

import json

from wintermute.core.nl_translator import (
    NL_TOOLS,
    _fix_unescaped_control_chars,
    _task_list_fastpath,
    is_nl_tool_call,
)

# ── is_nl_tool_call ─────────────────────────────────────────────────


def test_is_nl_tool_call_true_for_nl_tool_with_description():
    assert is_nl_tool_call("task", {"description": "list my tasks"}) is True


def test_is_nl_tool_call_tolerates_extra_keys():
    """The translator uses only the description; extra keys are discarded."""
    assert is_nl_tool_call("worker_delegation", {"description": "do it", "timeout": 600}) is True


def test_is_nl_tool_call_false_for_structured_args():
    assert is_nl_tool_call("task", {"action": "list"}) is False


def test_is_nl_tool_call_false_for_non_nl_tool():
    assert is_nl_tool_call("read_file", {"description": "read something"}) is False


def test_is_nl_tool_call_false_when_description_is_not_a_string():
    assert is_nl_tool_call("task", {"description": {"nested": "object"}}) is False


def test_nl_tools_membership_is_stable():
    assert set(NL_TOOLS) == {"task", "worker_delegation", "skill"}


# ── _task_list_fastpath ─────────────────────────────────────────────


def test_task_list_fastpath_default_list():
    assert _task_list_fastpath("list tasks") == {"action": "list"}


def test_task_list_fastpath_is_case_and_punctuation_insensitive():
    assert _task_list_fastpath("  Show Me All Active Tasks!  ") == {"action": "list"}


def test_task_list_fastpath_status_variant():
    assert _task_list_fastpath("show completed tasks") == {
        "action": "list",
        "status": "completed",
    }


def test_task_list_fastpath_all_variant():
    assert _task_list_fastpath("list all tasks") == {"action": "list", "status": "all"}


def test_task_list_fastpath_returns_none_for_unknown_phrasing():
    """Anything unrecognised must fall through to the LLM translator."""
    assert _task_list_fastpath("create a task to water the plants") is None


def test_task_list_fastpath_returns_a_fresh_dict_each_call():
    """The default must be copied, or a caller mutating it corrupts the table."""
    first = _task_list_fastpath("list")
    first["action"] = "mutated"
    assert _task_list_fastpath("list") == {"action": "list"}


# ── _fix_unescaped_control_chars ────────────────────────────────────


def test_fix_control_chars_escapes_raw_newline_in_string():
    broken = '{"text": "line one\nline two"}'
    fixed = _fix_unescaped_control_chars(broken)
    assert json.loads(fixed) == {"text": "line one\nline two"}


def test_fix_control_chars_escapes_tab_and_carriage_return():
    broken = '{"text": "a\tb\rc"}'
    assert json.loads(_fix_unescaped_control_chars(broken)) == {"text": "a\tb\rc"}


def test_fix_control_chars_leaves_structural_whitespace_alone():
    valid = '{\n  "a": 1,\n  "b": 2\n}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"a": 1, "b": 2}


def test_fix_control_chars_preserves_existing_escapes():
    valid = '{"text": "already \\n escaped"}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"text": "already \n escaped"}


def test_fix_control_chars_preserves_escaped_quote():
    valid = '{"text": "he said \\"hi\\""}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"text": 'he said "hi"'}


def test_fix_control_chars_is_a_noop_on_clean_input():
    clean = '{"a": "b"}'
    assert _fix_unescaped_control_chars(clean) == clean
