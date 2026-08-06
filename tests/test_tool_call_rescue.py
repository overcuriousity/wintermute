"""Tests for weak-model tool-call rescue.

rescue_tool_calls salvages tool calls that a model emitted as text instead of
through the tool-calling API.  The negative cases matter as much as the
positive ones: rescuing prose into a phantom tool call is worse than rescuing
nothing.
"""

import json

from wintermute.core.tool_call_rescue import (
    _parse_arrow_style,
    _parse_cli_args,
    _parse_minimax_kv,
    _parse_yaml_like_kv,
    _try_parse_invoke_body,
    _try_parse_json_body,
    rescue_tool_calls,
)

KNOWN = {"read_file", "write_file", "worker_delegation", "task"}


def args_of(call) -> dict:
    """Decode a synthetic call's JSON argument string."""
    return json.loads(call.function.arguments)


# ── _try_parse_json_body ────────────────────────────────────────────


def test_json_body_flat_arguments():
    body = '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_parameters_alias():
    body = '{"name": "read_file", "parameters": {"path": "/tmp/x"}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_nested_function_shape():
    body = '{"function": {"name": "read_file", "arguments": {"path": "/tmp/x"}}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_array_of_calls():
    body = (
        '[{"name": "read_file", "arguments": {"path": "/a"}},'
        ' {"name": "write_file", "arguments": {"path": "/b"}}]'
    )
    calls = _try_parse_json_body(body, KNOWN)
    assert [c.function.name for c in calls] == ["read_file", "write_file"]


def test_json_body_rejects_unknown_tool_name():
    body = '{"name": "definitely_not_a_tool", "arguments": {}}'
    assert _try_parse_json_body(body, KNOWN) is None


def test_json_body_rejects_object_without_arguments_field():
    """Plain JSON in a discussion is not a tool call."""
    body = '{"name": "read_file", "description": "a tool that reads files"}'
    assert _try_parse_json_body(body, KNOWN) is None


def test_json_body_rejects_invalid_json():
    assert _try_parse_json_body("{not json at all", KNOWN) is None


def test_json_body_rejects_empty():
    assert _try_parse_json_body("   ", KNOWN) is None


# ── _try_parse_invoke_body ──────────────────────────────────────────


def test_invoke_body_single_block():
    body = '<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>'
    calls = _try_parse_invoke_body(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_invoke_body_multiple_blocks():
    body = (
        '<invoke name="read_file"><parameter name="path">/a</parameter></invoke>'
        '<invoke name="write_file"><parameter name="path">/b</parameter></invoke>'
    )
    calls = _try_parse_invoke_body(body, KNOWN)
    assert [c.function.name for c in calls] == ["read_file", "write_file"]


def test_invoke_body_skips_unknown_tool():
    body = '<invoke name="nope"><parameter name="path">/a</parameter></invoke>'
    assert _try_parse_invoke_body(body, KNOWN) is None


# ── _parse_minimax_kv ───────────────────────────────────────────────


def test_minimax_kv_quoted_and_bare():
    assert _parse_minimax_kv('path="/tmp/x" limit=10') == {"path": "/tmp/x", "limit": "10"}


def test_minimax_kv_unescapes_inner_quotes():
    assert _parse_minimax_kv(r'text="say \"hi\""') == {"text": 'say "hi"'}


# ── _parse_cli_args ─────────────────────────────────────────────────


def test_cli_args_quoted_value_and_numeric_coercion():
    parsed = _parse_cli_args('--operation "interaction_log"\n--limit 10')
    assert parsed == {"operation": "interaction_log", "limit": 10}


def test_cli_args_normalises_kebab_case_keys():
    assert _parse_cli_args("--sub-session-id abc123") == {"sub_session_id": "abc123"}


def test_cli_args_coerces_float():
    assert _parse_cli_args("--threshold 0.85") == {"threshold": 0.85}


def test_cli_args_single_quoted_value():
    assert _parse_cli_args("--path '/tmp/x'") == {"path": "/tmp/x"}


# ── _parse_yaml_like_kv ─────────────────────────────────────────────


def test_yaml_like_kv_dash_prefixed_lines():
    body = (
        '- objective: "Implement GUI functionality"\n'
        "- timeout: 600\n"
        '- sub_session_id: "sub_aa62f3c8"'
    )
    assert _parse_yaml_like_kv(body) == {
        "objective": "Implement GUI functionality",
        "timeout": 600,
        "sub_session_id": "sub_aa62f3c8",
    }


def test_yaml_like_kv_unquoted_values():
    assert _parse_yaml_like_kv("action: list\nlimit: 5") == {"action": "list", "limit": 5}


# ── _parse_arrow_style ──────────────────────────────────────────────


def test_arrow_style_with_json_args():
    body = '{tool => "read_file", args => {"path": "/tmp/x"}}'
    calls = _parse_arrow_style(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_arrow_style_rejects_unknown_tool():
    body = '{tool => "not_a_tool", args => {"path": "/tmp/x"}}'
    assert _parse_arrow_style(body, KNOWN) == []


# ── rescue_tool_calls, end to end ───────────────────────────────────


def test_rescue_generic_xml_wrapper():
    content = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"


def test_rescue_fenced_json_block():
    content = '```json\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"


def test_rescue_bare_invoke_block():
    content = (
        'Here you go:\n<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>'
    )
    calls = rescue_tool_calls(content, KNOWN)
    assert [c.function.name for c in calls] == ["read_file"]


def test_rescue_deduplicates_identical_calls():
    one = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    calls = rescue_tool_calls(one + one, KNOWN)
    assert len(calls) == 1


def test_rescue_ignores_plain_prose():
    """The most important negative case: prose must never become a tool call."""
    content = "I read the file and it contained the database configuration."
    assert rescue_tool_calls(content, KNOWN) == []


def test_rescue_ignores_markup_without_a_known_tool():
    content = '<tool_call>{"name": "not_a_real_tool", "arguments": {}}</tool_call>'
    assert rescue_tool_calls(content, KNOWN) == []


def test_rescue_returns_empty_for_empty_content():
    assert rescue_tool_calls("", KNOWN) == []


def test_rescue_returns_empty_when_no_tools_available():
    """An explicitly empty tool set means nothing can be rescued."""
    content = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    assert rescue_tool_calls(content, set()) == []


def test_rescue_assigns_unique_ids():
    content = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "/a"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "/b"}}</tool_call>'
    )
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 2
    assert calls[0].id != calls[1].id
    assert all(c.type == "function" for c in calls)
