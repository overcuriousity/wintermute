"""Rescue XML/text-encoded tool calls from assistant responses.

Some LLM models (MiniMax, certain Llama finetunes, etc.) emit tool calls as
XML or pseudo-XML markup in their text content instead of using the structured
``tool_calls`` field.  When this happens the inference loop sees
``tool_calls=0`` and would treat the response as terminal text.

This module detects those patterns and synthesises lightweight objects that
mimic ``choice.message.tool_calls[*]`` so the existing execution pipeline
can process them without modification.

Supported patterns
------------------
* ``<tool_call>{"name": "…", "arguments": {…}}</tool_call>``
* ``<TOOL_CALL>…</TOOL_CALL>``  (case-insensitive)
* ``<function_call>…</function_call>``
* ``<tool_name>…</tool_name>``  (where *tool_name* is a known tool name)
* ``<prefix:tool_call>…[/tool_name]`` (MiniMax hybrid style)
* Fenced code blocks labelled ``tool_call`` / ``json`` that contain
  ``{"name": "…", …}``
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight stand-in for ``openai.types.chat.ChatCompletionMessageToolCall``
# ---------------------------------------------------------------------------


@dataclass
class _FunctionStub:
    name: str
    arguments: str  # JSON string


@dataclass
class SyntheticToolCall:
    """Minimal duck-typed replacement for an OpenAI tool-call object."""

    id: str
    function: _FunctionStub
    type: str = "function"

    @staticmethod
    def make(name: str, arguments: str) -> "SyntheticToolCall":
        return SyntheticToolCall(
            id=f"rescue_{uuid.uuid4().hex[:12]}",
            function=_FunctionStub(name=name, arguments=arguments),
        )


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. Generic XML wrappers: <tool_call>…</tool_call> or <function_call>…
_RE_XML_GENERIC = re.compile(
    r"<\s*(?:\w+:)?\s*(?:tool_call|function_call)\s*>"
    r"(.*?)"
    r"(?:<\s*/\s*(?:\w+:)?\s*(?:tool_call|function_call)\s*>|\[/\w+\])",
    re.DOTALL | re.IGNORECASE,
)

# 2. Tool-name tags: <spawn_sub_session …>…</spawn_sub_session> or
#    <spawn_sub_session>…[/spawn_sub_session]
#    Built dynamically per call so it only matches known tool names.

# 3. Fenced code block containing a tool-call JSON object.
_RE_FENCED = re.compile(
    r"```(?:tool_call|json|tool)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# 4. MiniMax-style: <minimax:tool_call> tool_name\n key=value … [/tool_name]
_RE_MINIMAX = re.compile(
    r"<\s*\w+:\s*tool_call\s*>\s*"
    r"(\w+)\s+"                          # tool name
    r"(.*?)"                             # body (key="value" or JSON)
    r"\[/\1\]",                          # closing [/tool_name]
    re.DOTALL | re.IGNORECASE,
)


def _build_tool_name_pattern(tool_names: set[str]) -> re.Pattern[str]:
    """Build a regex that matches ``<known_tool>…</known_tool>``."""
    escaped = "|".join(re.escape(n) for n in sorted(tool_names))
    return re.compile(
        rf"<\s*(?:\w+:)?\s*({escaped})\s*>"
        rf"(.*?)"
        rf"(?:<\s*/\s*(?:\w+:)?\s*\1\s*>|\[/\1\])",
        re.DOTALL | re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _try_parse_json_body(body: str, known_tools: set[str]) -> Optional[list[SyntheticToolCall]]:
    """Try to parse a JSON body into tool call(s).

    Accepts either:
    * ``{"name": "tool", "arguments": {…}}``
    * ``{"name": "tool", "parameters": {…}}``  (alias)
    * ``[{…}, {…}]``  (array of the above)
    """
    body = body.strip()
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    items = data if isinstance(data, list) else [data]
    results: list[SyntheticToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function", "")
        if known_tools and name not in known_tools:
            continue
        args = item.get("arguments") or item.get("parameters") or {}
        if isinstance(args, str):
            args_str = args
        elif isinstance(args, dict):
            args_str = json.dumps(args)
        else:
            # Scalar or array — wrap so downstream always receives a JSON object.
            args_str = json.dumps({"value": args})
        results.append(SyntheticToolCall.make(name, args_str))
    return results or None


def _parse_minimax_kv(body: str) -> dict:
    """Parse MiniMax-style ``key="value"`` or ``key=value`` pairs."""
    result: dict = {}
    # Match key="value" (quoted) or key=value (unquoted, up to whitespace/newline)
    for m in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"|(\w+)\s*=\s*(\S+)', body):
        if m.group(1):
            result[m.group(1)] = m.group(2).replace('\\"', '"')
        elif m.group(3):
            result[m.group(3)] = m.group(4)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rescue_tool_calls(
    content: str,
    known_tool_names: set[str] | None = None,
) -> list[SyntheticToolCall]:
    """Detect and extract tool calls encoded as XML/text in *content*.

    Parameters
    ----------
    content:
        The ``choice.message.content`` text from the LLM response.
    known_tool_names:
        Set of valid tool names.  Used to validate extracted names and to
        build the tool-name-tag pattern.  If ``None``, validation is
        skipped (not recommended).

    Returns
    -------
    list[SyntheticToolCall]
        Zero or more synthetic tool-call objects ready for the execution
        pipeline.  An empty list means no rescue was possible.
    """
    if not content:
        return []

    known = known_tool_names if known_tool_names is not None else set()
    results: list[SyntheticToolCall] = []
    seen_calls: set[tuple[str, str]] = set()  # deduplicate by (name, arguments)

    def _add(tc: SyntheticToolCall) -> None:
        key = (tc.function.name, tc.function.arguments)
        if key not in seen_calls:
            seen_calls.add(key)
            results.append(tc)

    # --- Pattern 1: Generic XML wrappers ---
    for m in _RE_XML_GENERIC.finditer(content):
        parsed = _try_parse_json_body(m.group(1), known)
        if parsed:
            for tc in parsed:
                _add(tc)

    # --- Pattern 2: Fenced code blocks ---
    for m in _RE_FENCED.finditer(content):
        parsed = _try_parse_json_body(m.group(1), known)
        if parsed:
            for tc in parsed:
                _add(tc)

    # --- Pattern 3: MiniMax hybrid ---
    for m in _RE_MINIMAX.finditer(content):
        name = m.group(1).strip()
        body = m.group(2).strip()
        if known and name not in known:
            continue
        # Try JSON first, fall back to key=value
        try:
            args = json.loads(body)
            args_str = json.dumps(args)
        except (json.JSONDecodeError, ValueError):
            args = _parse_minimax_kv(body)
            if not args:
                continue
            args_str = json.dumps(args)
        _add(SyntheticToolCall.make(name, args_str))

    # --- Pattern 4: Known tool-name tags ---
    if known:
        pat = _build_tool_name_pattern(known)
        for m in pat.finditer(content):
            name = m.group(1)
            body = m.group(2).strip()
            # Could be JSON, key=value, or a single string argument
            try:
                args = json.loads(body)
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                else:
                    # Bare scalar or array — normalize to object so downstream
                    # tools always receive a dict.
                    args_str = json.dumps({"input": args})
            except (json.JSONDecodeError, ValueError):
                kv = _parse_minimax_kv(body)
                if kv:
                    args_str = json.dumps(kv)
                elif body:
                    # Treat as single-string value wrapped in {"input": …}
                    args_str = json.dumps({"input": body})
                else:
                    continue
            _add(SyntheticToolCall.make(name, args_str))

    if results:
        logger.warning(
            "Rescued %d XML-encoded tool call(s) from assistant content: %s",
            len(results),
            [tc.function.name for tc in results],
        )
    return results
