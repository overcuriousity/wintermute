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
* ``[TOOL_CALL]{tool => "…", args => {--key value …}}[/TOOL_CALL]``
  (bracket wrapper with hash-rocket notation and CLI-style ``--flag value``
  arguments)
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

# 2. Tool-name tags: <worker_delegation …>…</worker_delegation> or
#    <worker_delegation>…[/worker_delegation]
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

# 5. Bracket-wrapped TOOL_CALL: [TOOL_CALL]…[/TOOL_CALL]
#    Body may use hash-rocket notation: {tool => "name", args => {…}}
_RE_BRACKET_TOOL_CALL = re.compile(
    r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]",
    re.DOTALL | re.IGNORECASE,
)


def _build_tool_name_pattern(tool_names: set[str]) -> re.Pattern[str]:
    """Build a regex that matches ``<known_tool>…</known_tool>`` or
    ``[known_tool]…[/known_tool]`` (bracket-style tags)."""
    escaped = "|".join(re.escape(n) for n in sorted(tool_names))
    return re.compile(
        rf"(?:<\s*(?:\w+:)?\s*({escaped})\s*>|\[({escaped})\])"
        rf"(.*?)"
        rf"(?:<\s*/\s*(?:\w+:)?\s*(?:{escaped})\s*>|\[/(?:{escaped})\])",
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
        # Support both flat {"name": …, "arguments": …} and nested
        # {"function": {"name": …, "arguments": …}} shapes.
        func = item.get("function")
        nested_name: str = ""
        nested_args = None
        if isinstance(func, dict):
            nested_name = func.get("name") or ""
            nested_args = func.get("arguments") or func.get("parameters")
        elif isinstance(func, str):
            nested_name = func
        name = str(item.get("name") or nested_name).strip()
        if not name:
            continue
        if known_tools and name not in known_tools:
            continue
        # Require at least one explicit arguments field to avoid treating plain
        # JSON examples or discussion snippets as tool calls.
        has_args = "arguments" in item or "parameters" in item or nested_args is not None
        if not has_args:
            continue
        args = item.get("arguments") if "arguments" in item else (
            item.get("parameters") if "parameters" in item else (nested_args or {})
        )
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


def _parse_cli_args(body: str) -> dict:
    """Parse CLI-style ``--key value`` argument strings.

    Handles bodies like::

        --operation "interaction_log"
        --limit 10

    Keys are normalised from ``kebab-case`` to ``snake_case``.
    Numeric values are converted to ``int`` / ``float`` where possible.
    """
    result: dict = {}
    for m in re.finditer(
        r"--([\w][\w-]*)\s+"
        r'(?:"((?:[^"\\]|\\.)*)"|'   # double-quoted
        r"'((?:[^'\\]|\\.)*)'|"       # single-quoted
        r"(\S+))",                     # bare word
        body,
    ):
        key = m.group(1).replace("-", "_")
        if m.group(2) is not None:
            val: "str | int | float" = m.group(2)
        elif m.group(3) is not None:
            val = m.group(3)
        else:
            raw = m.group(4)
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        result[key] = val
    return result


def _parse_arrow_style(
    body: str, known_tools: set[str]
) -> list["SyntheticToolCall"]:
    """Parse hash-rocket style: ``{tool => "name", args => {…}}``.

    The *args* block may contain JSON, key=value, YAML-like, or
    ``--flag value`` CLI-style pairs.
    """
    # Extract tool name (tool => "name" or tool => name)
    name_m = re.search(
        r'\btool\s*=>\s*["\']?([\w]+)["\']?', body, re.IGNORECASE
    )
    if not name_m:
        return []
    name = name_m.group(1).strip()
    if known_tools and name not in known_tools:
        return []

    # Extract args block: args => { … } (possibly multi-line, possibly nested)
    # We scan for the opening brace after 'args =>' and match braces manually.
    args_m = re.search(r"\bargs\s*=>\s*\{", body, re.IGNORECASE)
    if not args_m:
        return [SyntheticToolCall.make(name, "{}")]

    brace_start = args_m.end() - 1  # position of '{'
    depth = 0
    brace_end = brace_start
    for i, ch in enumerate(body[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i
                break

    args_body = body[brace_start + 1 : brace_end].strip()

    # Attempt progressively simpler parses
    args: dict = {}
    try:
        parsed = json.loads("{" + args_body + "}")
        if isinstance(parsed, dict):
            args = parsed
    except (json.JSONDecodeError, ValueError):
        pass

    if not args:
        args = _parse_cli_args(args_body)
    if not args:
        args = _parse_minimax_kv(args_body)
    if not args:
        args = _parse_yaml_like_kv(args_body)

    return [SyntheticToolCall.make(name, json.dumps(args))]


def _parse_yaml_like_kv(body: str) -> dict:
    """Parse YAML-like ``key: value`` or ``- key: value`` pairs.

    Handles bodies like::

        - objective: "Implement GUI functionality"
        - timeout: 600
        - sub_session_id: "sub_aa62f3c8"

    Values may be quoted (``"…"``) or unquoted.  Numeric strings are
    converted to ``int`` / ``float`` where appropriate.
    """
    result: dict = {}
    for m in re.finditer(
        r'^\s*-?\s*(\w+)\s*:\s*'
        r'(?:"((?:[^"\\]|\\.)*)"'
        r"|'((?:[^'\\]|\\.)*)'"   # single-quoted
        r'|(.*?))'                    # unquoted – rest of line
        r'\s*$',
        body,
        re.MULTILINE,
    ):
        key = m.group(1)
        if m.group(2) is not None:
            val: "str | int | float" = m.group(2)
        elif m.group(3) is not None:
            val = m.group(3)
        else:
            raw = (m.group(4) or "").strip()
            # Try numeric conversion.
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        result[key] = val
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

    # Cheap pre-filter: skip all regex work if no known markup markers are present.
    if not any(marker in content for marker in ("<", "```", "[/")):
        return []

    # An explicitly empty set means "no tools available" — nothing to rescue.
    # None means "skip name validation" (useful in tests / tool-agnostic paths).
    if known_tool_names is not None and not known_tool_names:
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
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                # Bare scalar or array — normalize to object.
                args_str = json.dumps({"input": args})
        except (json.JSONDecodeError, ValueError):
            args = _parse_minimax_kv(body)
            if not args:
                continue
            args_str = json.dumps(args)
        _add(SyntheticToolCall.make(name, args_str))

    # --- Pattern 4: Known tool-name tags ---
    if known:
        # Build a lowercase lookup so case-insensitive tag matches (e.g.
        # <WORKER_DELEGATION>) resolve to the canonical tool name.
        _known_lower: dict[str, str] = {n.lower(): n for n in known}
        pat = _build_tool_name_pattern(known)
        for m in pat.finditer(content):
            # Group 1 = angle-bracket name, Group 2 = square-bracket name
            raw_name = m.group(1) or m.group(2)
            name = _known_lower.get(raw_name.lower(), raw_name)
            body = m.group(3).strip()
            # Could be JSON, key=value, YAML-like, or a single string argument
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
                if not kv:
                    kv = _parse_yaml_like_kv(body)
                if kv:
                    args_str = json.dumps(kv)
                elif body:
                    # Treat as single-string value wrapped in {"input": …}
                    args_str = json.dumps({"input": body})
                else:
                    continue
            _add(SyntheticToolCall.make(name, args_str))

    # --- Pattern 5: Bracket-wrapped [TOOL_CALL] with hash-rocket body ---
    for m in _RE_BRACKET_TOOL_CALL.finditer(content):
        body = m.group(1).strip()
        # Try arrow-style first ({tool => "…", args => {…}})
        parsed = _parse_arrow_style(body, known)
        if not parsed:
            # Fall back to plain JSON inside the brackets
            parsed = _try_parse_json_body(body, known) or []
        for tc in parsed:
            _add(tc)

    if results:
        logger.info(
            "Rescued %d XML-encoded tool call(s) from assistant content: %s",
            len(results),
            [tc.function.name for tc in results],
        )
    return results
