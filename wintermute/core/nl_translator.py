"""
Natural-Language Tool Call Translator.

Small/weak LLMs frequently produce malformed arguments for complex tools
like ``task`` (many properties) and ``spawn_sub_session`` (DAG semantics).
This module presents those tools as single-field "describe in English"
schemas to the main LLM, then uses a dedicated translator LLM to expand
the description into structured arguments.

Complementary to the existing ``tool_schema_validation`` Turing Protocol
hook — the Turing hook validates the *translated* args, not the raw
description.
"""

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from zoneinfo import ZoneInfo

from wintermute.infra.llm_utils import strip_fences

from wintermute.infra import prompt_loader

if TYPE_CHECKING:
    from wintermute.core.llm_thread import BackendPool

logger = logging.getLogger(__name__)

# Tools that have NL translation variants.
NL_TOOLS: frozenset[str] = frozenset({"task", "spawn_sub_session", "add_skill"})

# Maps tool name -> prompt template filename.
_PROMPT_MAP: dict[str, str] = {
    "task": "NL_TRANSLATOR_TASK.txt",
    "spawn_sub_session": "NL_TRANSLATOR_SPAWN_SUB_SESSION.txt",
    "add_skill": "NL_TRANSLATOR_ADD_SKILL.txt",
}


_TASK_LIST_PATTERNS: dict[str, "dict | None"] = {
    # Maps normalised description → structured args.
    # Value None means "default list" (active items).
    "list": None,
    "show": None,
    "show items": None,
    "list items": None,
    "show tasks": None,
    "list tasks": None,
    "show active": None,
    "list active": None,
    "show active tasks": None,
    "list active tasks": None,
    "show all active tasks": None,
    "show me all active tasks": None,
    "show me all tasks": {"action": "list", "status": "all"},
    "list all": {"action": "list", "status": "all"},
    "show all": {"action": "list", "status": "all"},
    "show all tasks": {"action": "list", "status": "all"},
    "list all tasks": {"action": "list", "status": "all"},
    "show everything": {"action": "list", "status": "all"},
    "list everything": {"action": "list", "status": "all"},
    "show completed": {"action": "list", "status": "completed"},
    "list completed": {"action": "list", "status": "completed"},
    "show completed tasks": {"action": "list", "status": "completed"},
    "list completed tasks": {"action": "list", "status": "completed"},
    "show paused": {"action": "list", "status": "paused"},
    "list paused": {"action": "list", "status": "paused"},
}

_TASK_LIST_DEFAULT: dict = {"action": "list"}


def _task_list_fastpath(description: str) -> "dict | None":
    """Return structured args if *description* matches a known task-list
    pattern, or ``None`` to fall through to the LLM translator."""
    normalised = re.sub(r"[.!?]+$", "", description.strip()).strip().lower()
    result = _TASK_LIST_PATTERNS.get(normalised, _SENTINEL)
    if result is _SENTINEL:
        return None
    return result if result is not None else _TASK_LIST_DEFAULT.copy()


_SENTINEL = object()


def is_nl_tool_call(tool_name: str, tool_args: dict) -> bool:
    """Detect whether a tool call uses the simplified NL schema.

    Returns True when the tool is NL-eligible and has a ``"description"``
    string key — even if the LLM also included extra keys (e.g. ``timeout``)
    alongside it.  The translator will use only the description; extra keys
    are discarded.
    """
    return (
        tool_name in NL_TOOLS
        and "description" in tool_args
        and isinstance(tool_args["description"], str)
    )



def _fix_unescaped_control_chars(text: str) -> str:
    """Escape literal control characters inside JSON string values.

    Weak models sometimes emit raw newlines/tabs inside string values, which
    is invalid per RFC 8259.  This walks the JSON character-by-character and
    replaces bare control characters found inside strings with their \\x
    equivalents, leaving the structural JSON (braces, brackets, commas) alone.
    """
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and in_string:
            # Pass through escape sequence intact.
            result.append(c)
            i += 1
            if i < len(text):
                result.append(text[i])
                i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif in_string and c == "\n":
            result.append("\\n")
        elif in_string and c == "\r":
            result.append("\\r")
        elif in_string and c == "\t":
            result.append("\\t")
        else:
            result.append(c)
        i += 1
    return "".join(result)


async def translate_nl_tool_call(
    pool: "BackendPool",
    tool_name: str,
    description: str,
    thread_id: Optional[str] = None,
    timezone_str: str = "UTC",
    current_datetime: Optional[str] = None,
) -> "dict | list | None":
    """Call the translator LLM to expand a natural-language description
    into structured tool arguments.

    Returns:
      - ``dict``  — single tool call arguments
      - ``list[dict]`` — multiple calls (multi-task / multi-spawn)
      - ``None`` — translation failed (LLM error, unparseable output)

    A dict with an ``"error"`` key signals ambiguity that needs user
    clarification.
    """
    # Fast-path: task list patterns bypass the LLM entirely.
    if tool_name == "task":
        fast = _task_list_fastpath(description)
        if fast is not None:
            logger.debug("NL translator fast-path for task: %r -> %s", description, fast)
            return fast

    prompt_file = _PROMPT_MAP.get(tool_name)
    if not prompt_file:
        logger.error("No NL translator prompt for tool %s", tool_name)
        return None

    try:
        system_prompt = prompt_loader.load(prompt_file)
    except FileNotFoundError:
        logger.error("NL translator prompt file missing: %s", prompt_file)
        return None

    # Build timezone-aware context prefix for the user message.
    if current_datetime is None:
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            tz = ZoneInfo("UTC")
        now = datetime.now(tz)
        current_datetime = now.strftime("%A, %Y-%m-%d %H:%M %Z") + f" ({timezone_str})"

    user_content = (
        f"[Current time: {current_datetime}. "
        f"All times without explicit timezone should use this timezone.]\n\n"
        + description
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await pool.call(messages=messages)
        if not response.choices:
            logger.warning("NL translator: LLM returned empty choices for %s", tool_name)
            logger.debug("Empty choices raw response: %s", response)
            return None
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("NL translator LLM call failed for %s", tool_name)
        return None

    if not raw:
        logger.warning("NL translator returned empty response for %s", tool_name)
        return None

    raw = strip_fences(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to fix literal control chars inside string values.
        fixed = _fix_unescaped_control_chars(raw)
        try:
            parsed = json.loads(fixed)
            logger.warning(
                "NL translator: repaired unescaped control chars in JSON for %s "
                "(lossy — newlines/tabs in values converted to escape sequences): %s",
                tool_name, raw[:300],
            )
        except json.JSONDecodeError:
            logger.warning("NL translator returned invalid JSON for %s: %s", tool_name, raw[:200])
            return None

    # Accept dict (single call) or list (multi-call).
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed

    logger.warning("NL translator returned unexpected type for %s: %s", tool_name, type(parsed).__name__)
    return None
