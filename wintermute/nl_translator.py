"""
Natural-Language Tool Call Translator.

Small/weak LLMs frequently produce malformed arguments for complex tools
like ``set_reminder`` (11 properties) and ``spawn_sub_session`` (DAG
semantics).  This module presents those tools as single-field "describe
in English" schemas to the main LLM, then uses a dedicated translator
LLM to expand the description into structured arguments.

Complementary to the existing ``tool_schema_validation`` Turing Protocol
hook — the Turing hook validates the *translated* args, not the raw
description.
"""

import json
import logging
import re
from typing import Optional

from wintermute import prompt_loader
from wintermute.llm_thread import BackendPool

logger = logging.getLogger(__name__)

# Tools that have NL translation variants.
NL_TOOLS: frozenset[str] = frozenset({"set_reminder", "spawn_sub_session"})

# Maps tool name -> prompt template filename.
_PROMPT_MAP: dict[str, str] = {
    "set_reminder": "NL_TRANSLATOR_SET_REMINDER.txt",
    "spawn_sub_session": "NL_TRANSLATOR_SPAWN_SUB_SESSION.txt",
}


def is_nl_tool_call(tool_name: str, tool_args: dict) -> bool:
    """Detect whether a tool call uses the simplified NL schema.

    Returns True when the tool is NL-eligible and the only argument key
    is ``"description"``.
    """
    return (
        tool_name in NL_TOOLS
        and set(tool_args.keys()) == {"description"}
        and isinstance(tool_args.get("description"), str)
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


async def translate_nl_tool_call(
    pool: BackendPool,
    tool_name: str,
    description: str,
    thread_id: Optional[str] = None,
) -> "dict | list | None":
    """Call the translator LLM to expand a natural-language description
    into structured tool arguments.

    Returns:
      - ``dict``  — single tool call arguments
      - ``list[dict]`` — multiple calls (multi-reminder / multi-spawn)
      - ``None`` — translation failed (LLM error, unparseable output)

    A dict with an ``"error"`` key signals ambiguity that needs user
    clarification.
    """
    prompt_file = _PROMPT_MAP.get(tool_name)
    if not prompt_file:
        logger.error("No NL translator prompt for tool %s", tool_name)
        return None

    try:
        system_prompt = prompt_loader.load(prompt_file)
    except FileNotFoundError:
        logger.error("NL translator prompt file missing: %s", prompt_file)
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": description},
    ]

    try:
        response = await pool.call(messages=messages)
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("NL translator LLM call failed for %s", tool_name)
        return None

    if not raw:
        logger.warning("NL translator returned empty response for %s", tool_name)
        return None

    raw = _strip_markdown_fences(raw)

    try:
        parsed = json.loads(raw)
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
