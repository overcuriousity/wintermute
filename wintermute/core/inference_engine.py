"""Shared tool-call execution pipeline.

Extracted from ``LLMThread._inference_loop`` and
``SubSessionManager._worker_loop`` to eliminate the duplicated per-tool-call
dispatch logic (~150 lines each).

Callers retain their own outer inference loops — which differ in message
management, context trimming, and terminal-response handling — but delegate
the per-tool-call pipeline to the shared :func:`process_tool_call` function.

Pipeline per tool call
----------------------
1. Parse JSON arguments (→ error on failure)
2. NL translation expansion (single-item or multi-item)
3. Turing Protocol ``pre_execution`` gate
4. Tool execution via ``run_in_executor``
5. NL translation summary prefix
6. Turing Protocol ``post_execution`` annotation
7. Interaction logging + event-bus emission
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from wintermute.core import nl_translator
from wintermute.infra import database
from wintermute import tools as tool_module

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback type for Turing Protocol pre/post-execution checks.
#
# Callers provide a closure that delegates to their own TP runner.
# Signature:
#   async (phase, tool_name, tool_args, tool_result,
#          assistant_response, tool_calls_made, nl_tools) -> result | None
# ---------------------------------------------------------------------------
TPCheckFn = Callable[..., Awaitable[Any]]


# ---------------------------------------------------------------------------
# Context & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCallContext:
    """Parameters for tool-call execution that vary between callers."""

    thread_id: str
    nesting_depth: int = 0
    parent_thread_id: Optional[str] = None
    scope: str = "main"           # "main" or "sub_session"
    pool_last_used: str = ""      # backend name, for interaction logging

    event_bus: Optional[Any] = None

    # NL translation
    nl_enabled: bool = False
    nl_tools: Optional[set] = None
    nl_translation_pool: Optional[Any] = None
    timezone_str: str = ""

    # Turing Protocol
    tp_enabled: bool = False
    tp_check: Optional[TPCheckFn] = None


@dataclass
class ToolCallOutcome:
    """Result of processing a single tool call."""

    content: str                                            # tool result text for messages
    tool_name: str
    raw_arguments: str                                      # original JSON from the model
    executed: bool = True                                   # False → parse/NL/TP error
    calls_made: list[str] = field(default_factory=list)     # tool names actually executed
    call_details: list[dict] = field(default_factory=list)  # [{name, arguments, result}]


# ---------------------------------------------------------------------------
# Internal: NL translation helpers
# ---------------------------------------------------------------------------

@dataclass
class _NLResult:
    """Internal result from NL translation attempt."""
    translated: Optional[dict] = None       # single-item translation
    multi_items: Optional[list] = None      # multi-item translation
    error: Optional[str] = None             # error / clarification message


async def _translate_nl(
    name: str,
    inputs: dict,
    ctx: ToolCallContext,
) -> _NLResult:
    """Run NL translation and log the call."""
    translated = await nl_translator.translate_nl_tool_call(
        ctx.nl_translation_pool, name, inputs["description"],
        thread_id=ctx.thread_id,
        timezone_str=ctx.timezone_str,
    )
    # Log the NL translation.
    try:
        await database.async_call(
            database.save_interaction_log,
            _time.time(), "nl_translation", ctx.thread_id,
            (ctx.nl_translation_pool.last_used
             if ctx.nl_translation_pool else ""),
            inputs["description"],
            json.dumps(translated) if translated is not None else "null",
            "ok" if translated is not None else "error",
        )
    except Exception:
        pass

    if translated is None:
        return _NLResult(error=(
            "[TRANSLATION ERROR] Failed to translate natural-language "
            "tool call. Please try rephrasing or use structured arguments."
        ))

    if isinstance(translated, dict) and "error" in translated:
        return _NLResult(error=(
            f"[CLARIFICATION NEEDED] "
            f"{translated.get('clarification_needed', translated['error'])}"
        ))

    if isinstance(translated, list):
        # Merge orphan metadata items (no required field) into
        # the preceding item.  Handles NL translator emitting
        # depends_on / not_before as separate array elements.
        merged: list[dict] = []
        for item in translated:
            if (merged
                    and "objective" not in item
                    and name == "spawn_sub_session"):
                merged[-1].update(item)
            else:
                merged.append(item)
        return _NLResult(multi_items=merged)

    return _NLResult(translated=translated)


async def _execute_multi_item(
    items: list[dict],
    name: str,
    raw_arguments: str,
    ctx: ToolCallContext,
    tool_calls_made: list[str],
) -> ToolCallOutcome:
    """Execute a multi-item NL-translated call, combining results."""
    combined: list[str] = []
    calls_made: list[str] = []
    call_details: list[dict] = []

    for i, item_args in enumerate(items):
        item_result = await asyncio.get_running_loop().run_in_executor(
            None, lambda _n=name, _a=item_args: tool_module.execute_tool(
                _n, _a,
                thread_id=ctx.thread_id,
                nesting_depth=ctx.nesting_depth,
                parent_thread_id=ctx.parent_thread_id,
            ),
        )
        summary = ", ".join(
            f"{k}={v!r}" for k, v in item_args.items() if k != "description"
        )
        combined.append(f"[{i + 1}] [Translated to: {summary}] {item_result}")
        tool_calls_made.append(name)
        calls_made.append(name)
        call_details.append({
            "name": name,
            "arguments": json.dumps(item_args),
            "result": item_result,
        })

    return ToolCallOutcome(
        content="\n\n".join(combined),
        tool_name=name,
        raw_arguments=raw_arguments,
        executed=True,
        calls_made=calls_made,
        call_details=call_details,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def process_tool_call(
    tc: Any,
    ctx: ToolCallContext,
    tool_calls_made: list[str],
    assistant_response: str = "",
) -> ToolCallOutcome:
    """Process a single tool call through the full pipeline.

    ``tool_calls_made`` is mutated in-place (tool names appended on
    execution).  Returns a :class:`ToolCallOutcome`; callers handle
    message placement in their own way.
    """
    name = tc.function.name
    raw_args = tc.function.arguments

    # -- Step 1: Parse JSON arguments --------------------------------
    try:
        inputs = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Malformed tool args for %s in %s (id=%s): %s — raw: %s",
            name, ctx.thread_id, tc.id, exc, raw_args[:500],
        )
        return ToolCallOutcome(
            content=(
                f"[ERROR] Could not parse arguments for tool "
                f"'{name}': {exc}. "
                f"Please retry with valid JSON arguments."
            ),
            tool_name=name,
            raw_arguments=raw_args,
            executed=False,
        )

    nl_was_translated = False

    # -- Step 2: NL Translation --------------------------------------
    if ctx.nl_enabled and nl_translator.is_nl_tool_call(name, inputs):
        nl_result = await _translate_nl(name, inputs, ctx)

        if nl_result.error is not None:
            return ToolCallOutcome(
                content=nl_result.error,
                tool_name=name,
                raw_arguments=raw_args,
                executed=False,
            )

        if nl_result.multi_items is not None:
            return await _execute_multi_item(
                nl_result.multi_items, name, raw_args, ctx, tool_calls_made,
            )

        # Single-item translation — update inputs and continue.
        if nl_result.translated is not None:
            inputs = nl_result.translated
            nl_was_translated = True

    # -- Step 3: Turing Protocol pre_execution -----------------------
    if ctx.tp_enabled and ctx.tp_check:
        pre_result = await ctx.tp_check(
            "pre_execution",
            tool_name=name,
            tool_args=inputs,
            tool_result=None,
            assistant_response=assistant_response,
            tool_calls_made=tool_calls_made,
            nl_tools=ctx.nl_tools,
        )
        if pre_result and pre_result.correction:
            logger.warning(
                "[%s] pre_execution hook blocked tool %s: %s",
                ctx.thread_id, name, pre_result.correction[:200],
            )
            return ToolCallOutcome(
                content=f"[BLOCKED BY TURING PROTOCOL] {pre_result.correction}",
                tool_name=name,
                raw_arguments=raw_args,
                executed=False,
            )

    # -- Step 4: Execute tool ----------------------------------------
    result = await asyncio.get_running_loop().run_in_executor(
        None, lambda _n=name, _i=inputs: tool_module.execute_tool(
            _n, _i,
            thread_id=ctx.thread_id,
            nesting_depth=ctx.nesting_depth,
            parent_thread_id=ctx.parent_thread_id,
        ),
    )
    tool_calls_made.append(name)

    # -- Step 5: NL summary ------------------------------------------
    if nl_was_translated:
        summary = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
        result = f"[Translated to: {summary}] {result}"

    # -- Step 6: Turing Protocol post_execution ----------------------
    if ctx.tp_enabled and ctx.tp_check:
        post_result = await ctx.tp_check(
            "post_execution",
            tool_name=name,
            tool_args=None,
            tool_result=result,
            assistant_response=assistant_response,
            tool_calls_made=tool_calls_made,
            nl_tools=ctx.nl_tools,
        )
        if post_result and post_result.correction:
            result += f"\n\n[TURING PROTOCOL WARNING] {post_result.correction}"

    # -- Step 7: Logging & events ------------------------------------
    if ctx.event_bus:
        ctx.event_bus.emit(
            "tool.executed", tool=name,
            thread_id=ctx.thread_id, scope=ctx.scope,
        )
    try:
        await database.async_call(
            database.save_interaction_log,
            _time.time(), "tool_call", ctx.thread_id,
            ctx.pool_last_used,
            json.dumps({"tool": name, "arguments": raw_args}),
            result[:500], "ok",
        )
    except Exception:
        pass

    logger.debug("[%s] Tool %s -> %s", ctx.thread_id, name, result[:200])

    return ToolCallOutcome(
        content=result,
        tool_name=name,
        raw_arguments=raw_args,
        executed=True,
        calls_made=[name],
        call_details=[{"name": name, "arguments": raw_args, "result": result}],
    )
