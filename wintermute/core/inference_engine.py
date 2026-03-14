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
3. Convergence Protocol ``pre_execution`` gate
4. Tool execution via ``execute_syscall`` + ``run_in_executor``
5. NL translation summary prefix
6. Convergence Protocol ``post_execution`` annotation
7. Interaction logging + event-bus emission
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from wintermute.core import nl_translator
from wintermute.core.tool_deps import ToolDeps
from wintermute.infra import database
from wintermute import tools as tool_module
from wintermute.tools.syscall import SyscallRequest

logger = logging.getLogger(__name__)


def normalize_message(message: Any) -> dict:
    """Convert a ChatCompletionMessage to a plain dict.

    Supports Pydantic models (``model_dump``), dataclass objects
    (``dataclasses.asdict``), and plain dicts (returned as-is).  This
    ensures message lists stay homogeneous ``list[dict]`` so downstream
    code never needs isinstance dispatch.
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True, exclude_unset=True)
    if dataclasses.is_dataclass(message) and not isinstance(message, type):
        return dataclasses.asdict(message)
    return dict(message)


def extract_content_text(msg: dict) -> str:
    """Safely extract text content from a normalized message dict.

    Handles ``content`` being a string, a list of content parts (multimodal
    messages), or ``None``.  Always returns a stripped string.
    """
    raw = msg.get("content")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()
    return str(raw).strip()


# ---------------------------------------------------------------------------
# Callback type for Convergence Protocol pre/post-execution checks.
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

    # Convergence Protocol
    tp_enabled: bool = False
    tp_check: Optional[TPCheckFn] = None

    # Per-result tool output truncation (0 = no limit)
    max_tool_output_chars: int = 0

    # Dependency container for tool execution.
    tool_deps: Optional[ToolDeps] = None


def make_tool_context(
    *,
    thread_id: str,
    scope: str,
    pool_last_used: str,
    event_bus: Any = None,
    nesting_depth: int = 0,
    parent_thread_id: Optional[str] = None,
    nl_enabled: bool = False,
    nl_tools: set[str] | None = None,
    nl_translation_pool: Any = None,
    timezone_str: str = "",
    tp_enabled: bool = False,
    tp_check: Optional[TPCheckFn] = None,
    max_tool_output_chars: int = 0,
    tool_deps: Optional[ToolDeps] = None,
) -> ToolCallContext:
    """Create a ToolCallContext — single factory used by both inference loops."""
    return ToolCallContext(
        thread_id=thread_id,
        nesting_depth=nesting_depth,
        parent_thread_id=parent_thread_id,
        scope=scope,
        pool_last_used=pool_last_used,
        event_bus=event_bus,
        nl_enabled=nl_enabled,
        nl_tools=nl_tools,
        nl_translation_pool=nl_translation_pool,
        timezone_str=timezone_str,
        tp_enabled=tp_enabled,
        tp_check=tp_check,
        max_tool_output_chars=max_tool_output_chars,
        tool_deps=tool_deps,
    )


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
        tool_profiles=ctx.tool_deps.tool_profiles if ctx.tool_deps else None,
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
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log NL translation", exc_info=True)

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
                    and name == "worker_delegation"):
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
        req = SyscallRequest(
            name=name, inputs=item_args,
            thread_id=ctx.thread_id,
            nesting_depth=ctx.nesting_depth,
            parent_thread_id=ctx.parent_thread_id,
            tool_deps=ctx.tool_deps,
        )
        sc_result = await asyncio.get_running_loop().run_in_executor(
            None, lambda _r=req: tool_module.execute_syscall(_r),
        )
        item_result = sc_result.data
        # Truncate individual item results (same logic as Step 4b).
        if ctx.max_tool_output_chars and len(item_result) > ctx.max_tool_output_chars:
            orig = len(item_result)
            notice = (
                f"\n\n[...truncated — {orig - ctx.max_tool_output_chars:,}"
                f" chars omitted (total {orig:,} chars)]"
            )
            keep = max(0, ctx.max_tool_output_chars - len(notice))
            item_result = item_result[:keep] + notice
            logger.info(
                "[%s] Truncated multi-item %s[%d] output: %d → %d chars",
                ctx.thread_id, name, i, orig, len(item_result),
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

    # Log once for the entire multi-item call (early return bypasses outer log).
    combined_content = "\n\n".join(combined)

    # Cap the combined result so N items don't blow the context budget.
    if ctx.max_tool_output_chars and len(combined_content) > ctx.max_tool_output_chars:
        original_len = len(combined_content)
        notice = (
            f"\n\n[...truncated combined output — "
            f"{original_len - ctx.max_tool_output_chars:,} chars omitted"
            f" (total {original_len:,} chars)]"
        )
        keep = max(0, ctx.max_tool_output_chars - len(notice))
        combined_content = combined_content[:keep] + notice
        logger.info(
            "[%s] Truncated combined multi-item %s output: %d → %d chars",
            ctx.thread_id, name, original_len, len(combined_content),
        )

    try:
        await database.async_call(
            database.save_interaction_log,
            _time.time(), "tool_call", ctx.thread_id,
            ctx.pool_last_used,
            json.dumps({"tool": name, "arguments": raw_arguments}),
            combined_content[:500], "ok",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log multi-item tool call", exc_info=True)

    return ToolCallOutcome(
        content=combined_content,
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
    # Support both Pydantic objects (legacy) and plain dicts (normalized).
    if isinstance(tc, dict):
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]
        tc_id = tc["id"]
    else:
        name = tc.function.name
        raw_args = tc.function.arguments
        tc_id = tc.id

    # -- Step 1: Parse JSON arguments --------------------------------
    try:
        inputs = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Malformed tool args for %s in %s (id=%s): %s — raw: %s",
            name, ctx.thread_id, tc_id, exc, raw_args[:500],
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

    # -- Step 3: Convergence Protocol pre_execution -----------------------
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
                content=f"[BLOCKED BY CONVERGENCE PROTOCOL] {pre_result.correction}",
                tool_name=name,
                raw_arguments=raw_args,
                executed=False,
            )

    # -- Step 4: Execute tool (syscall) --------------------------------
    req = SyscallRequest(
        name=name, inputs=inputs,
        thread_id=ctx.thread_id,
        nesting_depth=ctx.nesting_depth,
        parent_thread_id=ctx.parent_thread_id,
        tool_deps=ctx.tool_deps,
    )
    sc_result = await asyncio.get_running_loop().run_in_executor(
        None, lambda _r=req: tool_module.execute_syscall(_r),
    )
    result = sc_result.data
    tool_calls_made.append(name)

    # -- Step 4b: Truncate oversized output ---------------------------
    if ctx.max_tool_output_chars and len(result) > ctx.max_tool_output_chars:
        original_len = len(result)
        notice = (
            f"\n\n[...truncated — {original_len - ctx.max_tool_output_chars:,}"
            f" chars omitted (total {original_len:,} chars)]"
        )
        keep = max(0, ctx.max_tool_output_chars - len(notice))
        result = result[:keep] + notice
        logger.info(
            "[%s] Truncated %s output: %d → %d chars",
            ctx.thread_id, name, original_len, len(result),
        )

    # -- Step 5: NL summary ------------------------------------------
    if nl_was_translated:
        summary = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
        result = f"[Translated to: {summary}] {result}"

    # -- Step 6: Convergence Protocol post_execution ----------------------
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
            result += f"\n\n[CONVERGENCE PROTOCOL WARNING] {post_result.correction}"

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
    except Exception:  # noqa: BLE001
        logger.debug("Failed to log tool call %s", name, exc_info=True)

    logger.debug("[%s] Tool %s -> %s", ctx.thread_id, name, result[:200])

    return ToolCallOutcome(
        content=result,
        tool_name=name,
        raw_arguments=raw_args,
        executed=True,
        calls_made=[name],
        call_details=[{"name": name, "arguments": raw_args, "result": result}],
    )
