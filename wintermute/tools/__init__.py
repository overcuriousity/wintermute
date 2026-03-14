"""
Tool definitions and execution for the AI runtime environment.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).
The syscall interface (``execute_syscall``) is the primary entry point;
``execute_tool`` is a thin backwards-compatible wrapper.

Tool categories
---------------
  "execution"     – shell, file I/O (available to all sub-session modes)
  "research"      – web search, URL fetching (available to all sub-session modes)
  "orchestration" – memory, tasks, skills, sub-session spawning (main agent
                    and "full"-mode sub-sessions only)
"""

import json
import logging
import time
from collections import deque
from typing import Any, Optional

from wintermute.core.tool_deps import ToolDeps
from wintermute.core.tool_schemas import (  # noqa: F401 — re-exported
    TOOL_SCHEMAS,
    TOOL_CATEGORIES,
    NL_TOOL_SCHEMAS,
    NL_SCHEMA_MAP,
    get_tool_schemas,
)
from wintermute.tools.syscall import SyscallRequest, SyscallResult  # noqa: F401 — re-exported
from wintermute.tools.task_tools import tool_task, _describe_schedule  # noqa: F401
from wintermute.tools.memory_tools import tool_append_memory, tool_skill
from wintermute.tools.io_tools import tool_execute_shell, tool_read_file, tool_write_file
from wintermute.tools.web_tools import tool_search_web, tool_fetch_url
from wintermute.tools.session_tools import tool_worker_delegation, tool_query_telemetry

logger = logging.getLogger(__name__)

# Backwards-compat alias (was private, now public in tool_schemas).
_NL_SCHEMA_MAP = NL_SCHEMA_MAP

# ---------------------------------------------------------------------------
# In-memory tool call log (bounded ring buffer for the debug UI)
# ---------------------------------------------------------------------------

_TOOL_CALL_LOG: deque[dict] = deque(maxlen=500)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "worker_delegation":  tool_worker_delegation,
    "task":               tool_task,
    "append_memory":      tool_append_memory,
    "skill":              tool_skill,
    "execute_shell":      tool_execute_shell,
    "read_file":          tool_read_file,
    "write_file":         tool_write_file,
    "search_web":         tool_search_web,
    "fetch_url":          tool_fetch_url,
    "query_telemetry":    tool_query_telemetry,
}

# Reverse map: tool name → category (built from TOOL_CATEGORIES).
_TOOL_CATEGORY: dict[str, str] = {}
for _cat, _names in TOOL_CATEGORIES.items():
    for _n in _names:
        _TOOL_CATEGORY[_n] = _cat


def execute_syscall(request: SyscallRequest) -> SyscallResult:
    """Execute a tool via the structured syscall interface.

    Performs a category permission check when ``request.allowed_categories``
    is set, then dispatches to the underlying tool function.
    """
    name = request.name
    category = _TOOL_CATEGORY.get(name, "unknown")

    # Category permission gate.
    if request.allowed_categories is not None and category not in request.allowed_categories:
        error_msg = f"Tool '{name}' (category '{category}') not permitted"
        return SyscallResult(
            success=False,
            data=json.dumps({"error": error_msg}),
            error=error_msg,
            tool_name=name,
            duration_ms=0.0,
            category=category,
        )

    fn = _DISPATCH.get(name)
    if fn is None:
        error_msg = f"Unknown tool: {name}"
        return SyscallResult(
            success=False,
            data=json.dumps({"error": error_msg}),
            error=error_msg,
            tool_name=name,
            duration_ms=0.0,
            category=category,
        )

    logger.debug("Executing syscall '%s' with inputs: %s", name, request.inputs)
    t0 = time.monotonic()
    error_flag: str | None = None
    try:
        result = fn(
            request.inputs,
            thread_id=request.thread_id,
            nesting_depth=request.nesting_depth,
            parent_thread_id=request.parent_thread_id,
            tool_deps=request.tool_deps,
        )
    except (KeyError, TypeError, ValueError) as exc:
        result = json.dumps({"error": f"Tool '{name}' called with invalid arguments: {exc}"})
        error_flag = str(exc)
        logger.warning("Syscall '%s' raised %s: %s", name, type(exc).__name__, exc)

    duration_ms = round((time.monotonic() - t0) * 1000, 1)

    _TOOL_CALL_LOG.append({
        "ts": time.time(),
        "tool": name,
        "inputs": request.inputs,
        "thread_id": request.thread_id or "unknown",
        "result_preview": result[:300] if result else "",
        "duration_ms": duration_ms,
        "error": error_flag,
    })

    return SyscallResult(
        success=error_flag is None,
        data=result,
        error=error_flag,
        tool_name=name,
        duration_ms=duration_ms,
        category=category,
    )


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 nesting_depth: int = 0, parent_thread_id: Optional[str] = None,
                 tool_deps: Optional[ToolDeps] = None) -> str:
    """Execute a tool by name and return its JSON-string result.

    Thin wrapper around :func:`execute_syscall` for backwards compatibility.
    """
    request = SyscallRequest(
        name=name,
        inputs=inputs,
        thread_id=thread_id,
        nesting_depth=nesting_depth,
        parent_thread_id=parent_thread_id,
        tool_deps=tool_deps,
    )
    result = execute_syscall(request)
    return result.data
