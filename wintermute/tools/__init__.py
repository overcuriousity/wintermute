"""
Tool definitions and execution for the AI assistant.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).
The dispatcher ``execute_tool`` is the single entry point used by the LLM thread.

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
from wintermute.tools.task_tools import tool_task, _describe_schedule
from wintermute.tools.memory_tools import tool_append_memory, tool_add_skill
from wintermute.tools.io_tools import tool_execute_shell, tool_read_file, tool_write_file
from wintermute.tools.web_tools import tool_search_web, tool_fetch_url
from wintermute.tools.session_tools import tool_spawn_sub_session, tool_query_telemetry

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
    "spawn_sub_session":  tool_spawn_sub_session,
    "task":               tool_task,
    "append_memory":      tool_append_memory,
    "add_skill":          tool_add_skill,
    "execute_shell":      tool_execute_shell,
    "read_file":          tool_read_file,
    "write_file":         tool_write_file,
    "search_web":         tool_search_web,
    "fetch_url":          tool_fetch_url,
    "query_telemetry":    tool_query_telemetry,
}


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 nesting_depth: int = 0, parent_thread_id: Optional[str] = None,
                 tool_deps: Optional[ToolDeps] = None) -> str:
    """Execute a tool by name and return its JSON-string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    logger.debug("Executing tool '%s' with inputs: %s", name, inputs)
    t0 = time.monotonic()
    try:
        result = fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth,
                    parent_thread_id=parent_thread_id, tool_deps=tool_deps)
        error_flag = None
    except (KeyError, TypeError, ValueError) as exc:
        result = json.dumps({"error": f"Tool '{name}' called with invalid arguments: {exc}"})
        error_flag = str(exc)
        logger.warning("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
    duration_ms = round((time.monotonic() - t0) * 1000, 1)
    _TOOL_CALL_LOG.append({
        "ts": time.time(),
        "tool": name,
        "inputs": inputs,
        "thread_id": thread_id or "unknown",
        "result_preview": result[:300] if result else "",
        "duration_ms": duration_ms,
        "error": error_flag,
    })
    return result
