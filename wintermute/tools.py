"""
Tool definitions and execution for the AI assistant.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).
The dispatcher ``execute_tool`` is the single entry point used by the LLM thread.

Tool categories
---------------
  "execution"     – shell, file I/O (available to all sub-session modes)
  "research"      – web search, URL fetching (available to all sub-session modes)
  "orchestration" – memory, reminders, skills, sub-session spawning (main agent
                    and "full"-mode sub-sessions only)
"""

import json
import logging
import os
import subprocess
import time
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from wintermute import database
from wintermute import prompt_assembler

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

SEARXNG_URL = os.environ.get("WINTERMUTE_SEARXNG_URL", "http://127.0.0.1:8888")

# ---------------------------------------------------------------------------
# In-memory tool call log (bounded ring buffer for the debug UI)
# ---------------------------------------------------------------------------

_TOOL_CALL_LOG: deque[dict] = deque(maxlen=500)


def get_tool_call_log() -> list[dict]:
    """Return a copy of the tool call log, newest first."""
    return list(reversed(_TOOL_CALL_LOG))

# Maximum nesting depth for sub-session spawning.
# 0 = main agent, 1 = sub-session, 2 = sub-sub-session (max).
MAX_NESTING_DEPTH = 2

# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas
# ---------------------------------------------------------------------------

def _fn(name: str, description: str, parameters: dict) -> dict:
    """Wrap a function schema in the OpenAI tool envelope."""
    return {
        "type": "function",
        "function": {
            "name":        name,
            "description": description,
            "parameters":  parameters,
        },
    }


TOOL_SCHEMAS = [
    _fn(
        "spawn_sub_session",
        (
            "Spawn an autonomous background worker. Returns a session_id immediately; "
            "result arrives as a [SYSTEM EVENT]."
        ),
        {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Task description. Worker has no conversation access — be specific.",
                },
                "context_blobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Context snippets for the worker. Unneeded when using depends_on.",
                },
                "system_prompt_mode": {
                    "type": "string",
                    "enum": ["minimal", "full", "base_only", "none"],
                    "description": (
                        "'minimal' (default): lightweight agent, no memories/skills. "
                        "'full': complete context (memories + pulse + skills). "
                        "'base_only': core instructions only. "
                        "'none': no system prompt."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds before timeout (default: 300).",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Session IDs to wait for; results auto-passed as context. Prefer depends_on_previous.",
                },
                "depends_on_previous": {
                    "type": "boolean",
                    "description": "Depend on all sessions spawned in this context; avoids manually tracking IDs.",
                },
                "not_before": {
                    "type": "string",
                    "description": "Earliest start time (ISO-8601). Waits even if deps are satisfied.",
                },
            },
            "required": ["objective"],
        },
    ),
    _fn(
        "set_reminder",
        (
            "Schedule a one-time or recurring reminder. With ai_prompt, an autonomous AI "
            "inference runs when it fires; without, only a text notification is sent."
        ),
        {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Reminder text delivered to chat when it fires.",
                },
                "ai_prompt": {
                    "type": "string",
                    "description": (
                        "AI prompt to run when the reminder fires (full tool access). "
                        "Set whenever the user wants an action performed, not just a notification. "
                        "Write as a complete task instruction."
                    ),
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly", "monthly", "interval"],
                    "description": (
                        "once: specific time. daily: fixed time each day. "
                        "weekly: needs day_of_week. monthly: needs day_of_month. "
                        "interval: every N seconds, needs interval_seconds."
                    ),
                },
                "at": {
                    "type": "string",
                    "description": "Required except for interval. For once: ISO-8601 or natural language. For recurring: HH:MM.",
                },
                "day_of_week": {
                    "type": "string",
                    "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                    "description": "Required for weekly.",
                },
                "day_of_month": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 31,
                    "description": "Required for monthly.",
                },
                "interval_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Required for interval. Seconds between firings.",
                },
                "window_start": {
                    "type": "string",
                    "description": "For interval: earliest fire time, HH:MM.",
                },
                "window_end": {
                    "type": "string",
                    "description": "For interval: latest fire time, HH:MM.",
                },
                "system": {
                    "type": "boolean",
                    "description": "Fire as a system event with no chat delivery.",
                },
            },
            "required": ["message", "schedule_type"],
        },
    ),
    _fn(
        "append_memory",
        "Append a fact to MEMORIES.txt. One entry per call",
        {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "Fact or note to append.",
                }
            },
            "required": ["entry"],
        },
    ),
    _fn(
        "pulse",
        "Manage pulse items (working memory for ongoing tasks).",
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "complete", "list", "update"],
                    "description": "add: create item. complete: mark done. list: show items. update: modify existing.",
                },
                "content": {"type": "string", "description": "Item text (for add/update)."},
                "item_id": {"type": "integer", "description": "Item ID (for complete/update)."},
                "priority": {"type": "integer", "description": "1 (urgent) to 10 (low), default 5."},
                "status": {
                    "type": "string",
                    "enum": ["active", "completed", "all"],
                    "description": "For list: filter (default: active). For update: new status value ('active' or 'completed').",
                },
            },
            "required": ["action"],
        },
    ),
    _fn(
        "add_skill",
        "Create or overwrite a skill in data/skills/. Skills are auto-loaded into every system prompt.",
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Filename stem without extension (e.g. 'calendar').",
                },
                "documentation": {
                    "type": "string",
                    "description": "Markdown documentation for the skill. Be concise, max 500 chars.",
                },
            },
            "required": ["skill_name", "documentation"],
        },
    ),
    _fn(
        "execute_shell",
        "Run a shell command. Returns stdout, stderr, and exit_code.",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30).",
                },
            },
            "required": ["command"],
        },
    ),
    _fn(
        "read_file",
        "Read a file and return its contents.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                }
            },
            "required": ["path"],
        },
    ),
    _fn(
        "write_file",
        "Write content to a file, creating parent directories as needed.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write.",
                },
            },
            "required": ["path", "content"],
        },
    ),
    _fn(
        "list_reminders",
        "Returns active, completed, and failed reminders.",
        {"type": "object", "properties": {}},
    ),
    _fn(
        "delete_reminder",
        "Cancel and remove an active reminder by its ID.",
        {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Reminder ID to cancel (e.g. 'reminder_7735fb78').",
                }
            },
            "required": ["job_id"],
        },
    ),
    _fn(
        "search_web",
        "Search the web via SearXNG. Returns titles, URLs, and snippets.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    ),
    _fn(
        "fetch_url",
        "Fetch a web page and return it as plain text (HTML stripped).",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters of content to return (default: 20000).",
                },
            },
            "required": ["url"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool categories — controls which tools sub-sessions receive
# ---------------------------------------------------------------------------

TOOL_CATEGORIES: dict[str, str] = {
    "execute_shell":      "execution",
    "read_file":          "execution",
    "write_file":         "execution",
    "search_web":         "research",
    "fetch_url":          "research",
    "spawn_sub_session":  "orchestration",
    "set_reminder":       "orchestration",
    "append_memory":      "orchestration",
    "pulse":              "orchestration",
    "add_skill":          "orchestration",
    "list_reminders":     "orchestration",
    "delete_reminder":    "orchestration",
}


NL_TOOL_SCHEMAS = [
    _fn(
        "set_reminder",
        (
            "Schedule a one-time or recurring reminder. Describe what you want "
            "in plain English — the system will translate it into structured arguments."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the reminder: what, when, "
                        "how often, and any action to perform when it fires."
                    ),
                },
            },
            "required": ["description"],
        },
    ),
    _fn(
        "spawn_sub_session",
        (
            "Spawn one or more autonomous background workers as a DAG pipeline. "
            "Describe all tasks in plain English, including sequencing and parallel "
            "branches — the system will translate it into a dependency graph. "
            "Examples: 'research X then summarise it' (sequential), "
            "'check weather in Berlin and Tokyo' (parallel), "
            "'fetch A and B, then combine results' (fan-in)."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the full DAG pipeline: what "
                        "each worker should do, which tasks must run before others, "
                        "and which can run in parallel. Be explicit about sequencing "
                        "(e.g. 'then', 'after that', 'at the same time as')."
                    ),
                },
            },
            "required": ["description"],
        },
    ),
]

_NL_SCHEMA_MAP: dict[str, dict] = {
    schema["function"]["name"]: schema for schema in NL_TOOL_SCHEMAS
}


def get_tool_schemas(categories: set[str] | None = None,
                     nl_tools: set[str] | None = None) -> list[dict]:
    """Return tool schemas filtered by category, with optional NL substitution.

    If *categories* is None, return all schemas (used by the main agent).
    Otherwise return only schemas whose tool name maps to one of the
    requested categories.

    When *nl_tools* is provided, any tool whose name is in the set gets its
    schema replaced with the simplified single-field NL variant.
    """
    if categories is None:
        schemas = TOOL_SCHEMAS
    else:
        schemas = [
            schema for schema in TOOL_SCHEMAS
            if TOOL_CATEGORIES.get(schema["function"]["name"]) in categories
        ]
    if not nl_tools:
        return schemas
    result = []
    for schema in schemas:
        name = schema["function"]["name"]
        if name in nl_tools and name in _NL_SCHEMA_MAP:
            result.append(_NL_SCHEMA_MAP[name])
        else:
            result.append(schema)
    return result


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# These will be injected by the scheduler thread at startup.
_scheduler_set_reminder = None    # Callable[[dict], str]
_scheduler_list_reminders = None  # Callable[[], dict]
_scheduler_delete_reminder = None # Callable[[str], bool]

# This will be injected by the SubSessionManager at startup.
_sub_session_spawn = None  # Callable[[dict, str], str]


def register_scheduler(fn) -> None:
    """Called once by the scheduler thread to provide the set_reminder hook."""
    global _scheduler_set_reminder
    _scheduler_set_reminder = fn


def register_reminder_lister(fn) -> None:
    """Called once by the scheduler thread to provide the list_reminders hook."""
    global _scheduler_list_reminders
    _scheduler_list_reminders = fn


def register_reminder_deleter(fn) -> None:
    """Called once by the scheduler thread to provide the delete_reminder hook."""
    global _scheduler_delete_reminder
    _scheduler_delete_reminder = fn


def register_sub_session_manager(fn) -> None:
    """Called once by SubSessionManager to provide the spawn hook."""
    global _sub_session_spawn
    _sub_session_spawn = fn


def _tool_spawn_sub_session(inputs: dict, thread_id: Optional[str] = None,
                            nesting_depth: int = 0, **_kw) -> str:
    if nesting_depth >= MAX_NESTING_DEPTH:
        return json.dumps({
            "error": (
                f"Maximum nesting depth ({MAX_NESTING_DEPTH}) reached. "
                "Cannot spawn further sub-sessions."
            )
        })
    if _sub_session_spawn is None:
        return json.dumps({"error": "Sub-session manager not ready yet."})
    try:
        kwargs = dict(
            objective=inputs["objective"],
            context_blobs=inputs.get("context_blobs") or [],
            parent_thread_id=thread_id,
            system_prompt_mode=inputs.get("system_prompt_mode", "minimal"),
            nesting_depth=nesting_depth + 1,
        )
        if "timeout" in inputs:
            kwargs["timeout"] = int(inputs["timeout"])
        if "depends_on" in inputs:
            kwargs["depends_on"] = inputs["depends_on"]
        if inputs.get("depends_on_previous"):
            kwargs["depends_on_previous"] = True
        if "not_before" in inputs:
            kwargs["not_before"] = inputs["not_before"]
        session_id = _sub_session_spawn(**kwargs)
        has_deps = bool(inputs.get("depends_on")) or bool(inputs.get("depends_on_previous"))
        has_gate = bool(inputs.get("not_before"))
        if has_deps or has_gate:
            reasons = []
            if has_deps:
                reasons.append("dependencies")
            if has_gate:
                reasons.append(f"time gate ({inputs['not_before']})")
            status = f"pending (waiting for {' and '.join(reasons)})"
        else:
            status = "started"
        return json.dumps({
            "status": status,
            "session_id": session_id,
            "IMPORTANT": (
                "The worker is now running in the background. "
                "You do NOT have its results yet. "
                "Tell the user you started the task and STOP. "
                "Do NOT fabricate, guess, or anticipate what the worker will find. "
                "The results will be delivered to you later as a [SYSTEM EVENT]. "
                "Only then should you report them to the user."
            ),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("spawn_sub_session failed")
        return json.dumps({"error": str(exc)})


def _tool_set_reminder(inputs: dict, thread_id: Optional[str] = None, **_kw) -> str:
    if _scheduler_set_reminder is None:
        return json.dumps({"error": "Scheduler not ready yet."})
    try:
        # Pass thread_id through so the scheduler can bind the reminder
        if thread_id and not inputs.get("system"):
            inputs["thread_id"] = thread_id
        job_id = _scheduler_set_reminder(inputs)
        return json.dumps({"status": "scheduled", "job_id": job_id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("set_reminder failed")
        return json.dumps({"error": str(exc)})


def _tool_append_memory(inputs: dict, **_kw) -> str:
    try:
        total_len = prompt_assembler.append_memory(inputs["entry"])
        return json.dumps({"status": "ok", "total_chars": total_len})
    except Exception as exc:  # noqa: BLE001
        logger.exception("append_memory failed")
        return json.dumps({"error": str(exc)})


def _tool_pulse(inputs: dict, thread_id: Optional[str] = None, **_kw) -> str:
    try:
        action = inputs.get("action", "list")
        if action == "add":
            content = inputs.get("content")
            if not content:
                return json.dumps({"error": "content is required for add action"})
            effective_thread_id = inputs.get("thread_id") or thread_id
            item_id = database.add_pulse_item(
                content, priority=int(inputs.get("priority", 5)),
                thread_id=effective_thread_id,
            )
            return json.dumps({"status": "ok", "item_id": item_id})
        elif action == "complete":
            item_id = inputs.get("item_id")
            if item_id is None:
                return json.dumps({"error": "item_id is required for complete action"})
            ok = database.complete_pulse_item(int(item_id))
            return json.dumps({"status": "ok" if ok else "not_found"})
        elif action == "update":
            item_id = inputs.get("item_id")
            if item_id is None:
                return json.dumps({"error": "item_id is required for update action"})
            kwargs = {}
            if "content" in inputs:
                kwargs["content"] = inputs["content"]
            if "priority" in inputs:
                kwargs["priority"] = int(inputs["priority"])
            if "status" in inputs:
                kwargs["status"] = inputs["status"]
            ok = database.update_pulse_item(int(item_id), **kwargs)
            return json.dumps({"status": "ok" if ok else "not_found"})
        elif action == "list":
            status = inputs.get("status", "active")
            items = database.list_pulse_items(status)
            return json.dumps({"items": items, "count": len(items)})
        else:
            return json.dumps({"error": f"Unknown action: {action}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("pulse tool failed")
        return json.dumps({"error": str(exc)})


def _tool_add_skill(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.add_skill(inputs["skill_name"], inputs["documentation"])
        return json.dumps({"status": "ok", "skill": inputs["skill_name"]})
    except Exception as exc:  # noqa: BLE001
        logger.exception("add_skill failed")
        return json.dumps({"error": str(exc)})


def _tool_execute_shell(inputs: dict, **_kw) -> str:
    command = inputs["command"]
    timeout = int(inputs.get("timeout", 30))
    logger.info("execute_shell: %s", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return json.dumps({
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("execute_shell failed")
        return json.dumps({"error": str(exc)})


def _tool_read_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        return json.dumps({"content": path.read_text(encoding="utf-8")})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except OSError as exc:
        return json.dumps({"error": str(exc)})


def _tool_write_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inputs["content"], encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path)})
    except OSError as exc:
        return json.dumps({"error": str(exc)})


def _tool_list_reminders(_inputs: dict, **_kw) -> str:
    if _scheduler_list_reminders is not None:
        return json.dumps(_scheduler_list_reminders(), indent=2, ensure_ascii=False)
    return json.dumps({"active": [], "completed": [], "failed": []})


def _tool_delete_reminder(inputs: dict, **_kw) -> str:
    if _scheduler_delete_reminder is None:
        return json.dumps({"error": "Scheduler not ready yet."})
    try:
        job_id = inputs["job_id"]
        found = _scheduler_delete_reminder(job_id)
        if found:
            return json.dumps({"status": "cancelled", "job_id": job_id})
        return json.dumps({"error": f"Reminder not found: {job_id}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("delete_reminder failed")
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# HTML-to-text helper (stdlib only, no extra dependencies)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter that strips tags and scripts."""

    _IGNORE_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
        elif tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            self._pieces.append(data)

    def get_text(self) -> str:
        import re
        text = "".join(self._pieces)
        # Collapse runs of whitespace but preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# Research tool implementations
# ---------------------------------------------------------------------------

def _tool_search_web(inputs: dict, **_kw) -> str:
    import urllib.parse
    query = inputs["query"]
    max_results = int(inputs.get("max_results", 5))
    logger.info("search_web: %s", query)

    # --- Try SearXNG first ---
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "categories": "general"})
        req = Request(f"{SEARXNG_URL}/search?{params}", headers={"User-Agent": "wintermute/0.1"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
            for item in data.get("results", [])[:max_results]
        ]
        return json.dumps({"query": query, "source": "searxng", "results": results, "count": len(results)})
    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        if not any(s in reason for s in ("Connection refused", "No route to host", "timed out")):
            return json.dumps({"error": f"SearXNG request failed: {reason}"})
        logger.warning("SearXNG unreachable (%s), falling back to curl", reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SearXNG error (%s), falling back to curl", exc)

    # --- Fallback: DuckDuckGo Instant Answer API via curl (no auth required) ---
    try:
        safe_q = urllib.parse.quote_plus(query)
        proc = subprocess.run(
            f'curl -s --max-time 15 -A "wintermute/0.1" '
            f'"https://api.duckduckgo.com/?q={safe_q}&format=json&no_html=1&skip_disambig=1"',
            shell=True, capture_output=True, text=True, timeout=20,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"curl exited {proc.returncode}: {proc.stderr[:200]}")
        data = json.loads(proc.stdout)
        results = []
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({"title": data.get("Heading", query), "url": data["AbstractURL"], "snippet": data["AbstractText"]})
        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if "FirstURL" in topic:
                results.append({"title": topic.get("Text", "")[:80], "url": topic["FirstURL"], "snippet": topic.get("Text", "")})
            elif "Topics" in topic:
                for sub in topic["Topics"]:
                    if len(results) >= max_results:
                        break
                    if "FirstURL" in sub:
                        results.append({"title": sub.get("Text", "")[:80], "url": sub["FirstURL"], "snippet": sub.get("Text", "")})
        return json.dumps({
            "query": query, "source": "duckduckgo_fallback",
            "warning": "SearXNG unavailable. Start it with: cd ~/searxng-test && ./start-searxng.sh",
            "results": results[:max_results], "count": len(results[:max_results]),
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_web fallback failed")
        return json.dumps({"error": f"Both SearXNG and curl fallback failed: {exc}"})


def _tool_fetch_url(inputs: dict, **_kw) -> str:
    url = inputs["url"]
    max_chars = int(inputs.get("max_chars", 20000))
    logger.info("fetch_url: %s", url)

    try:
        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; wintermute/0.1; "
                "+https://github.com/wintermute)"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
        })
        with urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

            # Detect encoding from Content-Type header, fall back to utf-8
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            body = raw.decode(charset, errors="replace")

    except URLError as exc:
        reason = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        return json.dumps({"error": f"Failed to fetch URL: {reason}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_url failed")
        return json.dumps({"error": str(exc)})

    # Strip HTML if it looks like an HTML document.
    if "html" in content_type or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        body = _html_to_text(body)

    if len(body) > max_chars:
        body = body[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"

    return json.dumps({
        "url": url,
        "content_type": content_type,
        "length": len(body),
        "content": body,
    })


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "spawn_sub_session":  _tool_spawn_sub_session,
    "set_reminder":       _tool_set_reminder,
    "append_memory":      _tool_append_memory,
    "pulse":              _tool_pulse,
    "add_skill":          _tool_add_skill,
    "execute_shell":      _tool_execute_shell,
    "read_file":          _tool_read_file,
    "write_file":         _tool_write_file,
    "list_reminders":     _tool_list_reminders,
    "delete_reminder":    _tool_delete_reminder,
    "search_web":         _tool_search_web,
    "fetch_url":          _tool_fetch_url,
}


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 nesting_depth: int = 0) -> str:
    """Execute a tool by name and return its JSON-string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    logger.debug("Executing tool '%s' with inputs: %s", name, inputs)
    t0 = time.monotonic()
    try:
        result = fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth)
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
