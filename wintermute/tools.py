"""
Tool definitions and execution for the AI assistant.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).
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
            "the result arrives later as a [SYSTEM EVENT]."
        ),
        {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": (
                        "Full task description. Be specific — the worker has "
                        "no access to the current conversation."
                    ),
                },
                "context_blobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Manual context snippets for the worker. Not needed when "
                        "using depends_on (dependency results are passed automatically)."
                    ),
                },
                "system_prompt_mode": {
                    "type": "string",
                    "enum": ["minimal", "full", "base_only", "none"],
                    "description": (
                        "Worker context level. "
                        "'minimal' (default) — execution agent. "
                        "'full' — includes memories, pulse, skills. "
                        "'base_only' — core instructions only. "
                        "'none' — bare tool loop."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds before timeout. Default 300.",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Session IDs that must complete first. Their results are "
                        "auto-passed as context to this worker. "
                        "Prefer depends_on_previous over manually listing IDs."
                    ),
                },
                "depends_on_previous": {
                    "type": "boolean",
                    "description": (
                        "If true, automatically depend on ALL sessions you have "
                        "spawned so far in this worker context. Their results are "
                        "auto-passed as context. Use this instead of manually "
                        "listing session IDs in depends_on to avoid errors."
                    ),
                },
                "not_before": {
                    "type": "string",
                    "description": (
                        "Earliest datetime to start this task (ISO-8601). "
                        "Task waits even if dependencies are satisfied. "
                        "Use for time-gated workflows, e.g. 'upload after 20:00'."
                    ),
                },
            },
            "required": ["objective"],
        },
    ),
    _fn(
        "set_reminder",
        "Schedule a one-time or recurring reminder.",
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
                        "If set, a background AI inference runs with this prompt "
                        "when the reminder fires. The result is not sent to chat."
                    ),
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly", "monthly", "interval"],
                    "description": (
                        "'once' — fire at a specific time. "
                        "'daily' — every day at a fixed time. "
                        "'weekly' — requires day_of_week. "
                        "'monthly' — requires day_of_month. "
                        "'interval' — every N seconds (requires interval_seconds)."
                    ),
                },
                "at": {
                    "type": "string",
                    "description": (
                        "Required for all types except 'interval'. "
                        "For 'once': ISO-8601 or natural language ('in 2 hours', 'tomorrow 09:00'). "
                        "For recurring types: HH:MM (24h)."
                    ),
                },
                "day_of_week": {
                    "type": "string",
                    "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                    "description": "Required for 'weekly'.",
                },
                "day_of_month": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 31,
                    "description": "Required for 'monthly'.",
                },
                "interval_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Required for 'interval'. Seconds between firings.",
                },
                "window_start": {
                    "type": "string",
                    "description": "For 'interval': earliest fire time, HH:MM (24h).",
                },
                "window_end": {
                    "type": "string",
                    "description": "For 'interval': latest fire time, HH:MM (24h).",
                },
                "system": {
                    "type": "boolean",
                    "description": "If true, fires as a system event without chat delivery.",
                },
            },
            "required": ["message", "schedule_type"],
        },
    ),
    _fn(
        "append_memory",
        (
            "Append a new fact to MEMORIES.txt. Use this for day-to-day memory "
            "storage. Each call adds one entry — no need to reproduce existing content. "
            "Nightly consolidation handles deduplication and pruning automatically."
        ),
        {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "The fact or note to append (one logical entry).",
                }
            },
            "required": ["entry"],
        },
    ),
    _fn(
        "update_memories",
        (
            "Overwrite MEMORIES.txt with new content. Use ONLY for restructuring "
            "or removing specific entries. For adding new facts, use append_memory instead. "
            "Pass the *full* desired content."
        ),
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full replacement text for MEMORIES.txt.",
                }
            },
            "required": ["content"],
        },
    ),
    _fn(
        "update_pulse",
        (
            "Overwrite PULSE.txt. Include only active items; omit completed goals. "
            "Pass the *full* desired content."
        ),
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full replacement text for PULSE.txt.",
                }
            },
            "required": ["content"],
        },
    ),
    _fn(
        "add_skill",
        (
            "Create or overwrite a skill documentation file in data/skills/. "
            "Skills are loaded into every system prompt automatically."
        ),
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Filename stem (no extension), e.g. 'calendar'.",
                },
                "documentation": {
                    "type": "string",
                    "description": "Markdown documentation for the skill.",
                },
            },
            "required": ["skill_name", "documentation"],
        },
    ),
    _fn(
        "execute_shell",
        (
            "Run a shell command. Returns stdout, stderr, and exit code. "
            "Use for system operations, package management, and tasks "
            "not covered by read_file/write_file."
        ),
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Defaults to 30.",
                },
            },
            "required": ["command"],
        },
    ),
    _fn(
        "read_file",
        "Read a file from the filesystem and return its contents.",
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
                    "description": "Text content to write.",
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
        "search_web",
        "Search the web via SearXNG. Returns titles, URLs, and snippets.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Defaults to 5.",
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
                    "description": "The URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Maximum characters of content to return. "
                        "Defaults to 20000. Use a lower value for summaries."
                    ),
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
    "update_memories":    "orchestration",
    "update_pulse":  "orchestration",
    "add_skill":          "orchestration",
    "list_reminders":     "orchestration",
}


def get_tool_schemas(categories: set[str] | None = None) -> list[dict]:
    """Return tool schemas filtered by category.

    If *categories* is None, return all schemas (used by the main agent).
    Otherwise return only schemas whose tool name maps to one of the
    requested categories.
    """
    if categories is None:
        return TOOL_SCHEMAS
    return [
        schema for schema in TOOL_SCHEMAS
        if TOOL_CATEGORIES.get(schema["function"]["name"]) in categories
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# These will be injected by the scheduler thread at startup.
_scheduler_set_reminder = None  # Callable[[dict], str]
_scheduler_list_reminders = None  # Callable[[], dict]

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


def _tool_update_memories(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.update_memories(inputs["content"])
        return json.dumps({"status": "ok"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_memories failed")
        return json.dumps({"error": str(exc)})


def _tool_update_pulse(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.update_pulse(inputs["content"])
        return json.dumps({"status": "ok"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_pulse failed")
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
    "update_memories":    _tool_update_memories,
    "update_pulse":  _tool_update_pulse,
    "add_skill":          _tool_add_skill,
    "execute_shell":      _tool_execute_shell,
    "read_file":          _tool_read_file,
    "write_file":         _tool_write_file,
    "list_reminders":     _tool_list_reminders,
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
    result = fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth)
    duration_ms = round((time.monotonic() - t0) * 1000, 1)
    _TOOL_CALL_LOG.append({
        "ts": time.time(),
        "tool": name,
        "inputs": inputs,
        "thread_id": thread_id or "unknown",
        "result_preview": result[:300] if result else "",
        "duration_ms": duration_ms,
        "error": None,
    })
    return result
