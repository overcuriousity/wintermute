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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from ganglion import prompt_assembler

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

SEARXNG_URL = os.environ.get("GANGLION_SEARXNG_URL", "http://127.0.0.1:8888")

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
            "Spawn an isolated background worker to handle a complex, multi-step task. "
            "Returns immediately with a session_id. The worker runs autonomously using "
            "all available tools and reports its result back to this thread when done. "
            "Use this when a task would take many tool calls or a long time, so you can "
            "remain responsive to the user."
        ),
        {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": (
                        "Full description of the task for the worker to complete. "
                        "Be specific — the worker has no access to the current conversation."
                    ),
                },
                "context_blobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of context snippets to pass to the worker "
                        "(e.g. relevant memory excerpts, file contents, user preferences). "
                        "Only include what is directly relevant to the task."
                    ),
                },
                "system_prompt_mode": {
                    "type": "string",
                    "enum": ["minimal", "full", "base_only", "none"],
                    "description": (
                        "How much of the system prompt to give the worker. "
                        "'minimal' (default) — lightweight execution agent, fastest and cheapest. "
                        "'base_only' — core instructions only. "
                        "'full' — includes MEMORIES, HEARTBEATS, and SKILLS; use when the "
                        "worker needs full user context. "
                        "'none' — bare tool-use loop, for purely mechanical tasks."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Maximum wall-clock seconds the worker may run before being stopped. "
                        "Defaults to 300. Use a higher value for tasks known to be slow "
                        "(e.g. large installations, long web scrapes)."
                    ),
                },
            },
            "required": ["objective"],
        },
    ),
    _fn(
        "set_reminder",
        (
            "Schedule a reminder. The scheduler thread will fire it at the "
            "specified time and optionally invoke a fresh AI inference with "
            "the given prompt."
        ),
        {
            "type": "object",
            "properties": {
                "time_spec": {
                    "type": "string",
                    "description": (
                        "When to fire. Natural language like 'in 2 hours', "
                        "'tomorrow 09:00', 'every day at 08:00', "
                        "or ISO-8601 datetime. For recurring='interval', "
                        "specify the period: 'every 5 minutes', '30 minutes', '2 hours'."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "Human-readable reminder text sent to chat.",
                },
                "ai_prompt": {
                    "type": "string",
                    "description": (
                        "Optional. If set, an isolated AI inference is run "
                        "with this prompt when the reminder fires."
                    ),
                },
                "recurring": {
                    "type": "string",
                    "enum": ["none", "daily", "weekly", "monthly", "interval"],
                    "description": (
                        "Recurrence type. Defaults to 'none'. "
                        "Use 'interval' for fixed-period repetition; then "
                        "time_spec must describe the interval, e.g. "
                        "'every 5 minutes', '30 minutes', '2 hours'."
                    ),
                },
                "system": {
                    "type": "boolean",
                    "description": (
                        "If true, creates a system reminder not bound to any "
                        "thread. It fires as a system event without chat delivery."
                    ),
                },
            },
            "required": ["time_spec", "message"],
        },
    ),
    _fn(
        "update_memories",
        (
            "Overwrite MEMORIES.txt with new content. Use this to persist "
            "important facts about the user. Pass the *full* desired content."
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
        "update_heartbeats",
        (
            "Overwrite HEARTBEATS.txt with new content. Use this to update "
            "active goals and working memory. Pass the *full* desired content."
        ),
        {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Full replacement text for HEARTBEATS.txt.",
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
            "Run a bash command as the current user. Returns stdout, stderr, "
            "and the exit code. Use for reading files, running scripts, "
            "checking system state, etc."
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
        "Return all reminders from the reminder registry.",
        {"type": "object", "properties": {}},
    ),
    _fn(
        "search_web",
        (
            "Search the web using the local SearXNG instance. Returns a list of "
            "results with title, URL, and snippet. Use this for factual queries, "
            "current events, documentation lookups, etc."
        ),
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
        (
            "Fetch the content of a web page and return it as plain text. "
            "HTML is stripped automatically. Use this to read documentation, "
            "articles, or any web resource."
        ),
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
    "update_memories":    "orchestration",
    "update_heartbeats":  "orchestration",
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

# This will be injected by the scheduler thread at startup.
_scheduler_set_reminder = None  # Callable[[dict], str]

# This will be injected by the SubSessionManager at startup.
_sub_session_spawn = None  # Callable[[dict, str], str]


def register_scheduler(fn) -> None:
    """Called once by the scheduler thread to provide the set_reminder hook."""
    global _scheduler_set_reminder
    _scheduler_set_reminder = fn


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
        session_id = _sub_session_spawn(**kwargs)
        return json.dumps({
            "status": "started",
            "session_id": session_id,
            "hint": "Acknowledge the background task to the user. Avoid heavy shell work yourself this turn.",
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


def _tool_update_memories(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.update_memories(inputs["content"])
        return json.dumps({"status": "ok"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_memories failed")
        return json.dumps({"error": str(exc)})


def _tool_update_heartbeats(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.update_heartbeats(inputs["content"])
        return json.dumps({"status": "ok"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("update_heartbeats failed")
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
    registry = DATA_DIR / "reminders.json"
    try:
        return registry.read_text(encoding="utf-8")
    except FileNotFoundError:
        return json.dumps({"active": [], "completed": [], "failed": []})
    except OSError as exc:
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
        req = Request(f"{SEARXNG_URL}/search?{params}", headers={"User-Agent": "ganglion/0.1"})
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
            f'curl -s --max-time 15 -A "ganglion/0.1" '
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
                "Mozilla/5.0 (compatible; ganglion/0.1; "
                "+https://github.com/ganglion)"
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
    "update_memories":    _tool_update_memories,
    "update_heartbeats":  _tool_update_heartbeats,
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
    return fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth)
