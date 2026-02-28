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
import os
import re
import subprocess
import time
import urllib.parse
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from wintermute.infra import database
from wintermute.infra import prompt_assembler
from wintermute.core.tool_schemas import (  # noqa: F401 — re-exported
    TOOL_SCHEMAS,
    TOOL_CATEGORIES,
    NL_TOOL_SCHEMAS,
    NL_SCHEMA_MAP,
    get_tool_schemas,
)

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("WINTERMUTE_SEARXNG_URL", "http://127.0.0.1:8888")

# Backwards-compat alias (was private, now public in tool_schemas).
_NL_SCHEMA_MAP = NL_SCHEMA_MAP

# ---------------------------------------------------------------------------
# In-memory tool call log (bounded ring buffer for the debug UI)
# ---------------------------------------------------------------------------

_TOOL_CALL_LOG: deque[dict] = deque(maxlen=500)


# Maximum nesting depth for sub-session spawning.
# 0 = main agent, 1 = sub-session, 2 = sub-sub-session (max).
MAX_NESTING_DEPTH = 2


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

# These will be injected by the scheduler thread at startup.
_task_scheduler_ensure = None   # Callable[[str, dict, str|None, str|None, bool], None]
_task_scheduler_remove = None   # Callable[[str], None]
_task_scheduler_list = None     # Callable[[], list[dict]]

# This will be injected by the SubSessionManager at startup.
_sub_session_spawn = None  # Callable[[dict, str], str]

# Event bus — injected by main.py at startup.
_event_bus = None  # Optional[EventBus]


def register_task_scheduler(ensure_fn, remove_fn, list_fn) -> None:
    """Called once by the scheduler thread to provide task schedule management."""
    global _task_scheduler_ensure, _task_scheduler_remove, _task_scheduler_list
    _task_scheduler_ensure = ensure_fn
    _task_scheduler_remove = remove_fn
    _task_scheduler_list = list_fn


def set_searxng_url(url: str) -> None:
    """Override the default SearXNG URL (called from main.py config)."""
    global SEARXNG_URL
    SEARXNG_URL = url


def register_sub_session_manager(fn) -> None:
    """Called once by SubSessionManager to provide the spawn hook."""
    global _sub_session_spawn
    _sub_session_spawn = fn


def register_event_bus(bus) -> None:
    """Called once by main.py to provide the event bus."""
    global _event_bus
    _event_bus = bus


# Self-model profiler — injected by main.py at startup.
_self_model_profiler = None  # Optional[SelfModelProfiler]


def register_self_model(profiler) -> None:
    """Called once by main.py to provide the SelfModelProfiler instance."""
    global _self_model_profiler
    _self_model_profiler = profiler


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
        context_blobs = list(inputs.get("context_blobs") or [])

        # Auto-inject last user message from parent thread so the worker
        # has access to concrete values (tokens, URLs, IDs) the user provided
        # even if the LLM forgot to copy them into the objective/context_blobs.
        if thread_id:
            try:
                last_user_msg = database.get_last_user_message(thread_id)
                if last_user_msg:
                    context_blobs.append(
                        f"[Parent conversation — last user message]\n{last_user_msg}"
                    )
            except Exception:
                logger.debug("Failed to fetch parent user message", exc_info=True)

        # Query historical outcomes for similar objectives.
        try:
            similar = database.get_similar_outcomes(inputs["objective"], limit=5)
            if similar:
                lines = []
                total_dur = 0
                dur_count = 0
                success_count = 0
                for o in similar:
                    dur = o.get("duration_seconds")
                    tov = o.get("timeout_value")
                    tc = o.get("tool_call_count", 0)
                    st = o.get("status", "?")
                    obj_short = (o.get("objective") or "")[:60]
                    dur_str = f"{dur:.0f}s" if dur is not None else "?"
                    tov_str = f"{tov}s timeout" if tov is not None else "no timeout"
                    cont = o.get("continuation_count", 0)
                    extra = f", continued {cont}x" if cont else ""
                    lines.append(f"- \"{obj_short}\" ({tov_str}): {st} in {dur_str}, {tc} tool calls{extra}")
                    if dur is not None:
                        total_dur += dur
                        dur_count += 1
                    if st == "completed":
                        success_count += 1
                avg_dur = f"{total_dur / dur_count:.0f}s" if dur_count else "N/A"
                rate = f"{success_count * 100 / len(similar):.0f}%"
                blob = (
                    "[Historical Feedback] Similar past sub-sessions:\n"
                    + "\n".join(lines)
                    + f"\nAverage duration: {avg_dur} | Success rate: {rate}"
                )
                context_blobs.insert(0, blob)
        except Exception as exc:
            logger.debug("Historical feedback lookup failed: %s", exc, exc_info=True)

        kwargs = dict(
            objective=inputs["objective"],
            context_blobs=context_blobs,
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
        if "profile" in inputs:
            kwargs["profile"] = inputs["profile"]
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


def _task_add(inputs: dict, effective_scope: Optional[str]) -> str:
    content = inputs.get("content")
    if not content:
        return json.dumps({"error": "content is required for add action"})
    add_thread = inputs.get("thread_id") or effective_scope
    schedule_type = inputs.get("schedule_type")
    ai_prompt = inputs.get("ai_prompt")
    background = bool(inputs.get("background", False))

    schedule_config = None
    schedule_desc = None
    if schedule_type:
        sched_inputs = {k: inputs[k] for k in
            ("schedule_type", "at", "day_of_week", "day_of_month",
             "interval_seconds", "window_start", "window_end")
            if k in inputs}
        schedule_config = json.dumps(sched_inputs)
        schedule_desc = _describe_schedule(sched_inputs)

    task_id = database.add_task(
        content=content,
        priority=int(inputs.get("priority", 5)),
        thread_id=add_thread,
        schedule_type=schedule_type,
        schedule_desc=schedule_desc,
        schedule_config=schedule_config,
        ai_prompt=ai_prompt,
        background=background,
    )

    if schedule_type and _task_scheduler_ensure is not None:
        _task_scheduler_ensure(
            task_id, json.loads(schedule_config),
            ai_prompt, add_thread, background,
        )
        database.update_task(task_id, apscheduler_job_id=task_id)

    if _event_bus:
        _event_bus.emit("task.created", task_id=task_id,
                        content=content[:200],
                        schedule_type=schedule_type)
    result = {"status": "ok", "task_id": task_id}
    if schedule_desc:
        result["schedule"] = schedule_desc
    return json.dumps(result)


def _task_complete(inputs: dict, effective_scope: Optional[str]) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for complete action"})
    reason = (inputs.get("reason") or "").strip()
    if not reason:
        return json.dumps({"error": "reason is required for complete action — explain why this task is finished"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
        _task_scheduler_remove(task_id)
    ok = database.complete_task(task_id, reason=reason, thread_id=effective_scope)
    if ok and _event_bus:
        _event_bus.emit("task.completed", task_id=task_id, reason=reason[:200])
    return json.dumps({"status": "ok" if ok else "not_found", "reason": reason})


def _task_pause(inputs: dict, effective_scope: Optional[str]) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for pause action"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
        _task_scheduler_remove(task_id)
    ok = database.pause_task(task_id)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_resume(inputs: dict, effective_scope: Optional[str]) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for resume action"})
    ok = database.resume_task(task_id)
    if ok:
        # Re-register schedule with APScheduler.
        task = database.get_task(task_id)
        if task and task.get("schedule_config") and _task_scheduler_ensure:
            sched = json.loads(task["schedule_config"])
            _task_scheduler_ensure(
                task_id, sched,
                task.get("ai_prompt"), task.get("thread_id"),
                bool(task.get("background")),
            )
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_delete(inputs: dict, effective_scope: Optional[str]) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for delete action"})
    task = database.get_task(task_id)
    if task and task.get("apscheduler_job_id") and _task_scheduler_remove:
        _task_scheduler_remove(task_id)
    ok = database.delete_task(task_id)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_update(inputs: dict, effective_scope: Optional[str]) -> str:
    task_id = inputs.get("task_id")
    if not task_id:
        return json.dumps({"error": "task_id is required for update action"})
    kwargs = {}
    if "content" in inputs:
        kwargs["content"] = inputs["content"]
    if "priority" in inputs:
        kwargs["priority"] = int(inputs["priority"])
    ok = database.update_task(task_id, thread_id=effective_scope, **kwargs)
    return json.dumps({"status": "ok" if ok else "not_found"})


def _task_list(inputs: dict, effective_scope: Optional[str]) -> str:
    status = inputs.get("status", "active")
    items = database.list_tasks(status, thread_id=effective_scope)
    formatted = []
    for it in items:
        entry = {
            "id": it["id"],
            "content": it["content"],
            "priority": it["priority"],
            "status": it["status"],
        }
        if it.get("schedule_desc"):
            entry["schedule"] = it["schedule_desc"]
        if it.get("ai_prompt"):
            entry["ai_prompt"] = it["ai_prompt"][:100]
        if it.get("last_run_at"):
            entry["last_run_at"] = it["last_run_at"]
            entry["run_count"] = it.get("run_count", 0)
        if it.get("last_result_summary"):
            entry["last_result"] = it["last_result_summary"][:200]
        formatted.append(entry)
    return json.dumps({"tasks": formatted, "count": len(formatted)})


_TASK_ACTIONS: dict[str, Any] = {
    "add":      _task_add,
    "complete": _task_complete,
    "pause":    _task_pause,
    "resume":   _task_resume,
    "delete":   _task_delete,
    "update":   _task_update,
    "list":     _task_list,
}


def _tool_task(inputs: dict, thread_id: Optional[str] = None,
               parent_thread_id: Optional[str] = None, **_kw) -> str:
    """Unified task tool — handles add/update/complete/pause/resume/delete/list."""
    effective_scope = parent_thread_id or thread_id
    try:
        action = inputs.get("action", "list")
        handler = _TASK_ACTIONS.get(action)
        if handler is None:
            return json.dumps({"error": f"Unknown action: {action}"})
        return handler(inputs, effective_scope)
    except Exception as exc:  # noqa: BLE001
        logger.exception("task tool failed")
        return json.dumps({"error": str(exc)})


def _tool_append_memory(inputs: dict, **_kw) -> str:
    try:
        source = inputs.get("source", "user_explicit")
        total_len = prompt_assembler.append_memory(inputs["entry"], source=source)
        if _event_bus:
            _event_bus.emit("memory.appended", entry=inputs["entry"][:200])
        return json.dumps({"status": "ok", "total_chars": total_len})
    except Exception as exc:  # noqa: BLE001
        logger.exception("append_memory failed")
        return json.dumps({"error": str(exc)})




def _tool_add_skill(inputs: dict, **_kw) -> str:
    try:
        prompt_assembler.add_skill(
            inputs["skill_name"],
            inputs["documentation"],
            summary=inputs.get("summary"),
        )
        if _event_bus:
            _event_bus.emit("skill.added", skill_name=inputs["skill_name"])
        try:
            from wintermute.workers import skill_stats
            skill_stats.record_skill_written(inputs["skill_name"])
        except Exception:
            pass
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
        result = json.dumps({"content": path.read_text(encoding="utf-8")})
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except OSError as exc:
        return json.dumps({"error": str(exc)})
    # Track skill reads for skill_stats (only canonical data/skills/*.md).
    try:
        skills_dir = prompt_assembler.SKILLS_DIR.resolve()
        if path.suffix == ".md" and path.resolve().parent == skills_dir:
            from wintermute.workers import skill_stats
            skill_stats.record_read(path.stem)
    except Exception:
        pass
    return result


def _tool_write_file(inputs: dict, **_kw) -> str:
    path = Path(inputs["path"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(inputs["content"], encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path)})
    except OSError as exc:
        return json.dumps({"error": str(exc)})




def _tool_query_telemetry(inputs: dict, **_kw) -> str:
    """Query operational telemetry data."""
    query_type = inputs.get("query_type", "")
    since_hours = int(inputs.get("since_hours", 24))
    limit = int(inputs.get("limit", 10))
    status_filter = inputs.get("status_filter")
    since_ts = time.time() - since_hours * 3600

    try:
        if query_type == "outcome_stats":
            stats = database.get_outcome_stats()
            return json.dumps(stats)

        elif query_type == "recent_outcomes":
            kwargs = {"since": since_ts, "limit": limit}
            if status_filter:
                kwargs["status_filter"] = status_filter
            outcomes = database.get_outcomes_since(**kwargs)
            rows = []
            for o in outcomes:
                rows.append({
                    "session_id": o.get("session_id", ""),
                    "objective": (o.get("objective") or "")[:120],
                    "status": o.get("status", ""),
                    "duration_seconds": o.get("duration_seconds"),
                    "tool_call_count": o.get("tool_call_count", 0),
                })
            return json.dumps({"outcomes": rows, "count": len(rows)})

        elif query_type == "skill_stats":
            from wintermute.workers import skill_stats
            return json.dumps(skill_stats.get_all())

        elif query_type == "top_tools":
            tool_stats = database.get_tool_usage_stats(since_ts)
            return json.dumps({"tools": [{"name": t, "count": c} for t, c in tool_stats[:limit]]})

        elif query_type == "interaction_log":
            entries = database.get_interaction_log(limit=limit)
            rows = []
            for e in entries:
                rows.append({
                    "id": e.get("id"),
                    "action": e.get("action", ""),
                    "session": e.get("session", ""),
                    "input": (e.get("input") or "")[:200],
                    "output": (e.get("output") or "")[:200],
                    "status": e.get("status", ""),
                })
            return json.dumps({"entries": rows, "count": len(rows)})

        elif query_type == "self_model":
            if _self_model_profiler is None:
                return json.dumps({"error": "Self-model profiler not available"})
            summary = _self_model_profiler.get_summary()
            # Read raw YAML metrics.
            raw_metrics = {}
            try:
                yaml_path = _self_model_profiler.yaml_path
                if yaml_path.exists():
                    import yaml
                    raw_metrics = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except Exception:
                pass
            return json.dumps({"summary": summary, "metrics": raw_metrics})

        else:
            return json.dumps({"error": f"Unknown query_type: {query_type}"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("query_telemetry failed")
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


def _describe_schedule(inputs: dict) -> str:
    """Build a human-readable schedule string from structured inputs."""
    t = inputs.get("schedule_type", "once")
    if t == "once":
        return f"once at {inputs.get('at', '?')}"
    if t == "daily":
        return f"daily at {inputs.get('at', '?')}"
    if t == "weekly":
        return f"weekly on {inputs.get('day_of_week', '?')} at {inputs.get('at', '?')}"
    if t == "monthly":
        return f"monthly on day {inputs.get('day_of_month', '?')} at {inputs.get('at', '?')}"
    if t == "interval":
        secs = inputs.get("interval_seconds", "?")
        desc = f"every {secs}s"
        ws, we = inputs.get("window_start"), inputs.get("window_end")
        if ws and we:
            desc += f" from {ws} to {we}"
        return desc
    return str(inputs)


# ---------------------------------------------------------------------------
# Research tool implementations
# ---------------------------------------------------------------------------

def _tool_search_web(inputs: dict, **_kw) -> str:
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
    "task":               _tool_task,
    "append_memory":      _tool_append_memory,
    "add_skill":          _tool_add_skill,
    "execute_shell":      _tool_execute_shell,
    "read_file":          _tool_read_file,
    "write_file":         _tool_write_file,
    "search_web":         _tool_search_web,
    "fetch_url":          _tool_fetch_url,
    "query_telemetry":    _tool_query_telemetry,
}


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 nesting_depth: int = 0, parent_thread_id: Optional[str] = None) -> str:
    """Execute a tool by name and return its JSON-string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    logger.debug("Executing tool '%s' with inputs: %s", name, inputs)
    t0 = time.monotonic()
    try:
        result = fn(inputs, thread_id=thread_id, nesting_depth=nesting_depth,
                    parent_thread_id=parent_thread_id)
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
