"""
Tool definitions and execution for the AI assistant.

Tools are expressed as OpenAI-compatible function-calling schemas so they work
with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).
The dispatcher ``execute_tool`` is the single entry point used by the LLM thread.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from ganglion import prompt_assembler

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

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
                    "enum": ["full", "base_only", "none"],
                    "description": (
                        "How much of the system prompt to give the worker. "
                        "'base_only' (default) — core instructions only, fastest and cheapest. "
                        "'full' — includes MEMORIES, HEARTBEATS, and SKILLS; use when the "
                        "worker needs full user context. "
                        "'none' — bare tool-use loop, for purely mechanical tasks."
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
                            in_sub_session: bool = False, **_kw) -> str:
    if in_sub_session:
        return json.dumps({"error": "spawn_sub_session cannot be called from within a sub-session."})
    if _sub_session_spawn is None:
        return json.dumps({"error": "Sub-session manager not ready yet."})
    try:
        session_id = _sub_session_spawn(
            objective=inputs["objective"],
            context_blobs=inputs.get("context_blobs") or [],
            parent_thread_id=thread_id,
            system_prompt_mode=inputs.get("system_prompt_mode", "base_only"),
        )
        return json.dumps({"status": "started", "session_id": session_id})
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
}


def execute_tool(name: str, inputs: dict, thread_id: Optional[str] = None,
                 in_sub_session: bool = False) -> str:
    """Execute a tool by name and return its JSON-string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    logger.debug("Executing tool '%s' with inputs: %s", name, inputs)
    return fn(inputs, thread_id=thread_id, in_sub_session=in_sub_session)
