"""
Tool schema definitions (OpenAI function-calling format).

Separated from tools.py so that core modules (e.g. turing_protocol) can
access schema data without importing the tool implementations and their
heavyweight dependencies.
"""


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
                    "description": (
                        "Task description. Worker has NO conversation access — "
                        "include ALL concrete values (URLs, tokens, credentials, IDs, "
                        "parameters) verbatim. Never say 'the provided token'; "
                        "paste the actual token into the objective or context_blobs."
                    ),
                },
                "context_blobs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Context snippets passed verbatim to the worker. "
                        "Use this for large data: API tokens, request bodies, "
                        "file contents, credentials. Unneeded when using depends_on."
                    ),
                },
                "system_prompt_mode": {
                    "type": "string",
                    "enum": ["minimal", "full", "base_only", "none"],
                    "description": (
                        "'minimal' (default): lightweight agent, no memories/skills. "
                        "'full': complete context (memories + tasks + skills). "
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
                "profile": {
                    "type": "string",
                    "description": (
                        "Named tool profile (e.g. 'researcher', 'file_worker'). "
                        "Overrides system_prompt_mode and sets an optimised tool set. "
                        "See available profiles in config."
                    ),
                },
            },
            "required": ["objective"],
        },
    ),
    _fn(
        "task",
        (
            "Manage tracked tasks — goals, reminders, and scheduled actions. "
            "Use action 'add' to create, 'complete' to finish, 'pause'/'resume' to control schedules, "
            "'update' to modify, 'delete' to remove, 'list' to show."
        ),
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "complete", "pause", "resume", "delete", "list"],
                    "description": "Operation to perform.",
                },
                "content": {
                    "type": "string",
                    "description": "Task description (for add/update).",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (for update/complete/pause/resume/delete).",
                },
                "reason": {
                    "type": "string",
                    "description": "Required for complete: evidence why this task is truly finished.",
                },
                "priority": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "1 (urgent) to 10 (low), default 5.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "paused", "completed", "all"],
                    "description": "For list: filter (default: active).",
                },
                "schedule_type": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly", "monthly", "interval"],
                    "description": "Omit for unscheduled tasks.",
                },
                "at": {
                    "type": "string",
                    "description": "Time spec. For once: ISO-8601. For recurring: HH:MM.",
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
                    "description": "Required for interval.",
                },
                "window_start": {
                    "type": "string",
                    "description": "For interval: earliest fire time, HH:MM.",
                },
                "window_end": {
                    "type": "string",
                    "description": "For interval: latest fire time, HH:MM.",
                },
                "ai_prompt": {
                    "type": "string",
                    "description": "AI action to run when schedule fires. Write as a complete task instruction.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Silent execution — no chat delivery. Only valid with ai_prompt.",
                },
            },
            "required": ["action"],
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
                },
                "source": {
                    "type": "string",
                    "description": "Origin tag for this memory (e.g. 'user_explicit', 'harvest'). Default: 'user_explicit'.",
                },
            },
            "required": ["entry"],
        },
    ),
    _fn(
        "add_skill",
        "Create or overwrite a skill in data/skills/. A summary appears in the system prompt; full content is loaded on demand via read_file.",
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Filename stem without extension (e.g. 'calendar').",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary for the skills index (max 80 chars).",
                },
                "documentation": {
                    "type": "string",
                    "description": "Markdown documentation for the skill. Be concise, max 500 chars.",
                },
            },
            "required": ["skill_name", "summary", "documentation"],
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
        "query_telemetry",
        "Query your own operational telemetry — success rates, recent outcomes, skill stats, tool usage, interaction logs, and self-model summary.",
        {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["outcome_stats", "recent_outcomes", "skill_stats",
                             "top_tools", "interaction_log", "self_model"],
                    "description": (
                        "outcome_stats: aggregate success/failure counts. "
                        "recent_outcomes: latest sub-session results. "
                        "skill_stats: read/write/failure counts per skill. "
                        "top_tools: most-used tools. "
                        "interaction_log: recent log entries. "
                        "self_model: cached self-assessment summary + raw metrics."
                    ),
                },
                "since_hours": {
                    "type": "integer",
                    "description": "Lookback window in hours (default: 24).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 10).",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["completed", "failed", "timeout"],
                    "description": "Filter outcomes by status (for recent_outcomes).",
                },
            },
            "required": ["query_type"],
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
    "task":               "orchestration",
    "append_memory":      "orchestration",
    "add_skill":          "orchestration",
    "query_telemetry":    "orchestration",
}


NL_TOOL_SCHEMAS = [
    _fn(
        "task",
        (
            "Manage tasks — tracked goals, reminders, and scheduled actions. "
            "Describe what you want in plain English — the system will translate "
            "it into structured arguments."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description of the task operation. "
                        "Examples: 'add a high-priority task to fix the login bug', "
                        "'remind me daily at 9am to check email', "
                        "'every morning at 8 search for news and summarize', "
                        "'complete task_abc because the server is fixed', "
                        "'pause task_def', 'list all tasks'."
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
    _fn(
        "add_skill",
        (
            "Save a learned procedure as a reusable skill. Describe what skill "
            "to save and the procedure details in plain English — the system "
            "will format it into a properly structured skill file."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What skill to save: a name and the full procedure. "
                        "Example: 'save a skill called deploy-docker about "
                        "deploying containers: run docker compose up -d ...'"
                    ),
                },
            },
            "required": ["description"],
        },
    ),
]

NL_SCHEMA_MAP: dict[str, dict] = {
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
        if name in nl_tools and name in NL_SCHEMA_MAP:
            result.append(NL_SCHEMA_MAP[name])
        else:
            result.append(schema)
    return result
