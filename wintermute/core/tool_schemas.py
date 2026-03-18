"""
Tool schema definitions (OpenAI function-calling format).

Separated from tools.py so that core modules (e.g. convergence_protocol) can
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
        "worker_delegation",
        (
            "Manage background workers: spawn autonomous tasks, "
            "check status of running workers, or cancel them. "
            "Spawned task results arrive later via [SYSTEM EVENT]; "
            "status and cancel return immediately."
        ),
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "status", "cancel"],
                    "description": (
                        "'spawn' (default): start a new worker. "
                        "'status': list all running/pending workers. "
                        "'cancel': stop a worker or workflow by target_id."
                    ),
                },
                "target_id": {
                    "type": "string",
                    "description": (
                        "For cancel: session ID (sub_xxx), workflow ID, or 'all'. "
                        "Ignored for spawn/status."
                    ),
                },
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
                        "Worker capability profile. Determines which tools and "
                        "context the worker receives. Should be one of the "
                        "available profile names when spawning a worker."
                    ),
                },
            },
            "required": [],
        },
    ),
    _fn(
        "task",
        (
            "Manage tasks — goals, reminders, and scheduled actions. "
            "Actions: add, complete, update, delete, pause, resume, list."
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
                    "description": (
                        "Instruction for autonomous execution. Write a "
                        "complete, self-contained instruction the system "
                        "can execute independently. If omitted for scheduled "
                        "tasks, content is used as the prompt automatically."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "Run as background sub-session (auto-set when "
                        "ai_prompt + schedule are both present). Default: false."
                    ),
                },
            },
            "required": ["action"],
        },
    ),
    _fn(
        "append_memory",
        "Save a fact to long-term memory. One entry per call.",
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
        "skill",
        (
            "Skill store — saved step-by-step procedures. "
            "read/search returns documentation text; you then execute those steps yourself using your other tools. "
            "Actions: add (create/update), read (retrieve by name), search (semantic query)."
        ),
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "read", "search"],
                    "description": (
                        "'add': create or update a skill. "
                        "'read': retrieve a skill by exact name. "
                        "'search': find skills matching a query."
                    ),
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill identifier (alphanumeric, hyphens, underscores). Required for add/read.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary (max 80 chars). Optional for add; derived from first line of documentation if omitted.",
                },
                "documentation": {
                    "type": "string",
                    "description": "Markdown body for the skill. Be concise, max 500 chars. Required for add.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for action=search).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results for search (default: 5).",
                },
            },
            "required": ["action"],
        },
    ),
    _fn(
        "execute_shell",
        "Run a shell command. Returns stdout, stderr, exit_code.",
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
        "Search the web. Returns titles, URLs, and snippets.",
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
        "Query your own performance stats — success rates, outcomes, tool usage, and self-assessment.",
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
                        "skill_stats: per-skill access counts, versions, staleness (from skill store). "
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
        "send_file",
        "Send a file to the user. The file must exist on disk. "
        "Images are sent inline; other files as downloads.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to send.",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption or description for the file.",
                },
            },
            "required": ["path"],
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
    _fn(
        "restart_self",
        (
            "Restart the Wintermute process. Use after making code changes to yourself "
            "or when a full config reload is needed. The process shuts down gracefully "
            "(cancels all running operations, closes connections) then re-executes."
        ),
        {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the restart is needed (logged for diagnostics).",
                },
            },
            "required": [],
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
    "send_file":          "execution",
    "fetch_url":          "research",
    "worker_delegation":  "orchestration",
    "task":               "orchestration",
    "append_memory":      "orchestration",
    "skill":              "research",
    "query_telemetry":    "orchestration",
    "restart_self":       "orchestration",
}


NL_TOOL_SCHEMAS = [
    _fn(
        "task",
        (
            "Manage tasks — goals, reminders, and scheduled actions. "
            "Describe what you want in plain English."
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
        "worker_delegation",
        (
            "Manage background workers — spawn tasks (as DAG pipelines), "
            "check which workers are running, or cancel them. "
            "Describe what you need in plain English."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Plain-English description: what to spawn, or ask for status, "
                        "or which worker to cancel. "
                        "Examples: 'research AI safety then summarise', "
                        "'what workers are running?', 'cancel sub_a1b2c3d4', "
                        "'stop all background tasks'."
                    ),
                },
            },
            "required": ["description"],
        },
    ),
    _fn(
        "skill",
        (
            "Manage skills — save, retrieve, or search reusable procedures. "
            "Describe what you need in plain English."
        ),
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What skill operation to perform: add, read, or search. "
                        "Examples: 'save a skill called deploy-docker about "
                        "deploying containers: run docker compose up -d ...', "
                        "'look up the deploy-docker skill', "
                        "'search for skills about CI/CD pipelines'."
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


def _inject_profiles(schema: dict, profiles: dict[str, dict]) -> dict:
    """Return a copy of *schema* with profile names injected into the description."""
    import copy
    props = schema.get("function", {}).get("parameters", {}).get("properties", {})
    profile_prop = props.get("profile")
    if profile_prop is None:
        return schema
    names = ", ".join(sorted(profiles))
    new_desc = (
        f"Named tool profile. Available: {names}. "
        "Overrides system_prompt_mode and sets an optimised tool set. "
        "Only use one of the listed names."
    )
    schema = copy.deepcopy(schema)
    schema["function"]["parameters"]["properties"]["profile"]["description"] = new_desc
    schema["function"]["parameters"]["properties"]["profile"]["enum"] = sorted(profiles)
    return schema


def get_tool_schemas(categories: set[str] | None = None,
                     nl_tools: set[str] | None = None,
                     tool_profiles: dict[str, dict] | None = None,
                     exclude_names: set[str] | None = None) -> list[dict]:
    """Return tool schemas filtered by category, with optional NL substitution.

    If *categories* is None, return all schemas (used by the main agent).
    Otherwise return only schemas whose tool name maps to one of the
    requested categories.

    When *nl_tools* is provided, any tool whose name is in the set gets its
    schema replaced with the simplified single-field NL variant.

    When *tool_profiles* is provided, the ``worker_delegation`` schema's
    ``profile`` field gets an ``enum`` constraint and description listing
    the available profile names.

    When *exclude_names* is provided, schemas whose tool name is in the set
    are dropped from the result (used by lite mode to hide worker_delegation).
    """
    if categories is None:
        schemas = list(TOOL_SCHEMAS)
    else:
        schemas = [
            schema for schema in TOOL_SCHEMAS
            if TOOL_CATEGORIES.get(schema["function"]["name"]) in categories
        ]
    if exclude_names:
        schemas = [s for s in schemas if s["function"]["name"] not in exclude_names]
    # Inject profile names into the worker_delegation schema.
    if tool_profiles:
        schemas = [
            _inject_profiles(s, tool_profiles)
            if s["function"]["name"] == "worker_delegation"
            else s
            for s in schemas
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
