# Tools

Wintermute exposes 12 tools as OpenAI-compatible function-calling schemas, compatible with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).

## Tool Categories

Tools are grouped into three categories that control which tools are available in different contexts:

| Category | Available To | Tools |
|----------|-------------|-------|
| **execution** | All agents | `execute_shell`, `read_file`, `write_file`, `send_message` |
| **research** | All agents | `search_web`, `fetch_url`, `skill` |
| **orchestration** | Main agent + `full`-mode sub-sessions | `worker_delegation`, `task`, `append_memory`, `query_telemetry`, `restart_self` |
| *(uncategorized)* | Main agent only | `send_file` |

## Tool Filtering by Sub-session Mode

| Mode | Categories Available |
|------|---------------------|
| `minimal` (default) | execution, research |
| `base_only` | execution, research |
| `none` | execution, research |
| `full` | execution, research, orchestration |

## Tool Profiles

Named tool profiles provide config-driven presets for common sub-session worker patterns. Instead of specifying individual `tool_names` or relying on coarse category-based modes, the LLM can use a named profile when spawning workers.

### Default Profiles

| Profile | Tools | Prompt Mode |
|---------|-------|-------------|
| `researcher` | `search_web`, `fetch_url` | `minimal` |
| `file_worker` | `execute_shell`, `read_file`, `write_file` | `minimal` |
| `full_worker` | `execute_shell`, `read_file`, `write_file`, `search_web`, `fetch_url` | `minimal` |
| `orchestrator` | `worker_delegation`, `task`, `append_memory`, `skill` | `full` |

### Custom Profiles

Define custom profiles in `config.yaml` under `tool_profiles`:

```yaml
tool_profiles:
  my_profile:
    tools: [search_web, fetch_url, execute_shell]
    prompt_mode: base_only
```

### Relationship to `system_prompt_mode`

A profile sets both the tool set and the system prompt mode. If `system_prompt_mode` is explicitly passed alongside `profile`, the explicit mode takes precedence. If `tool_names` is explicitly passed, it overrides the profile's tool list.

## Tool Reference

### Execution Tools

#### `execute_shell`

Run a bash command as the current user.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `command` | string | yes | Shell command to execute |
| `timeout` | integer | no | Timeout in seconds (default: 30) |

Returns: `stdout`, `stderr`, `exit_code`

#### `read_file`

Read a file from the filesystem.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative file path |

Returns: `content`

#### `write_file`

Write content to a file, creating parent directories as needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative file path |
| `content` | string | yes | Text content to write |

#### `send_message`

Send a text message directly to the user's chat. Use for notifications, alerts, and reminders from sub-sessions — not for normal conversation replies (those go through the standard inference flow). Frontend-agnostic: emits a `send_message` event on the EventBus; each frontend (Matrix, Signal) subscribes and handles delivery independently.

When called from a sub-session, the message is routed to the originating chat thread (via `parent_thread_id`), not the sub-session itself.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | yes | The message text to send to the user |

Returns: `status`, `thread_id`

#### `send_file`

Send a file to the user. Images are sent inline; other files as downloads. Frontend-agnostic: emits a `send_file` event on the EventBus; each frontend (Matrix, Signal) subscribes and handles delivery independently. Main agent only — not available in sub-sessions.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | Absolute or relative path to the file |
| `caption` | string | no | Optional caption or description |

Returns: `status`, `path`, `filename`, `mime_type`, `file_size`

### Research Tools

#### `search_web`

Search the web using the local SearXNG instance. Falls back to DuckDuckGo via `curl` when SearXNG is unavailable.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | The search query |
| `max_results` | integer | no | Maximum results to return (default: 5) |

Returns: `query`, `source`, `results[]` (title, url, snippet), `count`

#### `fetch_url`

Fetch the content of a web page and return it as plain text. HTML is stripped automatically.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | yes | The URL to fetch |
| `max_chars` | integer | no | Maximum characters to return (default: 20000) |

Returns: `url`, `content_type`, `length`, `content`

### Orchestration Tools

#### `worker_delegation`

Spawn an isolated background worker for a complex, multi-step task.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `objective` | string | yes | Full task description for the worker |
| `context_blobs` | string[] | no | Context snippets to pass to the worker |
| `system_prompt_mode` | enum | no | `"minimal"` (default), `"full"`, `"base_only"`, `"none"` |
| `timeout` | integer | no | Max seconds before timeout (default: 300) |
| `depends_on` | string[] | no | Session IDs that must complete first. Prefer `depends_on_previous` over manually listing IDs. |
| `depends_on_previous` | boolean | no | If true, automatically depend on all sessions spawned so far by the calling worker. Eliminates the need to track session IDs manually — prevents hallucinated-ID deadlocks. |
| `not_before` | string | no | Earliest start time (ISO-8601). Task waits even if deps are done. |
| `profile` | string | no | Named tool profile (e.g. `"researcher"`, `"file_worker"`). Sets an optimised tool set and prompt mode. Overrides `system_prompt_mode`. |

Returns: `status`, `session_id`

Maximum nesting depth: 2 (main -> sub -> sub-sub).

**Dependency safety:** Unknown session IDs in `depends_on` are automatically stripped with a warning log, preventing permanent deadlocks from hallucinated or mistyped IDs.

#### `task`

Manage tasks — tracked goals and scheduled actions. Tasks are stored in SQLite. Scheduled behavior is explicitly controlled via `execution_mode` for reminders vs autonomous runs.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | enum | yes | `"add"`, `"update"`, `"complete"`, `"pause"`, `"resume"`, `"delete"`, `"list"` |
| `content` | string | no | Task text (for add/update) |
| `task_id` | integer | no | Task ID (for update/complete/pause/resume/delete) |
| `priority` | integer | no | 1 (urgent) to 10 (low), default 5 |
| `status` | enum | no | Filter for list: `"active"` (default), `"completed"`, `"paused"`, `"all"` |
| `reason` | string | no | Required for `complete`: evidence that the task is genuinely finished |
| `schedule_type` | enum | no | `"once"`, `"daily"`, `"weekly"`, `"monthly"`, `"interval"` |
| `at` | string | no | ISO-8601 datetime, natural language, or HH:MM |
| `day_of_week` | enum | no | Required for `weekly`: `mon`-`sun` |
| `day_of_month` | integer | no | Required for `monthly`: 1-31 |
| `interval_seconds` | integer | no | Required for `interval` |
| `window_start` | string | no | For `interval`: earliest fire time (HH:MM) |
| `window_end` | string | no | For `interval`: latest fire time (HH:MM) |
| `ai_prompt` | string | no | Prompt for autonomous execution; required when `execution_mode` is `autonomous_notify` or `autonomous_silent` |
| `execution_mode` | enum | no | Scheduled execution behavior: `reminder` (chat reminder), `autonomous_notify` (run AI and post results), `autonomous_silent` (run AI silently) |
| `background` | boolean | no | Deprecated legacy flag; ignored when `execution_mode` is set. For legacy payloads with `ai_prompt`, `true` maps to `autonomous_notify` and `false` to `autonomous_silent`. |

Returns vary by action. `add` returns `status`, `task_id`. `list` returns tasks grouped by status.

#### `append_memory`

Append a new fact to the memory store. Preferred for day-to-day memory storage — no need to reproduce existing content. Deduplication is handled automatically at add-time (similar entries are merged via LLM). Nightly dreaming provides additional consolidation.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry` | string | yes | The fact or note to append (one logical entry) |
| `source` | string | no | Origin tag for this memory. Default: `"user_explicit"`. Other values: `"harvest"` (memory harvest workers). Used by dreaming to protect user-explicit memories from stale pruning. |

Returns: `status`, `total_entries`

#### `skill`

Unified skill management tool with three actions: `add`, `read`, and `search`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | enum | yes | `"add"`, `"read"`, or `"search"` |
| `skill_name` | string | add/read | Skill identifier (alphanumeric, hyphens, underscores) |
| `summary` | string | no | One-line summary (derived from first line of documentation if omitted) |
| `documentation` | string | add | Markdown documentation for the skill |
| `query` | string | search | Search query for relevance-ranked results |
| `top_k` | integer | no | Max results for search (default: 5, max: 50) |

#### `query_telemetry`

Query the system's own operational telemetry — success rates, recent outcomes, skill stats, tool usage, interaction logs, and self-model summary.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_type` | enum | yes | `"outcome_stats"`, `"recent_outcomes"`, `"skill_stats"`, `"top_tools"`, `"interaction_log"`, `"self_model"` |
| `since_hours` | integer | no | Lookback window in hours (default: 24) |
| `limit` | integer | no | Max results to return (default: 10) |
| `status_filter` | enum | no | Filter for `recent_outcomes`: `"completed"`, `"failed"`, `"timeout"` |

**Query types:**

| Type | Returns |
|------|---------|
| `outcome_stats` | Aggregate sub-session success/failure/timeout counts |
| `recent_outcomes` | Latest sub-session results with objective, status, duration |
| `skill_stats` | Per-skill access counts, versions, and staleness/recency (from skill store) |
| `top_tools` | Most-used tools in the lookback window |
| `interaction_log` | Recent interaction log entries (inference, tool calls, reflections) |
| `self_model` | Cached self-assessment summary + raw metrics from YAML |

## NL Translation Mode

When `nl_translation.enabled: true` in config, `task`,
`worker_delegation`, and `skill` are presented to the main LLM with
simplified single-field schemas. Instead of filling in all structured
parameters, the LLM writes a plain-English description:

```json
{"description": "remind me daily at 9am to check email"}
```

A dedicated translator LLM then expands this into the full structured
arguments (`action`, `content`, `schedule_type`, `at`, etc. for tasks;
`skill_name`, `summary`, `documentation` for skills) before execution.
The tool result includes a `[Translated to: ...]` prefix showing the
expanded arguments.

The translator can return JSON arrays for multi-item requests (e.g.
"add three tasks" or "research X then summarize it") — each item
is executed separately and results are combined.

If the description is ambiguous, the translator returns a clarification
request that the main LLM relays to the user.

This feature is complementary to the Convergence Protocol's validation hooks,
which validate the *translated* structured arguments.
