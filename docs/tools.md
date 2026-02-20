# Tools

Wintermute exposes 12 tools as OpenAI-compatible function-calling schemas, compatible with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).

## Tool Categories

Tools are grouped into three categories that control which tools are available in different contexts:

| Category | Available To | Tools |
|----------|-------------|-------|
| **execution** | All agents | `execute_shell`, `read_file`, `write_file` |
| **research** | All agents | `search_web`, `fetch_url` |
| **orchestration** | Main agent + `full`-mode sub-sessions | `spawn_sub_session`, `set_routine`, `append_memory`, `agenda`, `add_skill`, `list_routines`, `delete_routine` |

## Tool Filtering by Sub-session Mode

| Mode | Categories Available |
|------|---------------------|
| `minimal` (default) | execution, research |
| `base_only` | execution, research |
| `none` | execution, research |
| `full` | execution, research, orchestration |

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

#### `spawn_sub_session`

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

Returns: `status`, `session_id`

Maximum nesting depth: 2 (main -> sub -> sub-sub).

**Dependency safety:** Unknown session IDs in `depends_on` are automatically stripped with a warning log, preventing permanent deadlocks from hallucinated or mistyped IDs.

#### `set_routine`

Schedule a routine with optional AI inference on trigger.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | yes | Human-readable routine text |
| `ai_prompt` | string | no | Prompt for AI inference when routine fires |
| `schedule_type` | enum | yes | `"once"`, `"daily"`, `"weekly"`, `"monthly"`, `"interval"` |
| `at` | string | no | ISO-8601 datetime, natural language, or HH:MM |
| `day_of_week` | enum | no | Required for `weekly`: `mon`-`sun` |
| `day_of_month` | integer | no | Required for `monthly`: 1-31 |
| `interval_seconds` | integer | no | Required for `interval` |
| `window_start` | string | no | For `interval`: earliest fire time (HH:MM) |
| `window_end` | string | no | For `interval`: latest fire time (HH:MM) |
| `background` | boolean | no | Only valid with `ai_prompt`. When true, the AI task runs silently without delivering results to chat. Use for autonomous maintenance tasks. |

Returns: `status`, `job_id`

#### `append_memory`

Append a new fact to MEMORIES.txt. Preferred for day-to-day memory storage — no need to reproduce existing content. Nightly consolidation handles deduplication automatically.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry` | string | yes | The fact or note to append (one logical entry) |

Returns: `status`, `total_chars`

#### `agenda`

Manage active agenda items (working memory for ongoing tasks). Agenda items are stored in SQLite — no file rewrites needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | enum | yes | `"add"`, `"complete"`, `"list"`, `"update"` |
| `content` | string | no | Item text (for add/update) |
| `item_id` | integer | no | Item ID (for complete/update) |
| `priority` | integer | no | 1 (urgent) to 10 (low), default 5 |
| `status` | enum | no | Filter for list: `"active"` (default), `"completed"`, `"all"` |
| `reason` | string | no | Required for `complete`: evidence that the item is genuinely finished |

#### `add_skill`

Create or overwrite a skill documentation file in `data/skills/`. A summary appears in the system prompt's skills TOC; the full content is loaded on demand via `read_file`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | string | yes | Filename stem (no extension), e.g. `"calendar"` |
| `summary` | string | yes | One-line summary for the skills index (max 80 chars) |
| `documentation` | string | yes | Markdown documentation for the skill |

## NL Translation Mode

When `nl_translation.enabled: true` in config, `set_routine`,
`spawn_sub_session`, and `add_skill` are presented to the main LLM with
simplified single-field schemas. Instead of filling in all structured
parameters, the LLM writes a plain-English description:

```json
{"description": "remind me daily at 9am to check email"}
```

A dedicated translator LLM then expands this into the full structured
arguments (`schedule_type`, `at`, `message`, etc. for routines;
`skill_name`, `summary`, `documentation` for skills) before execution.
The tool result includes a `[Translated to: ...]` prefix showing the
expanded arguments.

The translator can return JSON arrays for multi-item requests (e.g.
"set three routines" or "research X then summarize it") — each item
is executed separately and results are combined.

If the description is ambiguous, the translator returns a clarification
request that the main LLM relays to the user.

This feature is complementary to the Turing Protocol's validation hooks,
which validate the *translated* structured arguments.

#### `list_routines`

Returns active, completed, and failed routines. No parameters. History is capped at the 200 most recent entries per category.

Returns: `active[]`, `completed[]`, `failed[]`

#### `delete_routine`

Cancel and remove a scheduled routine by its job ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `job_id` | string | yes | The job ID returned by `set_routine` or `list_routines` |

Returns: `status`, `job_id`
