# Tools

Wintermute exposes 12 tools as OpenAI-compatible function-calling schemas, compatible with any OpenAI-compatible endpoint (llama-server, vLLM, LM Studio, OpenAI, etc.).

## Tool Categories

Tools are grouped into three categories that control which tools are available in different contexts:

| Category | Available To | Tools |
|----------|-------------|-------|
| **execution** | All agents | `execute_shell`, `read_file`, `write_file` |
| **research** | All agents | `search_web`, `fetch_url` |
| **orchestration** | Main agent + `full`-mode sub-sessions | `spawn_sub_session`, `task`, `append_memory`, `add_skill` |

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
| `orchestrator` | `spawn_sub_session`, `task`, `append_memory`, `add_skill` | `full` |

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
| `profile` | string | no | Named tool profile (e.g. `"researcher"`, `"file_worker"`). Sets an optimised tool set and prompt mode. Overrides `system_prompt_mode`. |

Returns: `status`, `session_id`

Maximum nesting depth: 2 (main -> sub -> sub-sub).

**Dependency safety:** Unknown session IDs in `depends_on` are automatically stripped with a warning log, preventing permanent deadlocks from hallucinated or mistyped IDs.

#### `task`

Manage tasks — tracked goals and scheduled actions. Tasks are stored in SQLite. Tasks can optionally have schedules — scheduled tasks with `ai_prompt` run autonomous sub-sessions when the schedule fires.

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
| `ai_prompt` | string | no | Prompt for AI inference when schedule fires |
| `background` | boolean | no | Only valid with `ai_prompt`. When true, the AI task runs silently without delivering results to chat. Use for autonomous maintenance tasks. |

Returns vary by action. `add` returns `status`, `task_id`. `list` returns tasks grouped by status.

#### `append_memory`

Append a new fact to MEMORIES.txt. Preferred for day-to-day memory storage — no need to reproduce existing content. Nightly consolidation handles deduplication automatically.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry` | string | yes | The fact or note to append (one logical entry) |

Returns: `status`, `total_chars`

#### `add_skill`

Create or overwrite a skill documentation file in `data/skills/`. A summary appears in the system prompt's skills TOC; the full content is loaded on demand via `read_file`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | string | yes | Filename stem (no extension), e.g. `"calendar"` |
| `summary` | string | yes | One-line summary for the skills index (max 80 chars) |
| `documentation` | string | yes | Markdown documentation for the skill |

## NL Translation Mode

When `nl_translation.enabled: true` in config, `task`,
`spawn_sub_session`, and `add_skill` are presented to the main LLM with
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

This feature is complementary to the Turing Protocol's validation hooks,
which validate the *translated* structured arguments.
