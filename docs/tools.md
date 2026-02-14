# Tools

Wintermute exposes 11 tools as OpenAI-compatible function-calling schemas, compatible with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI, etc.).

## Tool Categories

Tools are grouped into three categories that control which tools are available in different contexts:

| Category | Available To | Tools |
|----------|-------------|-------|
| **execution** | All agents | `execute_shell`, `read_file`, `write_file` |
| **research** | All agents | `search_web`, `fetch_url` |
| **orchestration** | Main agent + `full`-mode sub-sessions | `spawn_sub_session`, `set_reminder`, `update_memories`, `update_pulse`, `add_skill`, `list_reminders` |

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
| `depends_on` | string[] | no | Session IDs that must complete first |

Returns: `status`, `session_id`

Maximum nesting depth: 2 (main -> sub -> sub-sub).

#### `set_reminder`

Schedule a reminder with optional AI inference on trigger.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `message` | string | yes | Human-readable reminder text |
| `ai_prompt` | string | no | Prompt for AI inference when reminder fires |
| `schedule_type` | enum | yes | `"once"`, `"daily"`, `"weekly"`, `"monthly"`, `"interval"` |
| `at` | string | no | ISO-8601 datetime, natural language, or HH:MM |
| `day_of_week` | enum | no | Required for `weekly`: `mon`-`sun` |
| `day_of_month` | integer | no | Required for `monthly`: 1-31 |
| `interval_seconds` | integer | no | Required for `interval` |
| `window_start` | string | no | For `interval`: earliest fire time (HH:MM) |
| `window_end` | string | no | For `interval`: latest fire time (HH:MM) |
| `system` | boolean | no | If true, fires as a system event (no chat) |

Returns: `status`, `job_id`

#### `update_memories`

Overwrite MEMORIES.txt with new content. Pass the full desired content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | yes | Full replacement text for MEMORIES.txt |

#### `update_pulse`

Overwrite PULSE.txt with new content. Pass the full desired content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | yes | Full replacement text for PULSE.txt |

#### `add_skill`

Create or overwrite a skill documentation file in `data/skills/`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `skill_name` | string | yes | Filename stem (no extension), e.g. `"calendar"` |
| `documentation` | string | yes | Markdown documentation for the skill |

#### `list_reminders`

Return all reminders from the reminder registry. No parameters.

Returns: `active[]`, `completed[]`, `failed[]`
