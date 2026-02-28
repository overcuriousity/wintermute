# Configuration Reference

All configuration lives in `config.yaml`. You can generate it in three ways:

1. **AI-driven onboarding (recommended):** `bash onboarding.sh` — walks you through every option conversationally using your own LLM (experimental, requires function-calling support)
2. **Classic setup script:** `bash setup.sh` — programmatic menu-driven prompts
3. **Manual:** `cp config.yaml.example config.yaml` and edit by hand

The `config.yaml.example` file contains detailed documentation for every field.

## Full Annotated Example

See `config.yaml.example` in the repository root for the complete reference with inline documentation. A minimal working config looks like:

```yaml
inference_backends:
  - name: "main"
    provider: "openai"
    base_url: "http://localhost:8080/v1"
    api_key: "llama-server"
    model: "qwen2.5:72b"
    context_size: 32768
    max_tokens: 4096

llm:
  base: ["main"]
```

## Section Details

### `inference_backends`

A list of named backend definitions. Each entry describes one LLM endpoint+model
combination that can be referenced by name in the `llm` role mapping.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `name` | yes | — | Unique name for this backend (referenced in `llm` role mapping) |
| `provider` | no | `"openai"` | `"openai"`, `"anthropic"`, `"gemini-cli"`, or `"kimi-code"` |
| `base_url` | openai only | — | OpenAI-compatible API base URL (not needed for `anthropic`, `gemini-cli`, or `kimi-code`) |
| `api_key` | openai/anthropic | — | API key (not needed for `gemini-cli` or `kimi-code`) |
| `model` | yes | — | Model name the endpoint accepts |
| `context_size` | no | `32768` | Total token window the model supports |
| `max_tokens` | no | `4096` | Maximum tokens per response |
| `reasoning` | no | `false` | Enable reasoning/thinking token support |

### `llm`

Maps roles to ordered lists of backend names. Each role specifies which
`inference_backends` entries to use, in priority order. If the first backend
fails (API error, timeout), the next one is tried automatically.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `base` | no | first backend | Primary conversation inference |
| `compaction` | no | first backend | Context history summarisation |
| `sub_sessions` | no | first backend | Background sub-session workers |
| `memory_harvest` | no | `sub_sessions` | Background memory extraction from conversations |
| `reflection` | no | `compaction` | Reflection cycle LLM analysis |
| `dreaming` | no | first backend | Nightly memory consolidation |
| `turing_protocol` | no | first backend | Turing Protocol validation pipeline |
**Failover:** When multiple backends are listed, they are tried in order on
API errors. The first backend's `context_size` is used for token budget
calculations.

#### Example: mixed providers

```yaml
inference_backends:
  - name: "gemini"
    provider: "gemini-cli"
    model: "gemini-2.5-pro"
    context_size: 1048576
    max_tokens: 8192

  - name: "ollama_small"
    provider: "openai"
    base_url: "http://localhost:8080/v1"
    api_key: "ollama"
    model: "qwen2.5:7b"
    context_size: 32768
    max_tokens: 2048

llm:
  base: ["gemini"]
  compaction: ["ollama_small"]
  sub_sessions: ["gemini", "ollama_small"]   # failover
  dreaming: ["ollama_small"]
  turing_protocol: ["ollama_small"]

turing_protocol:
  backends: ["ollama_small"]
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
```

#### Provider: `anthropic`

Uses Anthropic's native [Messages API](https://docs.anthropic.com/en/api/messages)
with prompt caching support. Requires a paid API key from
[console.anthropic.com](https://console.anthropic.com/) (pay-per-token billing).

> **Note:** Claude Pro/Max subscriptions do **not** include API access.
> OAuth tokens (`sk-ant-oat01-*`) from subscriptions are restricted to official
> Anthropic applications and cannot be used with Wintermute.

When `provider: "anthropic"` is set, only `api_key` is needed — no `base_url`.

```yaml
- name: "claude"
  provider: "anthropic"
  api_key: "sk-ant-api03-..."
  model: "claude-sonnet-4-20250514"
  context_size: 200000
  max_tokens: 8192
```

**Available models:**

| Model | Context | Notes |
|-------|---------|-------|
| `claude-sonnet-4-20250514` | 200k | Recommended — fast and capable |
| `claude-opus-4-20250514` | 200k | Most capable, higher cost |
| `claude-haiku-4-20250414` | 200k | Fastest, lowest cost — good for background tasks |

**Prompt caching:** System prompts larger than 3 KB and tool definitions are
automatically sent with `cache_control: ephemeral`, reducing costs on cache hits.

#### Provider: `gemini-cli`

> **Unstable / Alpha** — The `gemini-cli` provider piggybacks on Google's
> Cloud Code Assist OAuth flow. Credentials may expire unpredictably and
> the upstream API surface may change without notice. Suitable for
> experimentation; not recommended as your only backend.

Uses Google's Cloud Code Assist API via credentials extracted from a locally-installed
[gemini-cli](https://github.com/google/gemini-cli) (`npm i -g @google/gemini-cli`).
This provides free access to Gemini models (2.5 Flash/Pro, 3 Flash/Pro).

When `provider: "gemini-cli"` is set, `base_url` and `api_key` are not needed.
Authentication is handled via OAuth — run `uv run python -m wintermute.gemini_auth`
to set up credentials, or select the Gemini option during `onboarding.sh` / `setup.sh`.

Available models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-pro-preview`, `gemini-3-flash-preview`.

**Systemd note:** NVM paths are not available in systemd's minimal environment.
Wintermute auto-probes common NVM/Volta paths at startup. See
[installation.md — Systemd / headless service](installation.md#systemd--headless-service)
if using a non-standard Node installation.

#### Provider: `kimi-code`

Uses [Kimi-Code](https://kimi.com) ($19/month flat-rate subscription) via an OpenAI-compatible
endpoint at `https://api.kimi.com/coding/v1`. Authentication uses OAuth device-code flow —
no API keys needed.

When `provider: "kimi-code"` is set, `base_url` and `api_key` are not needed.
Credentials are stored in `data/kimi_credentials.json` and persist across restarts.

**Authentication:**
- During `onboarding.sh` / `setup.sh`: option 7 runs interactive device-code auth
- CLI: `uv run python -m wintermute.kimi_auth`
- In chat: `/kimi-auth` command (Matrix or web UI)
- Auto-trigger: if kimi-code is configured but not authenticated, Wintermute
  automatically starts the device flow on startup and broadcasts the verification
  URL to connected interfaces

```yaml
- name: "kimi"
  provider: "kimi-code"
  model: "kimi-for-coding"
  context_size: 131072
  max_tokens: 8192
```

**Available models:** The model list is dynamic (fetched from `https://api.kimi.com/coding/v1/models`).
Known models include:

| Model | Context | Reasoning | Notes |
|-------|---------|-----------|-------|
| `kimi-for-coding` | 131072 | no | Default coding model |
| `kimi-code` | 131072 | no | Current Kimi Code platform model |
| `kimi-k2.5` | 131072 | yes | Latest model, supports thinking/reasoning, image and video input |

Set `reasoning: true` for models that support thinking (e.g. `kimi-k2.5`) to enable
`max_completion_tokens` and reasoning token extraction in the web UI.

### `turing_protocol`

Phase-aware three-stage validation pipeline (detect → validate → correct)
that catches and corrects violations in assistant responses. Each hook fires
at most once per turn — a single, concise correction is issued with no
escalation or re-checking.

Hooks operate at three **phases** of the inference cycle:

| Phase | When it fires |
|-------|---------------|
| `post_inference` | After the LLM produces a response, before delivery |
| `pre_execution` | After the LLM requests a tool call, before `execute_tool()` runs |
| `post_execution` | After `execute_tool()` returns, before the result enters history |

Hooks are **scoped** to run in `main` (user-facing thread), `sub_session`
(background workers), or both.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `backends` | no | — | **Deprecated** — use `llm.turing_protocol` instead. Still supported for backwards compatibility |
| `validators` | no | all enabled | Per-hook enable/disable overrides (see below) |

**Disabling:** Set `llm.turing_protocol: []` to disable entirely, or set individual
validators to `false` to suppress specific checks.

**Default behavior:** If the `turing_protocol:` section is omitted entirely,
the protocol defaults to using the base model backends with all validators
enabled.

Currently available validators:

| Validator | Phase | Scope | Type | Description |
|-----------|-------|-------|------|-------------|
| `workflow_spawn` | post_inference | main | programmatic | Detects when the model claims to have spawned a session without calling `spawn_sub_session` |
| `phantom_tool_result` | post_inference | main | programmatic | Detects when the model presents fabricated tool output ("I checked and found…") without having called the tool |
| `empty_promise` | post_inference | main | programmatic | Detects when the model commits to an action ("I'll do X") as a final response without calling any tool. Excludes responses that end with a question (seeking confirmation) |
| `objective_completion` | post_inference | sub_session | LLM | Gates sub-session exit: uses a dedicated LLM call to evaluate whether the worker's response genuinely satisfies its objective before allowing it to finish |
| `task_complete` | pre_execution | sub_session | programmatic | Blocks `task(action='complete')` calls that lack a substantive `reason`. Always-on; not configurable via the `validators` map |
| `tool_schema_validation` | pre_execution | main + sub_session | programmatic | Validates tool arguments against the tool's JSON Schema before execution (required fields, types, enums, constraints). Always-on; not configurable via the `validators` map |

For a detailed explanation of each hook, phases, scopes, and how to write custom hooks, see [turing-protocol.md](turing-protocol.md).

**Validator overrides** support simple booleans or granular dicts:

```yaml
turing_protocol:
  backends: ["local_small"]
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
    objective_completion:
      enabled: true
      scope: "sub_session"          # "main", "sub_session", or both
```

### `nl_translation`

Natural-language tool call translation for weak/small LLMs. When enabled,
complex tools (`task`, `spawn_sub_session`, `add_skill`) are
presented to the main LLM as a single "describe in English" field. A
dedicated translator LLM expands the description into structured arguments.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `false` | Enable NL translation (opt-in) |
| `backends` | no | turing_protocol backends | Ordered list of backend names for the translator LLM |
| `tools` | no | `[task, spawn_sub_session, add_skill]` | Which tools use simplified NL schemas |

The translator can return JSON arrays to add multiple tasks or
spawn multiple sub-sessions from a single description. Ambiguous input
triggers a clarification request back to the user.

Complementary to the Turing Protocol's `tool_schema_validation` hook —
validation runs on the *translated* structured arguments, not the raw
description.

### `matrix`

See [matrix-setup.md](matrix-setup.md) for full setup instructions.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `homeserver` | yes | — | Matrix homeserver URL |
| `user_id` | yes | — | Bot's full Matrix user ID |
| `password` | no | `""` | Bot account password (recommended) |
| `access_token` | no | `""` | Manual access token |
| `device_id` | no | `""` | Device ID (auto-filled with password login) |
| `allowed_users` | yes | `[]` | User IDs allowed to interact |
| `allowed_rooms` | no | `[]` | Room ID whitelist (empty = all rooms) |

### `whisper`

Transcribes Matrix voice messages using an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Only relevant if Matrix is enabled. Without this, voice messages show a placeholder.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `false` | Enable voice transcription |
| `base_url` | yes (if enabled) | — | Whisper-compatible API base URL |
| `api_key` | yes (if enabled) | — | API key (`"none"` for unauthenticated local) |
| `model` | no | — | Model name (e.g. `"whisper-large-v3"`) |
| `language` | no | `""` | ISO-639-1 language hint (e.g. `"de"`). Empty = auto-detect |

### `web`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Enable/disable the web interface |
| `host` | no | `"127.0.0.1"` | Bind address |
| `port` | no | `8080` | Listen port |

### `dreaming`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `hour` | no | `1` | Hour (UTC, 0-23) for nightly consolidation |
| `minute` | no | `0` | Minute (0-59) |

### `update_checker`

Periodically checks the git remote for new commits and notifies via Matrix when updates are available. Requires git to be installed. The cached result is also shown by the `/status` command.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Set `false` to disable entirely |
| `check_on_startup` | no | `true` | Run a check immediately on startup (otherwise the first check happens after `interval_hours`) |
| `interval_hours` | no | `24` | Hours between periodic checks |
| `remote_url` | no | `""` | Git remote URL to check. Empty = use `origin` |

Notifications are sent to the Matrix rooms listed in `matrix.allowed_rooms` (or all rooms if that list is empty). If Matrix is not configured, the checker still runs (results visible via `/status`) but no notifications are sent.

### `memory`

Controls how memories are injected into the system prompt. By default, the entire `MEMORIES.txt` file is loaded each turn (flat-file mode). When a vector backend is configured, only the top-K most relevant memories are injected based on the current conversation context.

Three backends are available:

| Backend | Dependencies | Description |
|---------|-------------|-------------|
| `flat_file` | none | Default. Loads entire MEMORIES.txt (current behavior). Zero config. |
| `fts5` | none | SQLite FTS5 keyword search with BM25 ranking. Creates `data/memory_index.db`. |
| `local_vector` | embeddings endpoint | Semantic vector search with numpy cosine similarity. Vectors stored in `data/local_vectors.db`. No external vector service required. |
| `qdrant` | Qdrant instance + embeddings endpoint | Semantic vector search via Qdrant. Requires a running Qdrant instance and an OpenAI-compatible embeddings endpoint. |

When any non-`flat_file` backend is active, MEMORIES.txt is kept as a git-versioned backup via dual-write on every mutation. If the configured backend fails to initialize at startup, Wintermute warns and falls back to `flat_file` automatically.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `backend` | no | `"flat_file"` | `"flat_file"`, `"fts5"`, `"local_vector"`, or `"qdrant"` |
| `top_k` | no | `10` | Maximum memories to inject per turn |
| `score_threshold` | no | `0.3` | Minimum relevance score (vector backends only) |

#### `memory.embeddings` (local_vector and qdrant only)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `endpoint` | yes | — | OpenAI-compatible `/v1/embeddings` base URL |
| `api_key` | no | `""` | API key for the embeddings endpoint |
| `model` | no | `"text-embedding-3-small"` | Embedding model name |
| `dimensions` | no | `1536` | Vector dimensions (must match the model's output and the Qdrant collection) |
| `send_dimensions` | no | `false` | Send `dimensions` parameter in the embedding request (only OpenAI supports this) |
| `batch_size` | no | `32` | Maximum number of texts sent per embeddings API call. Reduce if the server returns HTTP 500 "input too large to process" (e.g. LiteLLM with a low physical batch token limit). |

#### `memory.qdrant` (qdrant only)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `url` | no | `"http://localhost:6333"` | Qdrant server URL (supports HTTP and HTTPS) |
| `api_key` | no | `""` | Qdrant API key (required for Qdrant Cloud or authenticated instances) |
| `collection` | no | `"wintermute_memories"` | Qdrant collection name |

#### `memory.dreaming` (vector backends only)

Controls the vector-native dreaming pipeline. Only used when `backend` is `local_vector` or `qdrant`. When `flat_file` is active, only flat-file LLM consolidation + task/skill consolidation run.

**Housekeeping settings:**

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `dedup_similarity_threshold` | no | `0.85` | Cosine similarity threshold for deduplication clustering |
| `stale_days` | no | `90` | Prune memories not accessed in this many days |
| `stale_min_access` | no | `3` | ...and accessed fewer than this many times |
| `working_set_size` | no | `50` | Top-N most-accessed memories exported to MEMORIES.txt |

**Association (REM-inspired) settings:**

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `association_enabled` | no | `true` | Enable the associative discovery phase |
| `association_min_entries` | no | `20` | Minimum memory entries required to run |
| `association_cooldown_hours` | no | `24` | Minimum hours between runs |
| `association_seed_count` | no | `5` | Number of seed memories for Qdrant `recommend()` |
| `association_candidate_count` | no | `20` | Number of candidate memories to evaluate |

**Schema (NREM slow-wave inspired) settings:**

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `schema_enabled` | no | `true` | Enable the schema abstraction phase |
| `schema_min_entries` | no | `30` | Minimum memory entries required to run |
| `schema_cooldown_hours` | no | `72` | Minimum hours between runs |
| `schema_cluster_sample` | no | `10` | Number of dedup clusters to sample for schema extraction |

**Prediction settings:**

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `prediction_enabled` | no | `true` | Enable the predictive pattern extraction phase |
| `prediction_min_entries` | no | `40` | Minimum memory entries required to run |
| `prediction_cooldown_hours` | no | `168` | Minimum hours between runs (default: 7 days) |
| `prediction_new_memory_threshold` | no | `10` | Minimum new memories since last dreaming cycle |

**Cold boot:** When a vector backend is configured and the index is empty but `MEMORIES.txt` has content, all entries are automatically imported at startup.

**Access tracking:** Every `search()` call updates `last_accessed` and `access_count` for returned entries. This metadata drives stale pruning (phase 3) and working set export (phase 4) during dreaming.

**Source tagging:** Each memory entry is tagged with its origin: `user_explicit` (via append_memory tool), `harvest` (via memory harvest workers), `dreaming_merge` (created by dreaming dedup/contradiction resolution), or `unknown` (legacy/untagged). User-explicit memories are protected from stale pruning.

**Interaction logging:** All embedding API calls and vector operations (search, add, replace_all) are logged to the interaction log, visible in the debug panel under the `embedding`, `local_vector_*`, `qdrant_*` action types.

### `memory_harvest`

Periodically mines recent conversation history for personal facts and preferences, extracting them into `MEMORIES.txt` (and the vector store, if active) via background sub-sessions. Complements the `append_memory` tool (which the AI uses in real-time during conversation).

Triggers when **either** condition is met:
- `message_threshold` new user messages have accumulated since last harvest
- `inactivity_timeout_minutes` of silence (requires at least 5 new messages)

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Set `false` to disable entirely |
| `message_threshold` | no | `20` | Harvest after N new user messages |
| `inactivity_timeout_minutes` | no | `15` | Or after N idle minutes (needs ≥ 5 msgs) |
| `max_message_chars` | no | `2000` | Truncate long messages before sending to the worker |

### `reflection`

Event-driven feedback loop that observes sub-session outcomes, detects patterns (repeated failures, stale tasks, skill correlations), and adapts the system. Uses a three-tier architecture: rule engine (zero LLM cost) → LLM analysis (one-shot) → sub-session mutations (rare). Each cycle also triggers the self-model update. See [autonomy.md — Reflection Cycle](autonomy.md#reflection-cycle) for details.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Set `false` to disable entirely (zero overhead) |
| `batch_threshold` | no | `10` | Trigger batch analysis every N sub-session completions |
| `consecutive_failure_limit` | no | `3` | Auto-pause tasks after N consecutive failures |
| `lookback_seconds` | no | `86400` | Time window (seconds) for pattern detection (default: 24h) |
| `min_result_length` | no | `50` | Results shorter than this are flagged as "no meaningful output" (stale task check) |

### `self_model`

Operational self-awareness profiler that runs inside the reflection cycle. Aggregates metrics from sub-session outcomes, interaction log, and event bus, then auto-tunes internal parameters and injects a cached prose summary into the system prompt. See [autonomy.md — Self-Model](autonomy.md#self-model) for details.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Set `false` to disable entirely |
| `sub_session_timeout_range` | no | `[120, 900]` | `[min, max]` seconds for sub-session timeout auto-tuning |
| `memory_harvest_threshold_range` | no | `[5, 50]` | `[min, max]` message count for harvest threshold auto-tuning |
| `summary_max_chars` | no | `300` | Maximum characters for the generated prose summary |

### `scheduler`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `timezone` | no | `"UTC"` | IANA timezone for all task scheduling (also affects `dreaming` schedule). Examples: `Europe/Berlin`, `America/New_York` |

### Per-Thread Configuration

Individual threads can override settings at runtime via the `/config` slash
command or the web debug API.  Overrides are stored in SQLite (`thread_config`
table) and persist across restarts.  When no override is set, hardcoded defaults
apply.  All changes are logged to the interaction log with action `config_change`
or `config_reset` for auditability.

| Key | Default | Description |
|-----|---------|-------------|
| `backend_name` | *(main pool)* | Pin inference to a named backend from `inference_backends`. When unset, the role-based `llm.base` pool is used. |
| `session_timeout_minutes` | `null` (disabled) | Idle timeout in minutes. After this period of inactivity the thread's session is automatically reset. |
| `sub_sessions_enabled` | `true` | Whether background sub-session workers can be spawned from this thread. |
| `system_prompt_mode` | `"full"` | `"full"` includes all prompt sections (memories, tasks, skills, etc.). `"minimal"` includes only Core Instructions + Current Time + Conversation Summary. |

See [Slash Commands](commands.md) for `/config` usage.

### `logging`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `level` | no | `"INFO"` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR`. `DEBUG` is very verbose (includes all LLM calls) |
| `directory` | no | `"logs"` | Log directory relative to working directory. Logs rotate daily, 7-day retention |

### `tool_profiles`

Named presets for sub-session workers. Each profile maps to a specific tool set and system prompt mode, providing a simpler alternative to manually specifying `tool_names` and `system_prompt_mode`.

```yaml
tool_profiles:
  researcher:
    tools: [search_web, fetch_url]
    prompt_mode: minimal
  file_worker:
    tools: [execute_shell, read_file, write_file]
    prompt_mode: minimal
  full_worker:
    tools: [execute_shell, read_file, write_file, search_web, fetch_url]
    prompt_mode: minimal
  orchestrator:
    tools: [spawn_sub_session, task, append_memory, add_skill]
    prompt_mode: full
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `tools` | yes | — | List of tool names available to the worker |
| `prompt_mode` | no | `"minimal"` | System prompt mode: `"minimal"`, `"full"`, `"base_only"`, or `"none"` |

Profiles are used via `spawn_sub_session(profile="researcher")`. Available profile names are automatically injected into the delegation section of the system prompt so the LLM knows what's available.

### `seed`

Controls the conversation seed — an automatic system event injected when a new conversation starts (first message in an empty thread or after `/new`). The seed prompts the LLM to introduce itself, mention relevant memories/tasks, and explain its capabilities.

Seed prompts are language-specific files in `data/prompts/SEED_{language}.txt`. Shipped languages: `en`, `de`, `fr`, `es`, `it`, `zh`, `ja`. Add your own by creating a `SEED_{code}.txt` file. Falls back to English if the configured language file is missing.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `language` | no | `"en"` | ISO-639-1 language code for the seed prompt (`"en"`, `"de"`, etc.) |

### `context.component_size_limits`

| Key | Default | Description |
|-----|---------|-------------|
| `memories` | `10000` | Char limit before MEMORIES.txt auto-summarisation |
| `tasks` | `5000` | Char limit before tasks auto-summarisation |
| `skills_total` | `2000` | Char limit for the skills TOC (summaries only; full skills are loaded on demand) |
