# Configuration Reference

All configuration lives in `config.yaml`. Copy `config.yaml.example` to get started:

```bash
cp config.yaml.example config.yaml
```

## Full Annotated Example

```yaml
# ── Matrix (optional) ─────────────────────────────────────────────
# Remove or leave empty to disable Matrix. The web UI works standalone.
matrix:
  homeserver: https://matrix.org          # Your Matrix homeserver URL
  user_id: "@bot:matrix.org"              # Full Matrix user ID of the bot account

  # Option A (recommended): supply password — Wintermute logs in automatically,
  # fills in access_token/device_id below, and refreshes them on expiry.
  password: ""
  # Option B: supply token + device_id from a manual login (leave password empty).
  access_token: ""
  device_id: ""                           # Auto-filled when using password login.

  allowed_users:                          # Matrix user IDs allowed to interact
    - "@admin:matrix.org"
  allowed_rooms: []                       # Room ID whitelist (empty = any room)

# ── Web Interface ─────────────────────────────────────────────────
# Enabled by default. Open http://127.0.0.1:8080 in your browser.
web:
  enabled: true
  host: "127.0.0.1"                       # Use 0.0.0.0 to listen on all interfaces
  port: 8080

# ── Inference Backends ─────────────────────────────────────────────
# Named backend definitions. Each describes one endpoint+model combination.
# Referenced by name in the llm: role mapping.
inference_backends:
  - name: "local_large"
    provider: "openai"                    # "openai", "gemini-cli", or "kimi-code"
    base_url: "http://localhost:8080/v1"
    api_key: "llama-server"
    model: "qwen2.5:72b"
    context_size: 32768
    max_tokens: 4096
    reasoning: false

  - name: "local_small"
    provider: "openai"
    base_url: "http://localhost:8080/v1"
    api_key: "llama-server"
    model: "qwen2.5:7b"
    context_size: 32768
    max_tokens: 2048
    reasoning: false

  # UNSTABLE/ALPHA — gemini-cli piggybacks on Google's Cloud Code Assist
  # OAuth flow.  Credentials may expire unpredictably.  Not recommended
  # as your only backend.
  # - name: "gemini_pro"
  #   provider: "gemini-cli"
  #   model: "gemini-2.5-pro"
  #   context_size: 1048576
  #   max_tokens: 8192

  # Kimi-Code — $19/mo flat-rate subscription with device-code OAuth.
  # Auth: uv run python -m wintermute.kimi_auth  (or /kimi-auth in chat)
  # Credentials: data/kimi_credentials.json
  # - name: "kimi"
  #   provider: "kimi-code"
  #   model: "kimi-for-coding"
  #   context_size: 131072
  #   max_tokens: 8192

  # - name: "turing_backend"
  #   provider: "openai"
  #   base_url: "http://localhost:8080/v1"
  #   api_key: "llama-server"
  #   model: "qwen2.5:7b"
  #   context_size: 32768
  #   max_tokens: 150

# ── LLM Role Mapping ──────────────────────────────────────────────
# Maps roles to ordered lists of backend names.
# Multiple backends = automatic failover on API errors.
# Empty list [] = disabled.
# Omitted roles default to the first defined backend.
llm:
  base: ["local_large"]
  compaction: ["local_small", "local_large"]
  sub_sessions: ["local_large"]
  dreaming: ["local_small"]

# -- Turing Protocol (Post-Inference Validation) ---------------------
turing_protocol:
  backends: ["local_small"]               # small/fast model recommended
  validators:
    workflow_spawn: true                   # detect hallucinated workflow spawn claims
    phantom_tool_result: true              # detect fabricated tool output claims
    empty_promise: true                    # detect unfulfilled action commitments

# ── Context Compaction ────────────────────────────────────────────
# Compaction fires when history tokens exceed:
#   context_size - max_tokens - system_prompt_tokens
# No separate threshold setting is needed; adjust in the backend definition.

# ── Pulse Reviews ─────────────────────────────────────────────────
pulse:
  enabled: true                           # Set to false to disable pulse reviews
  review_interval_minutes: 60             # How often to auto-review pulse items

# ── Component Size Limits ─────────────────────────────────────────
# Characters before a component triggers AI auto-summarisation.
context:
  component_size_limits:
    memories: 10000                       # MEMORIES.txt
    pulse: 5000                           # Pulse items (DB)
    skills_total: 20000                   # Total across all skills/*.md

# ── Dreaming (Nightly Consolidation) ─────────────────────────────
dreaming:
  hour: 1                                 # Hour (0-23, UTC) to run consolidation
  minute: 0                               # Minute (0-59)

# ── Scheduler ─────────────────────────────────────────────────────
scheduler:
  timezone: "UTC"                         # Timezone for reminder scheduling (e.g. Europe/Berlin)

# ── Logging ───────────────────────────────────────────────────────
logging:
  level: "INFO"                           # DEBUG | INFO | WARN | ERROR
  directory: "logs"
```

## Section Details

### `inference_backends`

A list of named backend definitions. Each entry describes one LLM endpoint+model
combination that can be referenced by name in the `llm` role mapping.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `name` | yes | — | Unique name for this backend (referenced in `llm` role mapping) |
| `provider` | no | `"openai"` | `"openai"`, `"gemini-cli"`, or `"kimi-code"` |
| `base_url` | openai only | — | OpenAI-compatible API base URL (not needed for `gemini-cli` or `kimi-code`) |
| `api_key` | openai only | — | API key (not needed for `gemini-cli` or `kimi-code`) |
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
| `dreaming` | no | first backend | Nightly memory consolidation |
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
llama-server
  - name: "ollama_small"
    provider: "openai"
    base_url: llama-server//localhost:8080/v1"
    api_key: "ollama"
    model: "qwen2.5:7b"
    context_size: 32768
    max_tokens: 2048

llm:
  base: ["geminillama-server
  compaction: ["ollama_smallllama-server
  sub_sessionsllama-servermini", "ollama_small"]   # failover
  dreaming: ["ollama_small"]

turing_protocollama-server
  backends: ["ollama_small"]
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
```

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
to set up credentials, or select the Gemini option during `setup.sh`.

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
- During `setup.sh`: option 7 runs interactive device-code auth
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

Three-stage post-inference validation pipeline that detects, validates, and
corrects violations in assistant responses.  Fires after each inference round.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `backends` | no | first backend | Ordered list of backend names for the protocol's own LLM calls |
| `validators` | no | all enabled | Per-hook enable/disable overrides (`hook_name: true/false`) |

**Disabling:** Set `backends: []` to disable entirely, or set individual
validators to `false` to suppress specific checks.

**Default behavior:** If the `turing_protocol:` section is omitted entirely,
the protocol defaults to using the base model backends with all validators
enabled.

Currently available validators:

| Validator | Description |
|-----------|-------------|
| `workflow_spawn` | Detects when the model claims to have spawned a session without calling `spawn_sub_session` |
| `phantom_tool_result` | Detects when the model presents fabricated tool output (past tense — "I checked and found…") without having called the tool |
| `empty_promise` | Detects when the model commits to an action ("I'll do X", "Let me check") as a final response without calling any tool |

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

### `web`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Enable/disable the web interface |
| `host` | no | `"127.0.0.1"` | Bind address |
| `port` | no | `8080` | Listen port |

### `pulse`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Enable/disable periodic pulse reviews |
| `review_interval_minutes` | no | `60` | Minutes between automatic pulse reviews |

### `dreaming`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `hour` | no | `1` | Hour (UTC, 0-23) for nightly consolidation |
| `minute` | no | `0` | Minute (0-59) |

### `scheduler`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `timezone` | no | `"UTC"` | Timezone for reminder scheduling |

### `context.component_size_limits`

| Key | Default | Description |
|-----|---------|-------------|
| `memories` | `10000` | Char limit before MEMORIES.txt auto-summarisation |
| `pulse` | `5000` | Char limit before pulse auto-summarisation |
| `skills_total` | `20000` | Total char limit across all skills |
