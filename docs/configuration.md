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
    provider: "openai"                    # "openai" or "gemini-cli"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen2.5:72b"
    context_size: 32768
    max_tokens: 4096
    reasoning: false

  - name: "local_small"
    provider: "openai"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
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

  # - name: "supervisor_backend"
  #   provider: "openai"
  #   base_url: "http://localhost:11434/v1"
  #   api_key: "ollama"
  #   model: "qwen2.5:7b"
  #   context_size: 32768
  #   max_tokens: 150

# ── LLM Role Mapping ──────────────────────────────────────────────
# Maps roles to ordered lists of backend names.
# Multiple backends = automatic failover on API errors.
# Empty list [] = disabled (only meaningful for supervisor).
# Omitted roles default to the first defined backend.
llm:
  base: ["local_large"]
  compaction: ["local_small", "local_large"]
  sub_sessions: ["local_large"]
  dreaming: ["local_small"]
  supervisor: ["local_small"]             # Use [] to disable

# ── Context Compaction ────────────────────────────────────────────
# Compaction fires when history tokens exceed:
#   context_size - max_tokens - system_prompt_tokens
# No separate threshold setting is needed; adjust in the backend definition.

# ── Pulse Reviews ─────────────────────────────────────────────────
pulse:
  enabled: true                           # Set to false to disable pulse reviews
  review_interval_minutes: 60             # How often to auto-review PULSE.txt

# ── Component Size Limits ─────────────────────────────────────────
# Characters before a component triggers AI auto-summarisation.
context:
  component_size_limits:
    memories: 10000                       # MEMORIES.txt
    pulse: 5000                           # PULSE.txt
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
| `provider` | no | `"openai"` | `"openai"` (any OpenAI-compatible endpoint) or `"gemini-cli"` |
| `base_url` | openai only | — | OpenAI-compatible API base URL (not needed for `gemini-cli`) |
| `api_key` | openai only | — | API key (not needed for `gemini-cli`) |
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
| `supervisor` | no | first backend | Post-inference workflow validation; `[]` to disable |

**Failover:** When multiple backends are listed, they are tried in order on
API errors. The first backend's `context_size` is used for token budget
calculations.

**Disabling a role:** Set `supervisor: []` to disable supervisor checks
entirely. Other roles should not be empty.

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
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen2.5:7b"
    context_size: 32768
    max_tokens: 2048

llm:
  base: ["gemini"]
  compaction: ["ollama_small"]
  sub_sessions: ["gemini", "ollama_small"]   # failover
  dreaming: ["ollama_small"]
  supervisor: ["ollama_small"]
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

### `llm.supervisor`

Post-inference workflow validation. A lightweight one-shot LLM check that detects
when the main model claims to have started a background session without actually
calling `spawn_sub_session`. Runs asynchronously after the reply is delivered
(zero added latency on the happy path).

To disable, set `supervisor: []` in the `llm` section. For optimal cost, create
a dedicated backend with `max_tokens: 150` (the supervisor only produces a small
JSON response).

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
| `pulse` | `5000` | Char limit before PULSE.txt auto-summarisation |
| `skills_total` | `20000` | Total char limit across all skills |
