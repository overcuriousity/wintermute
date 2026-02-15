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

# ── LLM ───────────────────────────────────────────────────────────
# Any OpenAI-compatible endpoint: Ollama, vLLM, LM Studio, OpenAI, etc.
llm:
  base_url: "http://localhost:11434/v1"   # Ollama default
  api_key: "ollama"                        # Use "ollama" for Ollama, actual key for OpenAI etc.
  model: "qwen2.5:72b"                    # Any model name the endpoint accepts
  context_size: 32768                      # Total token window the model supports
  max_tokens: 4096                         # Maximum tokens in a single response
  # Per-purpose overrides. Unspecified fields inherit from the parent llm: block.
  compaction:
    model: "qwen2.5:7b"
    # base_url: "https://api.openai.com/v1"
    # api_key: "sk-..."
    # context_size: 128000
    # max_tokens: 2048
    # reasoning: false
  # sub_sessions:
  #   base_url: "https://api.openai.com/v1"
  #   api_key: "sk-..."
  #   model: "gpt-4o"
  #   context_size: 128000
  #   max_tokens: 8192
  #   reasoning: false
  # dreaming:
  #   base_url: "https://api.openai.com/v1"
  #   api_key: "sk-..."
  #   model: "qwen2.5:7b"
  #   context_size: 32768
  #   max_tokens: 2048
  #   reasoning: false
  # supervisor:                              # Post-inference workflow validation
  #   enabled: true                          # Set to false to disable supervisor checks entirely
  #   model: "qwen2.5:7b"                   # Cheap/fast model recommended (one-shot JSON check)
  #   max_tokens: 150                        # Only needs ~150 tokens for a JSON yes/no response
  #   reasoning: false

# ── Context Compaction ────────────────────────────────────────────
# Compaction fires when history tokens exceed:
#   context_size - max_tokens - system_prompt_tokens
# No separate threshold setting is needed; adjust context_size and max_tokens above.

# ── Pulse Reviews ─────────────────────────────────────────────────
pulse:
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

### `llm`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `provider` | no | `"openai"` | `"openai"` (any OpenAI-compatible endpoint) or `"gemini-cli"` (Google Cloud Code Assist via gemini-cli — **alpha**) |
| `base_url` | openai only | — | OpenAI-compatible API base URL (not needed for `gemini-cli`) |
| `api_key` | openai only | — | API key, use `"ollama"` for Ollama (not needed for `gemini-cli`) |
| `model` | yes | — | Model name the endpoint accepts |
| `context_size` | yes | — | Total token window the model supports |
| `max_tokens` | no | `4096` | Maximum tokens per response |
| `reasoning` | no | `false` | Enable reasoning/thinking token support |
| `compaction` | no | — | Override block for compaction (partial; inherits from parent) |
| `sub_sessions` | no | — | Override block for sub-sessions (partial; inherits from parent) |
| `dreaming` | no | — | Override block for dreaming (partial; inherits from parent) |
| `supervisor` | no | — | Override block for supervisor (partial; inherits from parent) |

#### Provider: `gemini-cli` (Alpha)

> **Alpha:** This provider is experimental. See [installation.md — Gemini CLI Provider](installation.md#gemini-cli-provider-free-google-models--alpha) for known limitations.

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

Mixed configurations work — e.g. `main` on gemini-cli with `compaction` on local Ollama:

```yaml
llm:
  provider: "gemini-cli"
  model: "gemini-2.5-pro"
  context_size: 1048576
  max_tokens: 8192
  compaction:
    provider: "openai"
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen2.5:7b"
    context_size: 32768
```

### `llm.supervisor`

Post-inference workflow validation. A lightweight one-shot LLM check that detects
when the main model claims to have started a background session without actually
calling `spawn_sub_session`. Runs asynchronously after the reply is delivered
(zero added latency on the happy path). Inherits from the parent `llm` block when
no explicit overrides are given; a cheap/fast model override is recommended.

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `enabled` | no | `true` | Set to `false` to disable supervisor checks entirely |
| `base_url` | no | inherits | API endpoint (inherits from parent `llm` block) |
| `api_key` | no | inherits | API key (inherits from parent `llm` block) |
| `model` | no | inherits | Model name; a cheap/fast model is recommended |
| `context_size` | no | inherits | Token window (inherits from parent `llm` block) |
| `max_tokens` | no | inherits | Max response tokens; `150` is sufficient for the JSON check |
| `reasoning` | no | `false` | Enable reasoning token support |

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
