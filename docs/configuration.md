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
  # Optional: use a smaller/faster model for context compaction summarisation.
  # Falls back to `model` if not set.
  compaction_model: "qwen2.5:7b"

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
  # Optional: use a dedicated model for dreaming.
  # Falls back to compaction_model, then model.
  # model: "qwen2.5:7b"

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
| `base_url` | yes | — | OpenAI-compatible API base URL |
| `api_key` | yes | — | API key (use `"ollama"` for Ollama) |
| `model` | yes | — | Model name the endpoint accepts |
| `context_size` | yes | — | Total token window the model supports |
| `max_tokens` | no | `4096` | Maximum tokens per response |
| `compaction_model` | no | same as `model` | Cheaper model for context compaction |

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
| `model` | no | compaction_model or model | Dedicated model for dreaming |

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
