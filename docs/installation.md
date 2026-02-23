# Installation

## Requirements

- Linux (Fedora / RHEL or Debian / Ubuntu)
- `bash` and `curl` — everything else is installed automatically
- An OpenAI-compatible LLM endpoint
- *(Strongly recommended)* A local [SearXNG](https://docs.searxng.org/) instance — Wintermute's `search_web` tool queries SearXNG for web search. Without it, searches fall back to DuckDuckGo's limited Instant Answer API. SearXNG is lightweight, privacy-respecting, and easy to deploy via Docker.
- *(Recommended)* A dedicated Matrix account for the bot

## Quickstart — AI-Driven Onboarding (experimental)

Clone the repository and run the AI-driven onboarding script:

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
bash onboarding.sh
```

The script works in two phases:

**Phase 1 (bash):** Installs system dependencies (Python 3.12+, curl, uv, build tools, libolm, ffmpeg), runs `uv sync`, and asks for your primary LLM endpoint (URL, model, API key). It validates that the endpoint is reachable and supports function calling.

**Phase 2 (AI-driven):** Hands off to an AI configuration assistant powered by your own LLM. The assistant walks you through every `config.yaml` section conversationally:

- Inference backends and LLM role mapping
- Web interface settings
- Matrix integration (with live credential testing and test message delivery)
- Whisper voice transcription
- Turing Protocol validators
- NL Translation
- Agenda, dreaming, memory harvest, scheduler, logging
- Systemd service installation

The AI gives recommendations, explains trade-offs, and runs in-flight validation (probing endpoints, testing Matrix login, triggering OAuth flows for Gemini/Kimi). Config values are written incrementally, so partial progress is preserved if you abort.

> **Experimental:** The AI-driven onboarding requires a model with function-calling support (e.g. Qwen 2.5, Llama 3.1+, GPT-4, Gemini). The script tests this before handoff and warns if unsupported.

### Onboarding script options

```
bash onboarding.sh --help        # Show all options
bash onboarding.sh --dry-run     # Show install plan without making changes
```

### What the AI tests during onboarding

| Test | When |
|------|------|
| LLM endpoint reachability | After you provide the URL |
| Function-calling capability | Before handoff to AI assistant |
| Matrix homeserver reachability | When configuring Matrix |
| Matrix credential validation | Login test + immediate logout |
| Matrix message delivery | Optional test message to a room |
| Gemini OAuth | If gemini-cli provider is selected |
| Kimi-Code device-code auth | If kimi-code provider is selected |

## Classic Setup Script (fallback)

The previous programmatic setup script is retained as `setup.sh`. Use it if your model doesn't support function calling or you prefer a non-AI workflow:

```bash
bash setup.sh
```

It walks through 5 stages (dependencies, Python environment, configuration, systemd, diagnostics) with traditional menu-driven prompts. See `bash setup.sh --help` for options (`--no-matrix`, `--no-systemd`, `--dry-run`).

### Matrix setup during onboarding

When you choose to enable Matrix (via either script), the bot's **password** is collected — not a token. On first start, Wintermute logs in automatically, creates a device, sets up E2E encryption, cross-signs the device, and saves a recovery key to `data/matrix_recovery.key`. No manual `curl` commands or token pasting required.

Some homeservers (e.g. matrix.org) require a one-time browser approval for cross-signing on first start — Wintermute logs the exact URL. After that, everything is fully automatic, including token refresh on expiry.

See [matrix-setup.md](matrix-setup.md) for details on E2E encryption, SAS verification, and troubleshooting.

## Manual Installation

### 1. Clone the repository

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
```

### 2. Install system dependencies

E2E encryption requires libolm headers:

```bash
# Fedora / RHEL
sudo dnf install -y gcc gcc-c++ cmake make libolm-devel python3-devel

# Debian / Ubuntu
sudo apt-get install -y build-essential cmake libolm-dev python3-dev
```

### 3. Install with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync
```

### 4. Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and fill in at minimum the `inference_backends` and `llm` sections:

```yaml
inference_backends:
  - name: "main"
    provider: "openai"
    base_url: "https://api.openai.com/v1"   # or your local endpoint
    api_key: "sk-..."
    model: "gpt-4o"
    context_size: 128000
    max_tokens: 4096

llm:
  base: ["main"]
```

For Matrix, supply the bot's **password** — Wintermute handles login and device creation automatically:

```yaml
matrix:
  homeserver: https://matrix.org
  user_id: "@wintermute:matrix.org"
  password: "bot-account-password"
  access_token: ""                    # auto-filled on first start
  device_id: ""                       # auto-filled on first start
  allowed_users:
    - "@you:matrix.org"
```

See [configuration.md](configuration.md) for all options.

### 5. Run

```bash
uv run wintermute
```

The web interface starts at `http://127.0.0.1:8080` by default.

## Anthropic Provider (Claude API)

Uses Anthropic's native Messages API with prompt caching. Requires a paid API key
from [console.anthropic.com](https://console.anthropic.com/) — pay-per-token billing.
Claude Pro/Max subscriptions do **not** include API access.

### Setup via setup script

Run `bash setup.sh` and select option **8) Anthropic** when prompted. The script will:

1. Ask for your API key (`sk-ant-api03-...`)
2. Prompt for a model (default: `claude-sonnet-4-20250514`)
3. Write `config.yaml` with `provider: "anthropic"`

### Manual setup

```yaml
inference_backends:
  - name: "claude"
    provider: "anthropic"
    api_key: "sk-ant-api03-..."
    model: "claude-sonnet-4-20250514"
    context_size: 200000
    max_tokens: 8192

llm:
  base: ["claude"]
```

Available models: `claude-sonnet-4-20250514` (recommended), `claude-opus-4-20250514`,
`claude-haiku-4-20250414` (good for background tasks — compaction, dreaming, validation).

## Gemini CLI Provider (Free Google Models) — Alpha

> **Alpha:** The Gemini Cloud Code Assist integration is experimental. Known
> limitations include aggressive rate limiting from Google's API and occasional
> tool-call parsing issues. For production use, an OpenAI-compatible endpoint
> (llama-server, vLLM, OpenAI, etc.) is recommended.

> **Unstable / Alpha** — The `gemini-cli` provider piggybacks on Google's
> Cloud Code Assist OAuth flow. Credentials may expire unpredictably and
> the upstream API surface may change without notice. Suitable for
> experimentation; not recommended as your only backend.

Wintermute can use Google's Gemini models for free via the Cloud Code Assist API,
using credentials from a locally-installed `gemini-cli`.

### Prerequisites

- **Node.js** and **npm** (for installing gemini-cli)

### Setup via onboarding script (recommended)

Run `bash onboarding.sh` and select option **6) Gemini (via gemini-cli)** when prompted
for the inference substrate. The script will:

1. Check for (or install) gemini-cli
2. Prompt for a model (default: `gemini-2.5-pro`)
3. Run the OAuth flow (opens your browser for Google sign-in)
4. Write `config.yaml` with `provider: "gemini-cli"`

### Manual setup

```bash
# 1. Install gemini-cli
npm install -g @google/gemini-cli

# 2. Run the OAuth setup
uv run python -m wintermute.gemini_auth

# 3. Configure config.yaml
```

```yaml
inference_backends:
  - name: "gemini"
    provider: "gemini-cli"
    model: "gemini-2.5-pro"
    context_size: 1048576
    max_tokens: 8192

llm:
  base: ["gemini"]
```

Available models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-pro-preview`, `gemini-3-flash-preview`.

Credentials are saved to `data/gemini_credentials.json` and refreshed automatically.

### Headless / SSH OAuth flow

On headless systems (no display server), the OAuth setup detects the missing `DISPLAY`/`WAYLAND_DISPLAY` and switches to a manual flow:

1. Run `uv run python -m wintermute.gemini_auth`
2. The script prints a Google OAuth URL — copy it and open it in a browser on any machine
3. Sign in with your Google account and authorize access
4. Your browser will redirect to `http://localhost:8085/...` which won't load on a headless system — that's expected
5. Copy the **full redirect URL** from your browser's address bar (including the `?code=` parameter)
6. Paste it back into the terminal prompt

If credentials expire or become invalid while running, Wintermute shows a message in the chat suggesting to re-run the auth setup. Re-run `uv run python -m wintermute.gemini_auth` and restart the service.

### Systemd / headless service

When running as a systemd service, NVM paths are **not** in `PATH` by default (systemd uses a minimal environment). Wintermute automatically probes common installation paths at startup:

- `~/.nvm/versions/node/*/bin/gemini` (NVM — default)
- `~/.local/share/nvm/versions/node/*/bin/gemini` (NVM — XDG)
- `~/.volta/bin/gemini` (Volta)
- `~/.local/bin/gemini` (pipx / manual)
- `/usr/local/bin/gemini` (system-wide npm)

If your Node installation is in an unusual location, add the node `bin` directory to the service's environment:

```ini
[Service]
Environment=PATH=/path/to/node/bin:%h/.local/bin:/usr/local/bin:/usr/bin
```

## Kimi-Code Provider ($19/mo Subscription)

Wintermute supports [Kimi-Code](https://kimi.com) as an inference backend. Kimi-Code
provides an OpenAI-compatible endpoint via a flat-rate subscription, authenticated with
OAuth device-code flow.

### Setup via onboarding script

Run `bash onboarding.sh` and select option **7) Kimi-Code** when prompted. The script will:

1. Prompt for a model (default: `kimi-for-coding`)
2. Run the device-code auth flow (prints a URL — open it in any browser)
3. Write `config.yaml` with `provider: "kimi-code"`

### Manual setup

```bash
# 1. Run device-code auth
uv run python -m wintermute.kimi_auth

# 2. Configure config.yaml
```

```yaml
inference_backends:
  - name: "kimi"
    provider: "kimi-code"
    model: "kimi-for-coding"
    context_size: 131072
    max_tokens: 8192

llm:
  base: ["kimi"]
```

Available models include `kimi-for-coding` (default), `kimi-code`, and `kimi-k2.5`
(supports reasoning — set `reasoning: true`). The full list is dynamic; see
[configuration.md](configuration.md#provider-kimi-code) for details.

### Authentication

Credentials are stored in `data/kimi_credentials.json` and tokens are refreshed
automatically. If credentials are missing on startup, Wintermute auto-triggers the
device flow and broadcasts the verification URL to connected interfaces.

You can also authenticate manually via the `/kimi-auth` command in Matrix or the web UI.

## Memory Storage (Vector Search)

By default, Wintermute uses `local_vector` for semantic memory search. This requires only an OpenAI-compatible embeddings endpoint (no external database).

- **`local_vector`** — SQLite + numpy. No external services beyond an embeddings endpoint. Recommended for most deployments.
- **`qdrant`** — Qdrant vector database. Only needed if you want a dedicated vector DB for larger-scale deployments.

To run Qdrant locally via Docker:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

See the [Qdrant documentation](https://qdrant.tech/documentation/quick-start/) for more options.

If no embeddings endpoint is configured, Wintermute falls back to `flat_file` (plain text, no ranking).

## Timezone Configuration

Wintermute injects the current local time into every system prompt so the LLM has accurate time awareness. This relies on the `scheduler.timezone` setting in `config.yaml`:

```yaml
scheduler:
  timezone: "Europe/Berlin"   # Your local timezone
```

If running in a container, ensure the container's system clock is accurate (e.g. via NTP). The timezone does **not** need to match the host's `/etc/localtime` — Wintermute uses the configured timezone from `config.yaml` regardless of the system timezone.

## Systemd User Service

Both `onboarding.sh` and `setup.sh` install a systemd **user** service automatically (no sudo required). It also enables lingering via `loginctl enable-linger` so the service starts at boot, not just at login.

If you prefer to set it up manually, create `~/.config/systemd/user/wintermute.service`:

```ini
[Unit]
Description=Wintermute AI Assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/wintermute
ExecStart=/path/to/uv run wintermute
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
```

Then enable and start:

```bash
systemctl --user daemon-reload
loginctl enable-linger $USER
systemctl --user enable --now wintermute
```

Control the service:

```bash
systemctl --user start wintermute
systemctl --user stop wintermute
systemctl --user restart wintermute
journalctl --user -u wintermute -f
```
