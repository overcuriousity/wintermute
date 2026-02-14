# Installation

## Requirements

- Linux (Fedora / RHEL or Debian / Ubuntu)
- `bash` and `curl` — everything else is installed automatically
- An OpenAI-compatible LLM endpoint
- *(Strongly recommended)* A local [SearXNG](https://docs.searxng.org/) instance — Wintermute's `search_web` tool queries SearXNG for web search. Without it, searches fall back to DuckDuckGo's limited Instant Answer API. SearXNG is lightweight, privacy-respecting, and easy to deploy via Docker.
- *(Recommended)* A dedicated Matrix account for the bot

## Quickstart (recommended)

Clone the repository and run the interactive setup script — it handles everything from system dependencies to a running daemon:

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
bash setup.sh
```

The script walks through 5 stages:

| Stage | What it does |
|-------|-------------|
| **[1/5] System dependencies** | Installs Python 3.12+, curl, uv, build tools, and libolm headers |
| **[2/5] Python environment** | Runs `uv sync`, verifies E2E encryption imports |
| **[3/5] Configuration** | Interactive config: LLM endpoint, web UI, timezone, SearXNG check, Matrix (optional) |
| **[4/5] Systemd service** | Installs a systemd user service with lingering enabled (no sudo needed) |
| **[5/5] Pre-flight diagnostics** | Checks imports, endpoint reachability, Matrix connectivity, E2E deps |

After diagnostics, the script offers to start the daemon immediately. If all checks pass, Wintermute is running as a persistent service.

> **Note:** The script only runs on Fedora/RHEL or Debian/Ubuntu. It will exit on unsupported systems.

### Setup script options

```
bash setup.sh --help        # Show all options
bash setup.sh --dry-run     # Show install plan without making changes
bash setup.sh --no-matrix   # Skip Matrix configuration
bash setup.sh --no-systemd  # Skip systemd service installation
```

### Matrix setup during onboarding

When you choose to enable Matrix, the script collects the bot's **password** — not a token. On first start, Wintermute logs in automatically, creates a device, sets up E2E encryption, cross-signs the device, and saves a recovery key to `data/matrix_recovery.key`. No manual `curl` commands or token pasting required.

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

Open `config.yaml` and fill in at minimum the `llm` section:

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # or your local endpoint
  api_key: "sk-..."
  model: "gpt-4o"
  context_size: 128000
  max_tokens: 4096
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

## Timezone Configuration

Wintermute injects the current local time into every system prompt so the LLM has accurate time awareness. This relies on the `scheduler.timezone` setting in `config.yaml`:

```yaml
scheduler:
  timezone: "Europe/Berlin"   # Your local timezone
```

If running in a container, ensure the container's system clock is accurate (e.g. via NTP). The timezone does **not** need to match the host's `/etc/localtime` — Wintermute uses the configured timezone from `config.yaml` regardless of the system timezone.

## Systemd User Service

The `setup.sh` script installs a systemd **user** service automatically (no sudo required). It also enables lingering via `loginctl enable-linger` so the service starts at boot, not just at login.

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
