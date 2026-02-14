# Installation

## Requirements

- Linux (Fedora / RHEL or Debian / Ubuntu)
- `bash` and `curl` — everything else is installed automatically
- An OpenAI-compatible LLM endpoint
- *(Strongly recommended)* A local [SearXNG](https://docs.searxng.org/) instance — Wintermute's `search_web` tool queries SearXNG for web search. Without it, searches fall back to DuckDuckGo's limited Instant Answer API. SearXNG is lightweight, privacy-respecting, and easy to deploy via Docker.
- *(Recommended)* A dedicated Matrix account for the bot

## Quickstart (recommended)

Clone the repository and run the interactive setup script — it handles everything:

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
bash setup.sh
```

`setup.sh` will:

1. Install Python 3.12+, `uv`, and all Python dependencies
2. Walk you through configuring `config.yaml` (LLM endpoint, Matrix credentials, timezone, ...)
3. Optionally install a **systemd user service** so Wintermute starts on boot
4. Run pre-flight checks (endpoint reachability, package imports, ...)

> **Note:** The script only runs on Fedora/RHEL or Debian/Ubuntu. It will exit on unsupported systems.

## Manual Installation

### 1. Clone the repository

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
```

### 2. Install with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync
```

### 3. Configure

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

Matrix and web sections are optional — if Matrix is omitted the web UI runs standalone. See [configuration.md](configuration.md) for all options.

### 4. Run

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

The `setup.sh` script can install a systemd user service automatically. If you prefer to set it up manually, create `~/.config/systemd/user/wintermute.service`:

```ini
[Unit]
Description=Wintermute AI Assistant
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/wintermute
ExecStart=/path/to/uv run wintermute
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

Then enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable --now wintermute
```
