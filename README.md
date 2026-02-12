# Wintermute

![Wintermute](static/Gemini_Generated_Image_7cdpwp7cdpwp7cdp.png)

**Wintermute** is a self-hosted personal AI assistant with persistent memory, autonomous background workers, and multi-interface support. It connects to any OpenAI-compatible LLM endpoint and reaches you via Matrix chat or a built-in web UI (for debuggging mainly).

---

## Concept

Wintermute accumulates knowledge about you over time, maintains an active working memory (*Pulse*), and learns reusable procedures as *skills*. Conversations across restarts are summarised and retained. A nightly *dreaming* pass consolidates memories autonomously while you sleep.

For long-running or complex tasks, Wintermute spawns isolated background workers (*sub-sessions*) so the main conversation stays responsive. Workers can themselves spawn further workers for parallelisable tasks, up to a configurable nesting depth.

The approach differs from different similar concepts by treating small LLMs and digital independence not as an afterthought, but from the beginning. It starts without corporate bloatware and emphasizes FOSS.

---

## Features

- **Persistent memory** — `MEMORIES.txt` (long-term facts), `PULSE.txt` (active goals / working memory), and `skills/*.md` (reusable procedures) survive restarts and are injected into every prompt
- **Multi-interface** — Matrix chat (with E2E encryption) and a browser-based web UI run simultaneously; each room / tab has independent conversation history
- **Sub-session workers** — long-running tasks are delegated to autonomous background agents that report back when done; the main agent stays responsive during execution; workers auto-resume after timeouts (up to 3 hops)
- **Tool-filtered workers** — minimal workers receive only execution + research tools; `full`-mode workers get orchestration tools too, keeping context lean
- **Web search** — `search_web` queries a local SearXNG instance and falls back to DuckDuckGo via `curl` when SearXNG is unavailable; `fetch_url` fetches and strips any web page
- **Reminders & scheduler** — one-time and recurring reminders with optional AI inference on trigger; per-timezone scheduling
- **Nightly dreaming** — automatic overnight consolidation of MEMORIES.txt and PULSE.txt via a direct LLM call (no tool loop, no conversation side effects)
- **Pulse reviews** — periodic autonomous reviews of PULSE.txt; fires globally and per active conversation thread
- **Context compaction** — when conversation history approaches the model's context window, older messages are summarised and retained as a rolling summary
- **Debug panel** — `http://localhost:8080/debug` provides a live view of sessions, sub-sessions, scheduled jobs, reminders, and the current system prompt
- **Any OpenAI-compatible backend** — Ollama, vLLM, LM Studio, OpenAI, or any compatible endpoint

---

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An OpenAI-compatible LLM endpoint
- *(Recommended)* A Matrix account for the bot

---

## Installation

### 1. Clone the repository

```bash
git clone https://git.mikoshi.de/overcuriousity/ganglion.git wintermute
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

Copy and edit the configuration file:

```bash
cp config.yaml.example config.yaml   # if an example exists, otherwise edit config.yaml directly
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

Matrix and web sections are optional — if Matrix is omitted the web UI runs standalone.

### 4. Run

```bash
uv run wintermute
```

The web interface starts at `http://127.0.0.1:8080` by default.

---

## Matrix Setup

### Create a dedicated Matrix account

Register a new account for the bot on your homeserver (e.g. via Element or the homeserver's registration page). The bot needs its own account — do not reuse your personal one.

### Obtain an access token and device ID

Log in once via curl to retrieve the credentials Wintermute needs:

```bash
curl -s -X POST \
  'https://matrix.org/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "m.login.password",
    "identifier": {
      "type": "m.id.user",
      "user": "@your-bot-name:matrix.org"
    },
    "password": "your-password",
    "initial_device_display_name": "Wintermute"
  }' | python3 -m json.tool
```

The response contains:

```json
{
  "access_token": "mct_...",
  "device_id": "ABCDEFGHIJ",
  "user_id": "@your-bot-name:matrix.org",
  ...
}
```

Copy `access_token` and `device_id` into `config.yaml`:

```yaml
matrix:
  homeserver: https://matrix.org
  user_id: "@your-bot-name:matrix.org"
  access_token: "mct_..."
  device_id: "ABCDEFGHIJ"
  allowed_users:
    - "@you:matrix.org"       # Your personal Matrix ID
  allowed_rooms: []           # Empty = any room you invite the bot to
```

### Invite the bot and start chatting

1. Start Wintermute: `uv run wintermute`
2. In Element (or any Matrix client), create a room or open a DM
3. Invite `@your-bot-name:matrix.org`
4. The bot joins and responds to messages from `allowed_users`

**End-to-end encryption** is handled automatically — the bot's Olm/Megolm keys are persisted to `data/matrix_store/` so they survive restarts. Incoming SAS verification requests from allowed users are auto-accepted.

---

## Web Search Setup (SearXNG)

`search_web` works immediately via a DuckDuckGo fallback, but for best results install SearXNG locally:

```bash
cd ~
git clone https://github.com/searxng/searxng.git searxng-test
cd searxng-test
# follow SearXNG quickstart or use the skills/searxng_installation.md guide
```

By default Wintermute expects SearXNG at `http://127.0.0.1:8888`. Override with:

```bash
export WINTERMUTE_SEARXNG_URL=http://127.0.0.1:8888
```

You can also pin searxng up via docker, which might be easier.

---

## Special Commands

Available in both Matrix and the web UI:

| Command | Effect |
|---------|--------|
| `/new` | Reset conversation history for the current thread |
| `/compact` | Force context compaction now |
| `/reminders` | List all scheduled reminders |
| `/pulse` | Manually trigger a pulse review |
*more to come*

---

## Security Disclaimer

The security approach emphasizes the user to be aware of the risks about running this potentially dangerous application.
It can manipulate the system it runs on with all permissions the user has under which the application is executed.
**It is not recommended to run this on your personal workstation or anywhere where you hold sensitive data.**
I recommend running this in a dedicated VM or sandboxed container environment (LXC recommended, docker in the making).

---

## Architecture

```
User (Matrix / Browser)
        │
        ▼
  LLMThread  ←─── system prompt (BASE + MEMORIES + PULSE + SKILLS)
  (asyncio)        assembled fresh each turn
        │
        ├── tool calls ──► execute_shell / read_file / write_file
        │                  search_web / fetch_url
        │                  update_memories / update_pulse / add_skill
        │                  set_reminder / list_reminders
        │
        └── spawn_sub_session ──► SubSessionManager
                                        │
                                        ├── asyncio.Task (worker 1)
                                        ├── asyncio.Task (worker 2)  [parallel]
                                        └── ...
                                              │
                                              └── result ──► enqueue_system_event
                                                              (back to LLMThread)

PulseLoop ───────────────────────────────► fire-and-forget sub-session (full mode)
ReminderScheduler ──────────────────────► LLMThread queue / sub-session
DreamingJob (nightly) ──────────────────► direct LLM API call (no tool loop)
```



## License

MIT
