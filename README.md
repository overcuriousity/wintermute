# Ganglion – Multi-Interface Personal AI Assistant

A Python-based personal AI assistant with persistent memory, reminder scheduling,
and autonomous tool use. Supports multiple Matrix rooms and multiple independent
web sessions, each with its own conversation context.

## Features

- **Multi-thread conversations** – each Matrix room and each browser tab has independent history and context compaction; memories/skills/heartbeats are shared
- **Persistent memory** – `MEMORIES.txt`, `HEARTBEATS.txt`, and `skills/` survive restarts
- **Reminder scheduler** – APScheduler with SQLite persistence; thread-bound or system-level reminders; handles missed reminders on restart
- **Tool-calling AI** – any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI, …) with full tool access (shell, filesystem, scheduling)
- **Heartbeat reviews** – periodic autonomous reviews run globally and per active thread
- **Context compaction** – automatic summarisation when a thread's history grows large
- **Matrix multi-room** – auto-joins rooms on invite from whitelisted users; optional room whitelist
- **Web interface** – built-in chat UI at `http://localhost:8080`; each browser tab is an isolated session
- **Graceful shutdown** – SIGTERM/SIGINT handled cleanly

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and **Python 3.12 or newer**.

The `matrix-nio[e2e]` dependency includes a C extension (`python-olm`) that
requires Python development headers. Install them before running `uv sync`:

```bash
# Fedora / RHEL
sudo dnf install python3-devel

# Debian / Ubuntu
sudo apt install python3-dev
```

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp config.yaml.example config.yaml
$EDITOR config.yaml   # set LLM endpoint and Matrix credentials (see Matrix Setup below)

# 3. Run
uv run ganglion
# Open http://127.0.0.1:8080 in your browser
```

Both Matrix and the web interface are optional — at least one must be enabled.

## Special Commands (in any Matrix room or web UI tab)

| Command      | Effect                                                        |
|--------------|---------------------------------------------------------------|
| `/new`       | Reset conversation for the current room/tab, start fresh     |
| `/compact`   | Force immediate context compaction for the current room/tab  |
| `/reminders` | List all scheduled reminders                                 |
| `/heartbeat` | Manually trigger a heartbeat review                          |

## Matrix Setup (Primary Interface)

Matrix is the recommended primary interface. End-to-end encryption is enabled
automatically, so the bot works in encrypted rooms without any extra steps.

### 1. Create a bot account

Register a dedicated Matrix account for the bot on any homeserver
(e.g. [matrix.org](https://matrix.org), or your own Synapse instance).
You can use Element or any other client for the registration.

### 2. Obtain an access token

Log into the bot account in **Element Web** (app.element.io or your homeserver):

1. **Settings → Help & About → Advanced → Access Token** — copy the token.

Alternatively, use the API directly:

```bash
curl -X POST \
  'https://YOUR_HOMESERVER/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","identifier":{"type":"m.id.user","user":"botusername"},"password":"YOURPASSWORD"}'
```

The response contains `access_token` and `device_id`.

### 3. Obtain the device ID (required)

`device_id` is required by matrix-nio's Olm store. Use the value from the
login response in step 2 — it is returned alongside `access_token`.

Alternatively, find it in Element after logging in as the bot:
**Settings → Security & Privacy → Session list** — copy the Session ID of the
bot's active session.

### 4. Configure `config.yaml`

```yaml
matrix:
  homeserver: https://YOUR_HOMESERVER        # e.g. https://matrix.org
  user_id: "@botusername:YOUR_HOMESERVER"    # full Matrix ID of the bot account
  access_token: "syt_..."                    # access token from step 2
  device_id: "ABCDEFGHIJ"                    # session/device ID from step 3
  allowed_users:                             # Matrix IDs allowed to interact
    - "@you:YOUR_HOMESERVER"
  allowed_rooms: []                          # empty = any room invited by allowed_users
```

### 5. Invite the bot and start chatting

1. Start Ganglion: `uv run ganglion`
2. In Element (or any Matrix client), open or create a room.
3. Invite the bot account: **Invite → @botusername:YOUR_HOMESERVER**
4. The bot auto-joins and is ready to chat immediately.

For an encrypted room (padlock icon), Ganglion handles key exchange
automatically — no manual steps required.

### Access control

| Config field     | Behaviour                                                      |
|------------------|----------------------------------------------------------------|
| `allowed_users`  | Only listed user IDs can invite the bot or send it messages.  |
| `allowed_rooms`  | If non-empty, the bot only joins/responds in listed room IDs. |

Both lists default to empty, which means the bot accepts all users/rooms —
**set `allowed_users` before exposing to a shared homeserver.**

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `uv sync` fails building `python-olm` with `pyconfig.h: No such file or directory` | Python development headers are missing. Install `python3-devel` (Fedora/RHEL) or `python3-dev` (Debian/Ubuntu) and retry. |
| Bot joins but messages show as undecrypted | The bot's Olm keys were not shared yet; send another message and allow a moment for key exchange, or restart the bot once. |
| Bot does not join on invite | Sender is not in `allowed_users`, or room is not in `allowed_rooms`. Check logs. |
| `Initial sync failed` on startup | Wrong `homeserver` URL or expired `access_token`. |
| `matrix.device_id is required` on startup | `device_id` is missing from config. See step 3 above. |
| New Olm session on every restart | `device_id` is wrong or changed between runs. |

## Reminder Types

Reminders created via the `set_reminder` tool can be:

- **Thread-bound** (default) – fires to the Matrix room or web tab where it was created
- **System reminders** (`system: true`) – runs as an autonomous AI inference with no chat delivery; useful for background maintenance tasks

## Project Layout

```
pyproject.toml           – project metadata and dependencies (uv)
config.yaml.example      – configuration template

ganglion/                – Python package
  main.py                – entry point, startup/shutdown orchestration
  database.py            – SQLite helpers for per-thread conversation history
  llm_thread.py          – OpenAI-compatible API calls, conversation history, context compaction
  matrix_thread.py       – Matrix connection, multi-room routing, auto-join
  web_interface.py       – aiohttp HTTP + WebSocket server with embedded chat UI
  scheduler_thread.py    – APScheduler wrapper, thread-bound reminder registry
  heartbeat.py           – global + per-thread heartbeat review loop
  tools.py               – tool schemas and implementations
  prompt_assembler.py    – dynamic system prompt construction from data/ files

data/
  BASE_PROMPT.txt        – immutable core instructions (edit to change personality)
  MEMORIES.txt           – long-term user facts (AI-editable, shared across threads)
  HEARTBEATS.txt         – active goals / working memory (AI-editable, shared)
  skills/                – capability documentation (AI-editable, shared)
  scripts/               – auxiliary scripts created by the AI
  conversation.db        – per-thread message history (gitignored)
  scheduler.db           – APScheduler job store (gitignored)
  reminders.json         – human-readable reminder registry (gitignored)

logs/                    – rotating daily logs (gitignored)
```

## Configuration

See `config.yaml.example` for all available options.

## Architecture

Multiple asyncio tasks run concurrently in a single event loop:

1. **LLM task** – processes messages sequentially from a shared queue; each item carries a `thread_id`; loads and saves per-thread history; executes tool calls
2. **Matrix task** – `sync_forever` loop; routes each room's messages to the LLM queue with the room's `room_id` as `thread_id`; auto-joins on invite
3. **Web task** – aiohttp server; each WebSocket connection gets a `web_<hex>` thread_id
4. **Heartbeat task** – periodically runs a global review + one review per active thread
5. **APScheduler** – runs within the same event loop using `AsyncIOExecutor`; reminders carry a `thread_id` and are delivered to the correct room/tab

A thread-aware `broadcast(text, thread_id)` function routes outgoing messages to the correct Matrix room or web client set.

## Cost / Resource Notes

Cost depends entirely on your chosen endpoint:

- **Local models (Ollama, vLLM, LM Studio)** – no per-call cost; GPU/CPU time only.
  Set `compaction_model` to a smaller model to reduce load for summarisation tasks.
- **OpenAI or hosted APIs** – billed per token; a smaller `compaction_model` helps.

Heartbeat frequency (`review_interval_minutes`) and the number of active threads are
the main tuning knobs for inference volume.
