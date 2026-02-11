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

Requires [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Configure
cp config.yaml.example config.yaml
$EDITOR config.yaml   # set LLM endpoint; Matrix is optional

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

## Matrix Configuration

```yaml
matrix:
  homeserver: https://matrix.org
  user_id: "@bot:matrix.org"
  access_token: "YOUR_TOKEN"
  allowed_users:          # only these users can interact with the bot
    - "@alice:matrix.org"
  allowed_rooms: []       # empty = accept any room invited by an allowed user
```

The bot auto-joins rooms when invited by a user in `allowed_users`. If `allowed_rooms`
is non-empty, only those specific room IDs are accepted.

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
