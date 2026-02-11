# Ganglion – Matrix AI Reminder Assistant

A Python-based personal AI assistant that operates exclusively through Matrix.
It maintains persistent memory, manages reminders, and can autonomously learn
new skills by documenting reusable procedures.

## Features

- **Persistent memory** – MEMORIES.txt, HEARTBEATS.txt, and skills/ survive restarts
- **Reminder scheduler** – APScheduler with SQLite persistence; handles missed reminders on restart
- **Tool-calling AI** – any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenAI, …) with full tool access (shell, filesystem, scheduling)
- **Heartbeat reviews** – periodic autonomous reviews of active goals
- **Context compaction** – automatic summarisation when conversation history grows large
- **Web interface** – built-in chat UI accessible at `http://localhost:8080`; works without Matrix
- **Graceful shutdown** – SIGTERM/SIGINT handled cleanly

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp config.yaml.example config.yaml
$EDITOR config.yaml   # set LLM endpoint; Matrix is optional

# 3. Run
python main.py
# Open http://127.0.0.1:8080 in your browser
```

Both Matrix and the web interface are optional — at least one must be enabled.
Outgoing notifications (reminders, heartbeat results) are broadcast to all
active interfaces simultaneously.

## Special Commands (in Matrix or web UI)

| Command      | Effect                                      |
|--------------|---------------------------------------------|
| `/new`       | Reset conversation, start fresh session     |
| `/compact`   | Force immediate context compaction          |
| `/reminders` | List all scheduled reminders                |
| `/heartbeat` | Manually trigger a heartbeat review         |

## Project Layout

```
main.py              – entry point, startup/shutdown orchestration
matrix_thread.py     – Matrix connection and message routing
llm_thread.py        – OpenAI-compatible API calls, conversation history, context compaction
matrix_thread.py     – Matrix connection and message routing (optional)
web_interface.py     – aiohttp HTTP + WebSocket server with embedded chat UI (optional)
scheduler_thread.py  – APScheduler wrapper, reminder registry
heartbeat.py         – periodic heartbeat review loop
tools.py             – tool schemas and implementations
prompt_assembler.py  – dynamic system prompt construction from data/ files
database.py          – SQLite helpers for conversation history

data/
  BASE_PROMPT.txt    – immutable core instructions (edit to change personality)
  MEMORIES.txt       – long-term user facts (AI-editable)
  HEARTBEATS.txt     – active goals / working memory (AI-editable)
  skills/            – capability documentation (AI-editable)
  scripts/           – auxiliary scripts created by the AI
  conversation.db    – message history (gitignored)
  scheduler.db       – APScheduler job store (gitignored)
  reminders.json     – human-readable reminder registry (gitignored)

logs/                – rotating daily logs (gitignored)
config.yaml.example  – configuration template
```

## Configuration

See `config.yaml.example` for all available options.

## Architecture

Three asyncio tasks run concurrently:

1. **Matrix task** – `sync_forever` loop, routes incoming messages to the LLM queue
2. **LLM task**    – processes messages sequentially, executes tool calls, persists history
3. **Heartbeat task** – sleeps between intervals, submits system-event messages to the LLM

The APScheduler runs within the same event loop using `AsyncIOExecutor`.

## Cost / Resource Notes

Cost depends entirely on your chosen endpoint:

- **Local models (Ollama, vLLM, LM Studio)** – no per-call cost; GPU/CPU time only.
  Set `compaction_model` to a smaller model to reduce load for summarisation tasks.
- **OpenAI or hosted APIs** – billed per token; a smaller `compaction_model` helps keep costs down.

Heartbeat frequency (`review_interval_minutes`) is the main tuning knob for inference volume.
