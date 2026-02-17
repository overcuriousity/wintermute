# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Wintermute

A self-hosted personal AI assistant (named after Neuromancer's AI) with persistent memory, autonomous background workers, and multi-interface support (Matrix with E2E encryption + web UI). Connects to any OpenAI-compatible LLM endpoint. Single-process Python asyncio architecture.

## Commands

```bash
uv sync                              # Install dependencies
uv run wintermute                    # Run the application
uv build                             # Build package

# Production (systemd user service)
bash setup.sh                        # Interactive installer
systemctl --user start wintermute
journalctl --user -u wintermute -f
```

No test suite exists. Configuration: copy `config.yaml.example` to `config.yaml`.

## Architecture

**Entry point:** `wintermute/main.py` → `asyncio.run(main())` starts all components as concurrent asyncio tasks.

**Core loop:** `LLMThread` owns the conversation history and an asyncio queue. User messages (from Matrix or web) are enqueued, processed through inference + tool call loops, with responses sent back to the originating interface.

**Key modules:**

| Module | Purpose |
|---|---|
| `llm_thread.py` | Inference engine, conversation history, context compaction, tool dispatch, Turing Protocol dispatch |
| `tools.py` | 12 tool definitions (OpenAI function-calling schemas) + `execute_tool()` dispatcher |
| `sub_session.py` | Background worker DAG: `TaskNode`/`Workflow` with `depends_on` edges, nested workers (depth 2), timeout continuation |
| `prompt_assembler.py` | Assembles system prompt per-turn: BASE_PROMPT + datetime + MEMORIES + pulse + skills |
| `prompt_loader.py` | Loads/validates prompt templates from `data/prompts/` (11 required files; missing = startup failure) |
| `turing_protocol.py` | Post-inference 3-stage pipeline: detect → validate → correct (injected back into LLM queue) |
| `matrix_thread.py` | Matrix client (mautrix) with E2E encryption |
| `web_interface.py` | aiohttp server: WebSocket chat, debug panel (`/debug`), REST API |
| `dreaming.py` | Nightly memory consolidation (direct LLM call, no tool loop) |
| `pulse.py` | Periodic autonomous pulse reviews via sub-session |
| `scheduler_thread.py` | APScheduler-based reminders; `ai_prompt` triggers sub-sessions |
| `database.py` | SQLite ops: messages, pulse, summaries, interaction_log |

**LLM provider abstraction:** `BackendPool` wraps `AsyncOpenAI` with ordered failover. Three provider types: `"openai"` (any compatible endpoint), `"gemini-cli"`, `"kimi-code"`. Role-based routing (`base`, `compaction`, `sub_sessions`, `dreaming`, `turing_protocol`).

**Tool categories:** Tools are filtered for sub-sessions — `execution`+`research` always available; `orchestration` only for `full`-mode workers.

**Context compaction:** When history exceeds token budget, older messages are summarized into a rolling summary (chained across cycles), archived in DB (30-day retention).

## Data Layout

- `data/prompts/*.txt` — All prompt templates (externalized, not hardcoded)
- `data/MEMORIES.txt` — Long-term memory (append-based, consolidated nightly)
- `data/conversation.db` — SQLite: messages, summaries, pulse, interaction_log
- `data/skills/*.md` — Learned procedures
- `data/TURING_PROTOCOL_HOOKS.txt` — Hook definitions (JSON)
- `config.yaml` — Runtime config (gitignored)

## Key Patterns

- System prompt is reassembled fresh every turn via `PromptAssembler`
- Sub-sessions use a DAG with event-driven dependency resolution (`_resolve_dependents()`)
- Turing Protocol corrections are sequence-numbered; stale corrections are silently dropped; max 2 re-checks per turn
- Database migrations are applied inline at startup via `ALTER TABLE ... ADD COLUMN`
- Slash commands (`/new`, `/compact`, `/pulse`, `/status`, `/dream`, etc.) are handled at the interface layer before reaching the LLM
