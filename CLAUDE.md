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
bash onboarding.sh                   # AI-driven installer (experimental)
bash setup.sh                        # Classic programmatic installer
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
| `prompt_assembler.py` | Assembles system prompt per-turn: BASE_PROMPT + datetime + MEMORIES + agenda + skills TOC (full skills loaded on demand via read_file) |
| `prompt_loader.py` | Loads/validates prompt templates from `data/prompts/` (required files; missing = startup failure) |
| `turing_protocol.py` | Phase-aware 3-stage pipeline: detect → validate → correct. Phases: `post_inference`, `pre_execution`, `post_execution`. Scoped to `main` and/or `sub_session`. |
| `matrix_thread.py` | Matrix client (mautrix) with E2E encryption; voice messages transcribed via configurable Whisper endpoint |
| `web_interface.py` | aiohttp server: WebSocket chat, debug panel (`/debug`), REST API |
| `dreaming.py` | Nightly memory consolidation (direct LLM call, no tool loop) |
| `agenda.py` | Periodic autonomous agenda reviews via sub-session |
| `memory_harvest.py` | Periodic conversation mining → MEMORIES.txt extraction via sub-sessions |
| `scheduler_thread.py` | APScheduler-based routines; `ai_prompt` triggers sub-sessions |
| `database.py` | SQLite ops: messages, agenda, summaries, interaction_log |

**LLM provider abstraction:** `BackendPool` wraps `AsyncOpenAI` with ordered failover. Four provider types: `"openai"` (any compatible endpoint), `"anthropic"` (native Messages API with prompt caching), `"gemini-cli"`, `"kimi-code"`. Role-based routing (`base`, `compaction`, `sub_sessions`, `dreaming`, `turing_protocol`).

**Tool categories:** Tools are filtered for sub-sessions — `execution`+`research` always available; `orchestration` only for `full`-mode workers. `spawn()` also accepts an explicit `tool_names` list to bypass categories (used by memory harvest).

**Context compaction:** When history exceeds token budget, older messages are summarized into a rolling summary (chained across cycles), archived in DB (30-day retention).

## Data Layout

- `data/` has its own local git repo for auto-versioning; mutations to memories and skills are auto-committed for rollback (`cd data && git log`)
- `data/prompts/*.txt` — All prompt templates (externalized, not hardcoded); seed prompts are per-language (`SEED_en.txt`, `SEED_de.txt`, ...)
- `data/MEMORIES.txt` — Long-term memory (append-based, consolidated nightly)
- `data/conversation.db` — SQLite: messages, summaries, agenda, interaction_log
- `data/skills/*.md` — Learned procedures (first line = summary for TOC; full content loaded on demand)
- `data/TURING_PROTOCOL_HOOKS.txt` — Hook definitions (JSON)
- `config.yaml` — Runtime config (gitignored)

## Key Patterns

- System prompt is reassembled fresh every turn via `PromptAssembler`
- Sub-sessions use a DAG with event-driven dependency resolution (`_resolve_dependents()`)
- Turing Protocol hooks have `phase` (post_inference/pre_execution/post_execution) and `scope` (main/sub_session) fields. Main thread uses async correction injection; sub-sessions use synchronous inline injection. `objective_completion` hook gates sub-session exit with LLM-based evaluation. Each hook fires at most once per turn (single-shot, no escalation). Stage 2 programmatic validators catch false positives (e.g. responses ending with `?` are not empty promises).
- Database migrations are applied inline at startup via `ALTER TABLE ... ADD COLUMN`
- Slash commands (`/new`, `/compact`, `/agenda`, `/status`, `/dream`, etc.) are handled at the interface layer before reaching the LLM
- all of the application architecture is aimed to optimize it to work even with weak/small models like ministral-3:8b in the backend.
