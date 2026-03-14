# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Wintermute

A self-hosted AI runtime environment (named after Neuromancer's AI) with persistent memory, autonomous background workers, and multi-interface support (Matrix with E2E encryption + web UI). Connects to any OpenAI-compatible LLM endpoint. Single-process Python asyncio architecture.

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
| `llm_thread.py` | Conversation history, context compaction, Convergence Protocol dispatch; delegates per-tool-call execution to `inference_engine` |
| `inference_engine.py` | Shared tool-call pipeline (`process_tool_call`): JSON parse, NL translation, CP pre/post execution gates, tool dispatch, interaction logging — used by both `llm_thread` and `sub_session` |
| `tools.py` | Tool definitions (OpenAI function-calling schemas) + `execute_tool()` dispatcher |
| `sub_session.py` | Background worker DAG: `TaskNode`/`Workflow` with `depends_on` edges, nested workers (depth 2), timeout continuation |
| `prompt_assembler.py` | Assembles system prompt per-turn: BASE_PROMPT + datetime + memories (vector search) + tasks + skills TOC (full skills loaded on demand via read_file) |
| `prompt_loader.py` | Loads/validates prompt templates from `data/prompts/` (required files; missing = startup failure) |
| `convergence_protocol.py` | Phase-aware 3-stage pipeline: detect → validate → correct. Phases: `post_inference`, `pre_execution`, `post_execution`. Scoped to `main` and/or `sub_session`. |
| `matrix_thread.py` | Matrix client (mautrix) with E2E encryption; voice messages transcribed via configurable Whisper endpoint |
| `web_interface.py` | aiohttp server: WebSocket chat, debug panel (`/debug`), REST API |
| `dreaming.py` | Nightly memory consolidation: 4-phase housekeeping pipeline (dedup, contradictions, stale pruning, working set export) + 3 creative phases |
| `memory_harvest.py` | Periodic conversation mining → memory store extraction via sub-sessions |
| `scheduler_thread.py` | APScheduler-based task scheduling; `ai_prompt` triggers sub-sessions |
| `database.py` | SQLite ops (per-thread cached connections): messages, tasks, summaries, interaction_log |

**LLM provider abstraction:** `BackendPool` wraps `AsyncOpenAI` with ordered failover. Four provider types: `"openai"` (any compatible endpoint), `"anthropic"` (native Messages API with prompt caching), `"gemini-cli"`, `"kimi-code"`. Role-based routing (`base`, `compaction`, `sub_sessions`, `dreaming`, `convergence_protocol`).

**Tool categories:** Tools are filtered for sub-sessions — `execution`+`research` always available; `orchestration` only for `full`-mode workers. `spawn()` also accepts an explicit `tool_names` list to bypass categories (used by memory harvest).

**Context compaction:** When history exceeds token budget, older messages are summarized into a rolling summary (chained across cycles), archived in DB (30-day retention).

## Data Layout

- `data/` has its own local git repo for auto-versioning; mutations to memories and skills are auto-committed for rollback (`cd data && git log`)
- `data/prompts/*.txt` — All prompt templates (externalized, not hardcoded); seed prompts are per-language (`SEED_en.txt`, `SEED_de.txt`, ...)
- Memory store (vector-indexed) — long-term user facts stored in the configured backend (local_vector/qdrant); embeddings endpoint required
- `data/conversation.db` — SQLite: messages, summaries, tasks, interaction_log
- `data/skills/` — Learned procedures (vector-indexed via skill store; legacy `*.md` files migrated at first startup)
- `data/scratchpad/{workflow_id}/` — Per-workflow directories for parallel worker communication (preserved after completion for later reference; overwritten if a new workflow reuses the same ID)
- `data/CONVERGENCE_PROTOCOL_HOOKS.txt` — Hook definitions (JSON)
- `config.yaml` — Runtime config (gitignored)

## Key Patterns

- System prompt is reassembled fresh every turn via `PromptAssembler`
- Sub-sessions use a DAG with event-driven dependency resolution (`_resolve_dependents()`)
- Multi-node workflows get a scratchpad directory (`data/scratchpad/{workflow_id}/`) where parallel workers can share intermediate findings via `read_file`/`write_file` — each worker writes to its own namespaced file, reads sibling files; directory is auto-cleaned on workflow completion
- Convergence Protocol hooks have `phase` (post_inference/pre_execution/post_execution) and `scope` (main/sub_session) fields. Main thread uses async correction injection; sub-sessions use synchronous inline injection. `objective_completion` hook gates sub-session exit with LLM-based evaluation. Each hook fires at most once per turn (single-shot, no escalation). Stage 2 programmatic validators catch false positives (e.g. responses ending with `?` are not empty promises).
- Database migrations are applied inline at startup via `ALTER TABLE ... ADD COLUMN`
- Slash commands (`/new`, `/compact`, `/tasks`, `/status`, `/dream`, etc.) are handled at the interface layer before reaching the LLM
- all of the application architecture is aimed to optimize it to work even with weak/small models like ministral-3:8b in the backend.
