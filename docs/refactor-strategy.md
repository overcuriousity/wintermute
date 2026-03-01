# Refactor Strategy: Issues #79–#108

**Status as of 2026-03-01** — Phase 2 complete (#104 + #83). This document covers the remaining 5 open issues and their implementation order.

## Completed (Phase 1)

All Phase 1 and Phase 2 issues have been merged:

| Issue | Summary | PR |
|---|---|---|
| #84 | Protocol/ABC for LLM backends | Merged |
| #86 | Slash command dispatcher extracted | Merged |
| #87 | Tool schemas extracted to `core/tool_schemas.py` | Merged |
| #88 | Types module (`core/types.py`) for shared data classes | Merged |
| #93 | `spawn()` helpers extracted in `sub_session.py` | Merged |
| #99 | `make_tool_context()` factory function | Merged |
| #100 | Unified TP runner (`core/tp_runner.py`) | Merged |
| #108 | Magic numbers exposed in `config.yaml` | Merged |
| #104 | `_process()` compaction retry extracted to `_run_inference_with_retry()` | Merged |
| #83 | Module-level DI globals replaced with `ToolDeps` dataclass | Merged |

## Remaining Open Issues (5)

| # | Type | File(s) | Current Size | Summary |
|---|---|---|---|---|
| **80** | arch | `tools.py` | 778 lines | Mixes 10 tool implementations, utility classes, dispatcher |
| **81** | arch | `infra/prompt_assembler.py` | 481 lines | Mixes prompt assembly, file I/O (`append_memory`, `add_skill`), object registry |
| **82** | arch | `main.py` | 826 lines | 4 post-construction injection blocks (~17 `obj._attr = ...` assignments) |
| **79** | arch | `core/llm_thread.py` | 1199 lines | God object: queue, history, compaction, prompt assembly, inference, TP, sessions |
| **85** | arch | `core/llm_thread.py` | — | Single asyncio queue serializes all threads |

## Implementation Plan

### Phase 2A — `_process()` decomposition ✅

**Issue #104** — Completed. Extracted `_run_inference_with_retry()` from `_process()`.

### Phase 2B — DI mechanism ✅

**Issue #83** — Completed. Created `ToolDeps` dataclass in `core/tool_deps.py`. Removed all `register_*`/`set_*` globals from `tools.py` and `prompt_assembler.py`. Dependencies now flow via `ToolDeps` through `ToolCallContext` and explicit parameters.

### Phase 3 — God object decomposition (sequential, after Phase 2B)

These three issues all depend on #83 being resolved first, because clean separation requires the DI globals to be gone. They can be done **in parallel** with each other.

#### Issue #80 — Split `tools.py` (778 lines)

Schemas are already in `core/tool_schemas.py`. Remaining split:
- `tools.py` → keep as thin dispatcher (`execute_tool()` + `_DISPATCH` dict)
- `tools/task_tools.py` — `_tool_task` + all `_task_*` handlers
- `tools/memory_tools.py` — `_tool_append_memory`, `_tool_add_skill`
- `tools/io_tools.py` — `_tool_read_file`, `_tool_write_file`, `_tool_execute_shell`
- `tools/web_tools.py` — `_tool_search_web`, `_tool_fetch_url`, `_HTMLTextExtractor`
- `tools/session_tools.py` — `_tool_spawn_sub_session`, `_tool_query_telemetry`

DI globals move into `ToolCallContext` (from #83), so tool functions receive deps as args.

#### Issue #81 — Split `prompt_assembler.py` (481 lines)

- `infra/prompt_assembler.py` → keep prompt assembly logic only (`assemble()`, `_parse_sections`, `_assemble_base`)
- `infra/memory_io.py` → `append_memory()`, `merge_consolidated_memories()`, `update_memories()`
- `infra/skill_io.py` → `add_skill()`
- Remove `set_tool_profiles()` / `set_self_model()` — pass as args (from #83)

#### Issue #82 — Fix two-phase construction in `main.py`

After #83 resolves the DI pattern, interfaces can receive all dependencies via constructor:
1. Defer construction of `MatrixThread` and `WebInterface` until all deps are available
2. Pass deps as constructor args (or a single config/deps object)
3. Remove all `obj._private_attr = ...` post-construction blocks from `main.py`

**All three depend on:** #83
**Can be parallelized:** Yes, #80 / #81 / #82 touch different files

### Phase 4 — LLMThread decomposition

**Issue #79** — Break up the LLMThread god object (1199 lines)

This is the capstone. After #104 splits `_process()` and #83/#80 clean up DI and tools, LLMThread's remaining responsibilities can be separated:

- **Queue + run loop** → stays in `LLMThread` (its core purpose)
- **Conversation history + DB** → extract `ConversationStore` or similar
- **Context compaction** → extract `ContextCompactor`
- **Inference orchestration** → already partially separated via `inference_engine.py`; complete the extraction
- **Session management** → extract `SessionManager` (timeouts, resets, per-thread config)

**Depends on:** #104, #83, #80

### Phase 5 — Per-thread queues

**Issue #85** — Replace single queue with per-thread queues

This is the final issue and requires LLMThread to be sufficiently decomposed (#79) so that adding per-thread queue routing doesn't re-inflate a god object.

Approach: one `asyncio.Queue` per `thread_id`, with a dispatcher that routes incoming messages. Enables true per-thread concurrency.

**Depends on:** #79

## Dependency Graph

```
Phase 2A              Phase 2B           Phase 3              Phase 4       Phase 5
──────────           ──────────         ──────────           ──────────    ──────────
                                        #80 (tools split)
#104 (_process) ─────────────────────────────────────────┐
                      #83 (DI) ───────→ #81 (prompt split) → #79 (LLMThread) → #85 (queues)
                                        #82 (constructors)
```

## Summary: Execution Order

| Order | Issue(s) | Phase | Parallel? | Status |
|---|---|---|---|---|
| 1 | **#104**, **#83** | 2A + 2B | Yes, parallel | ✅ Done |
| 2 | **#80**, **#81**, **#82** | 3 | Yes, parallel (all need #83 done) | Next |
| 3 | **#79** | 4 | No (needs #104 + #80 + #83) | |
| 4 | **#85** | 5 | No (needs #79) | |
