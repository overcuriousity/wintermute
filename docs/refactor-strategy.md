# Refactor Strategy: Issues #79–#108

**Status as of 2026-03-06** — Phase 3 complete (#80, #81, #82 all done). This document covers the remaining 2 open issues and their implementation order.

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
| #80 | `tools.py` split into `tools/` package | Merged |
| #81 | `prompt_assembler.py` split into `memory_io` + `skill_io` | [#156](https://github.com/overcuriousity/wintermute/pull/156) |
| #82 | Two-phase construction eliminated in `main.py` | Merged |

## Remaining Open Issues (2)

| # | Type | File(s) | Current Size | Summary |
|---|---|---|---|---|
| **79** | arch | `core/llm_thread.py` | 1324 lines | God object: queue, history, compaction, prompt assembly, inference, TP, sessions |
| **85** | arch | `core/llm_thread.py` | — | Single asyncio queue serializes all threads |

## Implementation Plan

### Phase 2A — `_process()` decomposition ✅

**Issue #104** — Completed. Extracted `_run_inference_with_retry()` from `_process()`.

### Phase 2B — DI mechanism ✅

**Issue #83** — Completed. Created `ToolDeps` dataclass in `core/tool_deps.py`. Removed all `register_*`/`set_*` globals from `tools.py` and `prompt_assembler.py`. Dependencies now flow via `ToolDeps` through `ToolCallContext` and explicit parameters.

### Phase 3 — God object decomposition (sequential, after Phase 2B)

These three issues all depend on #83 being resolved first, because clean separation requires the DI globals to be gone. They can be done **in parallel** with each other.

#### Issue #80 — Split `tools.py` (778 lines) ✅

Completed. `tools.py` converted to `tools/` package:
- `tools/__init__.py` — thin dispatcher (`execute_tool()` + `_DISPATCH` dict) + re-exports
- `tools/task_tools.py` — `tool_task` + all task action handlers + `_describe_schedule`
- `tools/memory_tools.py` — `tool_append_memory`, `tool_add_skill`
- `tools/io_tools.py` — `tool_read_file`, `tool_write_file`, `tool_execute_shell`
- `tools/web_tools.py` — `tool_search_web`, `tool_fetch_url`, `_HTMLTextExtractor`
- `tools/session_tools.py` — `tool_worker_delegation`, `tool_query_telemetry`

All external consumers (`from wintermute import tools as tool_module`) remain unchanged.

#### Issue #81 — Split `prompt_assembler.py` (481 lines) ✅

Completed. `prompt_assembler.py` reduced to ~350 lines (prompt assembly only):
- `infra/memory_io.py` — `append_memory()`, `update_memories()`, `merge_consolidated_memories()`, `write_memories_raw()`, `read_text_safe()`
- `infra/skill_io.py` — `add_skill()`
- Added `get_timezone()` public accessor (replaces direct `_timezone` access)
- `_memories_lock` encapsulated inside `memory_io` (no longer exposed)
- Consumers rewired to import from `memory_io`, `skill_io`, and `paths` directly

#### Issue #82 — Fix two-phase construction in `main.py` ✅

Completed. Eliminated all post-construction `obj._attr = ...` injection blocks:
- **Lazy getter** breaks the LLMThread ↔ SubSessionManager circular dependency (replaces `inject_sub_session_manager()`)
- **ToolDeps** refactored to hold typed object references (`sub_session_manager`, `task_scheduler`, `event_bus`, `self_model_profiler`) instead of 7+ individual callables
- **MatrixThread** and **WebInterface** constructors now accept all dependencies — deferred construction until all deps exist
- **ReflectionLoop** accepts `self_model` via constructor (removed `inject_self_model()`)
- **Dead code cleanup**: 8 unused MatrixThread injections and 2 unused WebInterface injections removed (leftovers from #86 slash-command extraction)
- `main.py` construction flow reordered with numbered steps and clear comments
- Only 3 typed-object assignments remain on ToolDeps (down from 8+ callable assignments) — these are inherent to the construction order, not arbitrary injection

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
| 2 | **#80**, **#81**, **#82** | 3 | Yes, parallel (all need #83 done) | ✅ All done |
| 3 | **#79** | 4 | No (needs #104 + #80 + #83) | |
| 4 | **#85** | 5 | No (needs #79) | |
