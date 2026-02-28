# Refactor Strategy: Issues #79–#108

**Status as of 2026-03-01** — Phase 1 complete. This document covers the remaining 7 open issues and their implementation order.

## Completed (Phase 1)

All six Phase 1 issues have been merged:

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

## Remaining Open Issues (7)

| # | Type | File(s) | Current Size | Summary |
|---|---|---|---|---|
| **104** | refactor | `core/llm_thread.py` | 1199 lines | `_process()` does 9+ things in ~180 lines |
| **83** | arch | `tools.py`, `infra/prompt_assembler.py`, `infra/memory_store.py` | — | 6+ module-level `None` globals used as DI, with `register_*`/`set_*` wiring |
| **80** | arch | `tools.py` | 778 lines | Mixes DI globals, 10 tool implementations, utility classes, dispatcher |
| **81** | arch | `infra/prompt_assembler.py` | 481 lines | Mixes prompt assembly, file I/O (`append_memory`, `add_skill`), object registry |
| **82** | arch | `main.py` | 826 lines | 4 post-construction injection blocks (~17 `obj._attr = ...` assignments) |
| **79** | arch | `core/llm_thread.py` | 1199 lines | God object: queue, history, compaction, prompt assembly, inference, TP, sessions |
| **85** | arch | `core/llm_thread.py` | — | Single asyncio queue serializes all threads |

## Implementation Plan

### Phase 2A — `_process()` decomposition (independent)

**Issue #104** — Split `_process()` into focused helpers

`_process()` (lines 629–809, 181 lines) performs 11 distinct operations. This is self-contained and benefits from the Phase 1 work (#99 factory, #100 TP runner).

Extract into named helpers:
- `_resolve_thread_context()` — config/pool resolution + activity tracking
- `_fetch_memory_context()` — vector query construction + memory search
- `_assemble_and_compact()` — prompt assembly + token budget + compaction
- `_persist_user_message()` — DB save with per-message-type logic
- `_run_inference_with_retry()` — inference loop + ContextTooLargeError retry
- `_persist_result()` — interaction log + assistant message save + events

**Depends on:** nothing (Phase 1 is done)
**Enables:** #79 (LLMThread decomposition is easier with smaller methods)

### Phase 2B — DI mechanism (independent, parallel with 2A)

**Issue #83** — Replace module-level globals with explicit dependency injection

Current state (6 globals):
- `tools.py`: `_task_scheduler_ensure`, `_task_scheduler_remove`, `_task_scheduler_list`, `_sub_session_spawn`, `_event_bus`, `_self_model_profiler`
- `prompt_assembler.py`: `_tool_profiles`, `_self_model_profiler` (duplicated!)
- `memory_store.py`: `_backend`, `_config` (singleton via `init()`)

Approach:
1. Create a `ToolDeps` dataclass (or similar) holding all tool-execution dependencies
2. Pass it through `ToolCallContext` (which already flows through the call chain)
3. Remove `register_*`/`set_*` functions and module-level globals from `tools.py`
4. For `prompt_assembler.py`: pass deps as arguments to `assemble()` instead of module globals
5. `memory_store.py` singleton is acceptable (it's a true singleton, initialized once)

**Depends on:** nothing
**Enables:** #80, #81, #82

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

| Order | Issue(s) | Phase | Parallel? |
|---|---|---|---|
| 1 | **#104**, **#83** | 2A + 2B | Yes, parallel |
| 2 | **#80**, **#81**, **#82** | 3 | Yes, parallel (all need #83 done) |
| 3 | **#79** | 4 | No (needs #104 + #80 + #83) |
| 4 | **#85** | 5 | No (needs #79) |
