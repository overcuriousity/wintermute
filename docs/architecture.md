# Architecture

## Component Overview

Wintermute runs as a single Python asyncio process with several concurrent tasks:

| Component | Module | Role |
|-----------|--------|------|
| **LLMThread** | `llm_thread.py` | Owns conversation history, runs inference + tool-use loops |
| **WebInterface** | `web_interface.py` | aiohttp HTTP server; debug/admin panel at `/debug`, REST API at `/api/debug/*` |
| **MatrixThread** | `matrix_thread.py` | Matrix client with E2E encryption (optional) |
| **SubSessionManager** | `sub_session.py` | Manages background worker sub-sessions and workflow DAGs |
| **Turing Protocol** | `turing_protocol.py` | Three-stage post-inference validation framework (detect, validate, correct) |
| **ReminderScheduler** | `scheduler_thread.py` | APScheduler-based reminder system |
| **AgendaLoop** | `agenda.py` | Periodic autonomous agenda item reviews |
| **DreamingLoop** | `dreaming.py` | Nightly memory consolidation |
| **GeminiCloudClient** | `gemini_client.py` | AsyncOpenAI-compatible wrapper for Google Cloud Code Assist API (duck-typed drop-in replacement) |
| **NL Translator** | `nl_translator.py` | Expands natural-language tool descriptions into structured arguments via a translator LLM |
| **PromptAssembler** | `prompt_assembler.py` | Builds system prompts from file components |
| **Database** | `database.py` | SQLite message persistence, thread management, and agenda storage |

## System Diagram

```
User (Matrix / Browser)
        |
        v
  LLMThread  <--- system prompt (BASE + MEMORIES + PULSE + SKILLS)
  (asyncio)        assembled fresh each turn
        |
        |-- tool calls --> execute_shell / read_file / write_file
        |                  search_web / fetch_url
        |                  append_memory / agenda / add_skill
        |                  set_reminder / list_reminders
        |
        +-- spawn_sub_session --> SubSessionManager
                                        |
                                        |-- Workflow DAG
                                        |   |-- worker A (no deps) --> starts immediately
                                        |   |-- worker B (no deps) --> starts immediately
                                        |   +-- worker C (depends_on=[A,B]) --> auto-starts
                                        |       when A and B complete; receives their results
                                        |
                                        |-- Nested spawning (full-mode workers)
                                        |   |-- worker X (full mode, depth 1) --> spawns Y, Z
                                        |   |   |-- worker Y (depth 2) --\
                                        |   |   +-- worker Z (depth 2) --+--> aggregated result
                                        |   |                                 delivered to root
                                        |   |                                 thread when all done
                                        |
                                        +-- result --> enqueue_system_event
                                                        (back to LLMThread)

AgendaLoop --------------------------------> per-thread sub-session (full mode, result → originating room)
ReminderScheduler ------------------------> LLMThread queue / sub-session
DreamingLoop (nightly) ------------------> direct LLM API call (no tool loop)
```

## Startup Flow

1. Load `config.yaml`
2. Configure logging (console + rotating file)
3. Initialise SQLite databases
4. Bootstrap `data/` directories (skills/, scripts/, archive/)
5. Restore APScheduler jobs (and execute missed reminders)
6. Build shared broadcast function (routes to Matrix rooms or web clients)
7. Start LLM inference task
8. Start web interface task (if enabled)
9. Start Matrix task (if configured)
10. Start agenda review loop
11. Start dreaming loop
12. Await shutdown signals (SIGTERM / SIGINT)

## Data Flow: User Message

1. User sends a message via Matrix or WebSocket
2. Message enters the LLMThread queue
3. LLMThread builds the message list from the SQLite DB
4. System prompt is assembled fresh (BASE + MEMORIES + PULSE + SKILLS + compaction summary)
5. If history tokens exceed the compaction threshold, context is compacted first
6. Message is saved to the DB, then inference runs
7. If the model returns tool calls:
   - If NL translation is enabled and the call uses a simplified schema, the translator LLM expands the description into structured arguments
   - Tools are executed and inference continues
8. Final response is saved to the DB and broadcast back to the user
9. Turing Protocol fires asynchronously (if enabled): runs a three-stage
   pipeline (detect → validate → correct) against the response, scoped by
   phase (`post_inference`, `pre_execution`, `post_execution`) and context
   (`main` thread or `sub_session`). If violations are confirmed (e.g. the
   model claimed to spawn a session but `spawn_sub_session` is not in
   `tool_calls_made`), a single corrective system event is injected so the
   model can self-correct. Each hook fires at most once per turn — there is
   no escalation or re-checking of the correction response

## Sub-session Lifecycle

1. Orchestrator (main LLM or a `full`-mode worker) calls `spawn_sub_session`
2. `SubSessionManager.spawn()` registers a `TaskNode` in a workflow DAG
3. Dependency IDs are validated — unknown IDs are stripped with a warning to prevent deadlocks
4. If `depends_on` is specified and dependencies aren't done yet, the node stays `pending`
5. Otherwise the worker starts immediately with its own in-memory message list
6. Worker runs `_worker_loop()`: inference + tool-call loop with filtered tools
7. On completion, `_resolve_dependents()` checks if pending nodes can now start
8. Result delivery depends on nesting:
   - **Direct children** (depth 1): result enters the parent thread via `enqueue_system_event`
   - **Nested children** (depth 2): individual reports are suppressed; when all siblings finish, an aggregated result is delivered to the root (user-facing) thread
9. If the worker times out, a continuation is auto-spawned (up to 3 hops)

## Workflow DAG

- Every sub-session is a node in a `Workflow` (auto-created if needed)
- `depends_on`: list of session_ids that must complete first
- `not_before`: optional time gate — task waits for this datetime even if deps are done
- Fan-in: task C with `depends_on=[A, B]` auto-starts when both finish (and time gate is met)
- Failure propagation: if a dependency fails, all transitive dependents are marked failed
- Resolution is event-driven (no polling): each completion triggers a check
- Workflows spanning multiple prior workflows are automatically merged
- `depends_on_previous`: workers can automatically depend on all sessions they've previously spawned, eliminating the need to track session IDs manually
- Unknown dependency IDs are stripped at spawn time to prevent deadlocks from hallucinated IDs

## Design for Small LLMs

Wintermute is explicitly designed to work with small, quantised models (3B–8B parameters) as the primary backend. Several architectural choices serve this goal:

**No framework abstraction layer.** Tool calls use the OpenAI function-calling wire format directly. There is no LangChain, LlamaIndex, or other intermediary that rewrites prompts, adds hidden tokens, or applies transformations the model can't see. What the model receives is exactly what you configure. This makes behaviour predictable and debuggable even with weak models.

**Turing Protocol.** A three-stage (detect → validate → correct) post-inference validation pipeline that catches the hallucination patterns small models are most prone to — claiming to have done things they didn't, fabricating tool output, or making promises without acting. Rather than requiring a stronger model, corrections are injected automatically so the model can self-correct. See [turing-protocol.md](turing-protocol.md) for the full reference.

**NL Translation (optional).** For models that struggle with multi-field structured JSON schemas, complex tool calls (`set_reminder`, `spawn_sub_session`) can be exposed as a single plain-English `description` field. A dedicated small translator LLM expands the description into structured arguments. See [tools.md — NL Translation Mode](tools.md#nl-translation-mode).

**Lean system prompt.** The system prompt is assembled from independent file-based components (`BASE_PROMPT.txt`, `MEMORIES.txt`, agenda, skills). Components have configurable character caps with auto-summarisation when exceeded. No framework boilerplate is injected — the prompt contains only what you wrote and what the model genuinely needs.

**Context compaction.** When conversation history approaches the context window, older messages are summarised via a chained rolling summary rather than truncated. This keeps the model oriented without requiring a large context window.

**Role-segregated backends.** Heavy tasks (compaction, dreaming, Turing Protocol detection) can be routed to different, purpose-sized backends. A small 3B model can serve as the Turing Protocol validator while a 7B model handles the main conversation.

## Memory Structure

```
data/
  BASE_PROMPT.txt            -- Immutable core instructions
  MEMORIES.txt               -- Long-term user facts (updated via append_memory)
  conversation.db (agenda)    -- Active goals / working memory (managed via agenda tool, stored in SQLite)
  skills/                    -- Learned procedures as *.md files (updated via add_skill tool)
  DREAM_MEMORIES_PROMPT.txt  -- Customisable dreaming prompt for MEMORIES consolidation
  DREAM_PULSE_PROMPT.txt     -- Customisable dreaming prompt for agenda consolidation
  COMPACTION_PROMPT.txt      -- Customisable prompt for context compaction summarisation
  matrix_crypto.db           -- Matrix E2E encryption keys
  matrix_recovery.key        -- Cross-signing recovery key
```
