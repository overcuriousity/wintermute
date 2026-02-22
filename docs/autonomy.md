# Autonomous Features

## Conversation Seed

**Module:** `wintermute/core/llm_thread.py`

When a new conversation starts (first user message in an empty thread, `/new` in Matrix, or session reset in the web UI), a seed system event is automatically injected before the user's message is processed. The seed prompts the LLM to introduce itself, mention relevant active goals from memories and agendas, and briefly explain its capabilities.

- Language-specific: configured via `seed.language` in `config.yaml` (default: `"en"`)
- Prompt files: `data/prompts/SEED_{language}.txt` (shipped: `en`, `de`, `fr`, `es`, `it`, `zh`, `ja`)
- Falls back to English if the configured language file is missing
- Add new languages by creating `data/prompts/SEED_{code}.txt`
- The seed reply is broadcast to the user before their first message is processed

Wintermute includes several autonomous background systems that operate without user interaction.

## Dreaming Loop

**Module:** `wintermute/workers/dreaming.py`

A nightly consolidation pass that reviews and prunes MEMORIES.txt and agenda DB items.

- Fires at a configurable hour (default: 01:00 UTC)
- Uses a direct LLM API call — no tool loop, no conversation side effects
- Each component is consolidated independently (a failure in one doesn't abort the other)
- Can use a dedicated backend (configured via `llm.dreaming` in `config.yaml`)
- Manually triggerable via the `/dream` command

**Consolidation logic:**
- MEMORIES.txt: removes duplicates, merges related facts, preserves distinct useful facts
- Agenda items: LLM returns JSON actions (complete, update, keep) applied via DB; completed items older than 30 days are purged
- Skills: deduplicates overlapping skills, then condenses each to ~150 tokens while preserving the first-line summary (used in the skills TOC)

The prompts used for consolidation are stored in `data/DREAM_MEMORIES_PROMPT.txt` and `data/DREAM_AGENDA_PROMPT.txt` and can be customised. See [system-prompts.md](system-prompts.md#customisable-prompt-templates).

## Memory Harvest

**Module:** `wintermute/workers/memory_harvest.py`

Periodic background extraction of personal facts and preferences from conversation history into MEMORIES.txt.

**Problem:** The `append_memory` tool exists but weak/small models rarely call it proactively. Conversations contain valuable personal details that are lost when history is compacted or archived.

**Solution:** A polling loop spawns sub-session workers that mine recent conversation history and call `append_memory` for each new fact discovered. Workers first read MEMORIES.txt to avoid duplicating existing entries.

**Trigger conditions (per thread, OR logic):**
- **Message threshold:** N or more new user messages since the last harvest (default: 20)
- **Inactivity timeout:** M minutes without activity AND at least 5 new user messages (default: 15 minutes)

**Scope:**
- Harvests from user-facing threads only (Matrix rooms, web sessions, "default")
- Sub-session threads (`sub_*`) are excluded
- Memories are written to the unified MEMORIES.txt (shared across all threads)
- Results are fire-and-forget (no messages delivered to chat)
- Visibility via logs and the debug panel (`/debug` → interaction_log, action=`memory_harvest`)

**Message filtering:**
- Only user + assistant messages are included in the transcript
- System events, tool call results, and sub-session notifications are excluded
- Individual messages exceeding `max_message_chars` are mid-truncated (beginning and end preserved, middle cut)
- Total transcript capped at ~60k characters (~15k tokens)

**Relationship to other memory systems:**
- Complements (does not replace) the `append_memory` tool — explicit user requests ("remember that I prefer X") still use the tool directly
- Nightly dreaming consolidation deduplicates any near-duplicate entries the harvest may create
- The harvest worker uses the `sub_sessions` backend pool — no separate LLM role needed

**Configuration:** See `memory_harvest:` section in `config.yaml`.

## Agenda Reviews

**Module:** `wintermute/workers/agenda.py`

Periodic autonomous reviews of active agenda items.

- Runs at a configurable interval (default: every 60 minutes)
- Spawns one sub-session per thread that has active agenda items bound to it (via `thread_id`)
- Each sub-session runs in `full` mode with `parent_thread_id` set to the originating thread, so results are delivered back to that room
- The sub-session lists agenda items via the `agenda` tool and takes appropriate actions (complete items, add new ones, set routines, update memories, run commands, etc.)
- If nothing needs attention the worker responds with `[NO_ACTION]` and the result is suppressed — no message is sent
- Agenda items without a `thread_id` (legacy/unbound items) are skipped by the review loop
- New agenda items automatically inherit the `thread_id` of the thread that created them

## Sub-sessions and Workflow DAG

**Module:** `wintermute/core/sub_session.py`

Background workers for complex, multi-step tasks.

### Sub-session Modes

| Mode | System Prompt | Tools | Use Case |
|------|--------------|-------|----------|
| `minimal` | Lightweight execution instructions | execution + research | Default, fast and cheap |
| `full` | Sectioned BASE + MEMORIES + AGENDA + SKILLS TOC | all including orchestration | When worker needs full context or must spawn further workers |
| `base_only` | Sectioned BASE_PROMPT.txt only | execution + research | Core instructions without memory overhead |
| `none` | Empty | execution + research | Purely mechanical tasks |

Both `full` and `base_only` modes use conditional BASE_PROMPT sections — only sections relevant to the worker's available tools are included. This saves ~800 tokens for minimal workers.

### Tool Profiles

As an alternative to manually selecting `system_prompt_mode` and `tool_names`, named tool profiles provide config-driven presets for common worker patterns:

```
spawn_sub_session(objective="Search for X", profile="researcher")
```

See [tools.md — Tool Profiles](tools.md#tool-profiles) for available profiles and how to define custom ones.

### Workflow DAG

Multi-step tasks are expressed as dependency graphs:

1. The orchestrator spawns tasks with `depends_on` to define the DAG
2. Tasks with no dependencies start immediately
3. When a task completes, its dependents are checked
4. If all dependencies are done, the dependent auto-starts with their results as context
5. If any dependency fails, all transitive dependents are marked failed

Example: research A + research B -> upload C (depends_on=[A, B])

#### `depends_on_previous`

Instead of manually tracking and listing session IDs (which is error-prone and can lead to hallucinated IDs), workers can set `depends_on_previous: true` to automatically depend on all sessions they have previously spawned. The system resolves the IDs programmatically.

#### Dependency ID Validation

Unknown session IDs in `depends_on` are automatically stripped with a warning log. This prevents permanent deadlocks caused by hallucinated or mistyped session IDs — the task proceeds with only the valid dependencies.

### Time-gated Workflows

Tasks can include a `not_before` parameter (ISO-8601 datetime) to delay execution until a specific time, even if all dependencies are already satisfied. This enables workflows like "research now, upload after 20:00":

```
spawn_sub_session(objective="Research topic")           → sub_aaa
spawn_sub_session(objective="Upload report",
                  depends_on=["sub_aaa"],
                  not_before="2025-01-15T20:00:00")    → sub_bbb
```

The time gate is checked after dependency resolution. When the gate isn't met yet, an asyncio callback is scheduled to re-check at the specified time — no polling required. Naive datetimes (without timezone) are interpreted in the configured `scheduler.timezone`.

### Continuation on Timeout

When a worker times out (default: 300 seconds):

1. Its full message history is preserved (updated after every tool call)
2. A continuation sub-session is auto-spawned with the prior messages
3. The new worker picks up exactly where the old one left off
4. Up to 3 continuation hops are allowed before giving up

### Nesting

- Maximum nesting depth: 2 (main agent -> sub-session -> sub-sub-session)
- Only `full`-mode workers have the `spawn_sub_session` tool available
- Other modes have no orchestration tools at all

### Nested Result Aggregation

When a `full`-mode sub-session (depth 1) spawns children (depth 2), individual child results are not delivered separately. Instead, the system:

1. Tracks the **root thread ID** (the original user-facing chat thread) through all nesting levels
2. Suppresses per-child reports (which would be delivered to the parent sub-session's ID — not a real chat thread)
3. Waits until **all children** of a parent sub-session reach a terminal state (completed, failed, or timeout)
4. Delivers one **aggregated message** containing all child results to the root thread

This is fully deterministic — no LLM inference is involved in the routing or aggregation logic. A deduplication set prevents duplicate delivery when multiple children complete near-simultaneously.

## Routine-triggered Inference

**Module:** `wintermute/workers/scheduler_thread.py`

Routines can optionally trigger AI inference when they fire:

- If `ai_prompt` is set on a routine, an isolated AI inference runs with that prompt
- Thread-bound routines deliver the response to the originating chat thread
- System routines (no thread) fire as system events
- Supports one-time, daily, weekly, monthly, and interval schedules
- Interval routines can be restricted to time windows (e.g. 08:00-20:00)

## Context Compaction

**Module:** `wintermute/core/llm_thread.py`

Automatic conversation history management:

- Triggers when history tokens exceed: `context_size - max_tokens - system_prompt_tokens`
- Summarises older messages using a dedicated compaction model (or the main model)
- The summarisation prompt is stored in `data/COMPACTION_PROMPT.txt` and can be customised (see [system-prompts.md](system-prompts.md#customisable-prompt-templates))
- Keeps the last 10 messages intact
- **Chained summaries**: each compaction includes the prior summary in its input, so context is preserved across compaction cycles rather than lost. The compaction prompt merges `[PRIOR SUMMARY]` and `[NEW MESSAGES]` into a single cohesive summary.
- Only one summary is kept per thread (old summaries are replaced, not accumulated)
- Summary is injected into the system prompt as a "Conversation Summary" section
- Archived messages older than 30 days are automatically purged from the database
- Can be manually triggered via the `/compact` command

## Component Auto-summarisation

After every inference, Wintermute checks if any memory component exceeds its size limit:

| Component | Default Limit |
|-----------|---------------|
| MEMORIES.txt | 10,000 chars |
| Agenda (DB) | 5,000 chars |
| skills/ (TOC) | 2,000 chars |

When exceeded, a system event is enqueued asking the AI to read, condense, and update the component.
