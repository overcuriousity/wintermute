# Autonomous Features

Wintermute includes several autonomous background systems that operate without user interaction.

## Dreaming Loop

**Module:** `wintermute/dreaming.py`

A nightly consolidation pass that reviews and prunes MEMORIES.txt and pulse DB items.

- Fires at a configurable hour (default: 01:00 UTC)
- Uses a direct LLM API call — no tool loop, no conversation side effects
- Each component is consolidated independently (a failure in one doesn't abort the other)
- Can use a dedicated backend (configured via `llm.dreaming` in `config.yaml`)
- Manually triggerable via the `/dream` command

**Consolidation logic:**
- MEMORIES.txt: removes duplicates, merges related facts, preserves distinct useful facts
- Pulse items: LLM returns JSON actions (complete, update, keep) applied via DB; completed items older than 30 days are purged

The prompts used for consolidation are stored in `data/DREAM_MEMORIES_PROMPT.txt` and `data/DREAM_PULSE_PROMPT.txt` and can be customised. See [system-prompts.md](system-prompts.md#customisable-prompt-templates).

## Pulse Reviews

**Module:** `wintermute/pulse.py`

Periodic autonomous reviews of active pulse items.

- Runs at a configurable interval (default: every 60 minutes)
- Spawns one sub-session per thread that has active pulse items bound to it (via `thread_id`)
- Each sub-session runs in `full` mode with `parent_thread_id` set to the originating thread, so results are delivered back to that room
- The sub-session lists pulse items via the `pulse` tool and takes appropriate actions (complete items, add new ones, set reminders, update memories, run commands, etc.)
- If nothing needs attention the worker responds with `[NO_ACTION]` and the result is suppressed — no message is sent
- Pulse items without a `thread_id` (legacy/unbound items) are skipped by the review loop
- New pulse items automatically inherit the `thread_id` of the thread that created them

## Sub-sessions and Workflow DAG

**Module:** `wintermute/sub_session.py`

Background workers for complex, multi-step tasks.

### Sub-session Modes

| Mode | System Prompt | Tools | Use Case |
|------|--------------|-------|----------|
| `minimal` | Lightweight execution instructions | execution + research | Default, fast and cheap |
| `full` | Full prompt (BASE + MEMORIES + PULSE + SKILLS) | all including orchestration | When worker needs full context or must spawn further workers |
| `base_only` | BASE_PROMPT.txt only | execution + research | Core instructions without memory overhead |
| `none` | Empty | execution + research | Purely mechanical tasks |

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

## Reminder-triggered Inference

**Module:** `wintermute/scheduler_thread.py`

Reminders can optionally trigger AI inference when they fire:

- If `ai_prompt` is set on a reminder, an isolated AI inference runs with that prompt
- Thread-bound reminders deliver the response to the originating chat thread
- System reminders (no thread) fire as system events
- Supports one-time, daily, weekly, monthly, and interval schedules
- Interval reminders can be restricted to time windows (e.g. 08:00-20:00)

## Context Compaction

**Module:** `wintermute/llm_thread.py`

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
| Pulse (DB) | 5,000 chars |
| skills/ (total) | 20,000 chars |

When exceeded, a system event is enqueued asking the AI to read, condense, and update the component.
