# Autonomous Features

Wintermute includes several autonomous background systems that operate without user interaction.

## Dreaming Loop

**Module:** `wintermute/dreaming.py`

A nightly consolidation pass that reviews and prunes MEMORIES.txt and PULSE.txt.

- Fires at a configurable hour (default: 01:00 UTC)
- Uses a direct LLM API call — no tool loop, no conversation side effects
- Each component is consolidated independently (a failure in one doesn't abort the other)
- Can use a dedicated model (falls back to `compaction_model`, then main `model`)
- Manually triggerable via the `/dream` command

**Consolidation logic:**
- MEMORIES.txt: removes duplicates, merges related facts, preserves distinct useful facts
- PULSE.txt: removes completed/stale items, merges overlapping goals, keeps active tasks

The prompts used for consolidation are stored in `data/DREAM_MEMORIES_PROMPT.txt` and `data/DREAM_PULSE_PROMPT.txt` and can be customised. See [system-prompts.md](system-prompts.md#customisable-prompt-templates).

## Pulse Reviews

**Module:** `wintermute/pulse.py`

Periodic autonomous reviews of PULSE.txt.

- Runs at a configurable interval (default: every 60 minutes)
- Spawns an isolated sub-session in `full` mode (fire-and-forget, no parent thread)
- The sub-session reads PULSE.txt and takes appropriate actions (set reminders, update memories, run commands, etc.)
- Never pollutes any user-facing conversation history

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
- Summary is injected into the system prompt as a "Conversation Summary" section
- Can be manually triggered via the `/compact` command

## Component Auto-summarisation

After every inference, Wintermute checks if any memory component exceeds its size limit:

| Component | Default Limit |
|-----------|---------------|
| MEMORIES.txt | 10,000 chars |
| PULSE.txt | 5,000 chars |
| skills/ (total) | 20,000 chars |

When exceeded, a system event is enqueued asking the AI to read, condense, and update the component.
