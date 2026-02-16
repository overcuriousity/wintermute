# System Prompts

## Overview

Wintermute assembles its system prompt fresh on every inference call from individual file components stored in `data/`.

## Components

### 1. BASE_PROMPT.txt — Core Instructions

The foundation of every prompt. Shipped with the repository and editable by the user.

Content covers:

- Role definition (personal AI assistant via Matrix chat)
- Memory system: clear boundary between MEMORIES (long-term facts) and PULSE (active working memory)
- Background process awareness (pulse reviews, nightly dreaming, context compaction)
- Tool usage patterns and categories
- Sub-session delegation and workflow DAGs with `depends_on`
- System event handling (sub-session results, reminders, errors, timeouts)
- Behavioural guidelines (concise responses, credential policy, confidence)

### 2. MEMORIES.txt — Long-Term Facts

Stores persistent facts about the user — preferences, biographical details, established decisions. Updated day-to-day via `append_memory` (preferred) or restructured via `update_memories` (full rewrite). Consolidated nightly by the dreaming loop to merge duplicates and prune outdated entries.

Key rule: if information would still matter in a month with no active project around it, it belongs in MEMORIES.

### 3. Pulse (SQLite) — Working Memory

Active goals, ongoing projects, time-sensitive tasks, and open questions. Managed via the `pulse` tool (add, complete, update, list actions). Each item has a priority (1=urgent, 10=low) and an auto-assigned ID. Reviewed autonomously every 60 minutes by the pulse loop and consolidated nightly by the dreaming loop.

Key rule: if it only matters because something is in progress right now, it belongs in Pulse.

### 4. skills/*.md — Learned Procedures

Each file in `data/skills/` is a Markdown document describing a reusable procedure the AI has learned. Created via the `add_skill` tool. All skills are loaded and injected into every prompt.

## Prompt Assembly Pipeline

The `prompt_assembler.assemble()` function builds the final system prompt:

```
# Core Instructions
{BASE_PROMPT.txt content}

---

# Current Time
{e.g. "Wednesday, 2025-01-15 14:32 CET"}

---

# User Memories
{MEMORIES.txt content}

---

# Active Pulse
{formatted pulse items from DB, e.g. "[P2] #3: Fix auth bug"}

---

# Conversation Summary
{compaction summary, if context was compacted}

---

# Skills

### Skill: {skill_name}
{skill content}

---

### Skill: {another_skill}
{another skill content}
```

Sections are separated by `---` dividers. Empty sections are omitted entirely.

## Sub-session System Prompt Modes

When spawning sub-sessions, the system prompt varies by mode:

| Mode | System Prompt Content |
|------|----------------------|
| `minimal` | Lightweight execution agent instructions (default) |
| `full` | Full assembled prompt (BASE + MEMORIES + PULSE + SKILLS) |
| `base_only` | BASE_PROMPT.txt only |
| `none` | No system prompt (bare tool-use loop) |

All modes append the task context and objective at the end.

## Size Limits and Auto-summarisation

Each component has a configurable character limit (set in `config.yaml` under `context.component_size_limits`):

| Component | Default Limit | Action When Exceeded |
|-----------|---------------|---------------------|
| MEMORIES.txt | 10,000 chars | AI is asked to condense and prioritise |
| Pulse (DB) | 5,000 chars | AI is asked to condense and prioritise |
| skills/ (total) | 20,000 chars | AI is asked to reorganise |

When a component exceeds its limit after any inference, a system event is enqueued asking the AI to read the component, condense it, and update it using the appropriate tool.

The nightly dreaming loop also consolidates MEMORIES.txt and pulse items independently using a direct LLM call (no tool loop).

## Customisable Prompt Templates

The following prompt templates are stored as editable files in `data/` and shipped with the repository. They can be freely modified.

| File | Used By | Placeholder | Purpose |
|------|---------|-------------|---------|
| `DREAM_MEMORIES_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating MEMORIES.txt overnight |
| `DREAM_PULSE_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating pulse items overnight (LLM returns JSON actions) |
| `COMPACTION_PROMPT.txt` | Context compaction | `{history}` | Instructions for summarising old conversation history |

Templates support an optional placeholder (`{content}` or `{history}`). If present, the relevant text is substituted in. If absent, it is appended to the end of the prompt. This means you can write free-form instructions without worrying about placeholder syntax.
