# System Prompts

## Overview

Wintermute assembles its system prompt fresh on every inference call from individual file components stored in `data/`.

## Components

### 1. BASE_PROMPT.txt — Core Instructions

The immutable foundation. Created automatically on first run with sensible defaults. Editable by the user.

Default content covers:

- Role definition (personal AI assistant)
- Memory system usage (when to update MEMORIES, PULSE, skills)
- Behavioural guidelines (concise responses, credential awareness, confidence)
- Tool-use patterns

### 2. MEMORIES.txt — Long-Term Facts

Stores persistent facts about the user. Updated by the AI via the `update_memories` tool whenever the user shares important information. Survives restarts.

### 3. PULSE.txt — Working Memory

Active goals, ongoing projects, and recurring concerns. Updated via the `update_pulse` tool. Reviewed periodically by the pulse loop and consolidated nightly by the dreaming loop.

### 4. skills/*.md — Learned Procedures

Each file in `data/skills/` is a Markdown document describing a reusable procedure the AI has learned. Created via the `add_skill` tool. All skills are loaded and injected into every prompt.

## Prompt Assembly Pipeline

The `prompt_assembler.assemble()` function builds the final system prompt:

```
# Core Instructions
{BASE_PROMPT.txt content}

---

# User Memories
{MEMORIES.txt content}

---

# Active Pulse
{PULSE.txt content}

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
| PULSE.txt | 5,000 chars | AI is asked to condense and prioritise |
| skills/ (total) | 20,000 chars | AI is asked to reorganise |

When a component exceeds its limit after any inference, a system event is enqueued asking the AI to read the component, condense it, and update it using the appropriate tool.

The nightly dreaming loop also consolidates MEMORIES.txt and PULSE.txt independently using a direct LLM call (no tool loop).

## Customisable Prompt Templates

The following prompt templates are stored as editable files in `data/`. They are created with sensible defaults on first run and can be freely modified.

| File | Used By | Placeholder | Purpose |
|------|---------|-------------|---------|
| `DREAM_MEMORIES_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating MEMORIES.txt overnight |
| `DREAM_PULSE_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating PULSE.txt overnight |
| `COMPACTION_PROMPT.txt` | Context compaction | `{history}` | Instructions for summarising old conversation history |

Templates support an optional placeholder (`{content}` or `{history}`). If present, the relevant text is substituted in. If absent, it is appended to the end of the prompt. This means you can write free-form instructions without worrying about placeholder syntax.
