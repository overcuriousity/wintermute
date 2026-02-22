# System Prompts

## Overview

Wintermute assembles its system prompt fresh on every inference call from individual file components stored in `data/`.

## Components

### 1. BASE_PROMPT.txt — Core Instructions

The foundation of every prompt. Shipped with the repository and editable by the user.

Content covers:

- Role definition (personal AI assistant via Matrix chat)
- Memory system: clear boundary between MEMORIES (long-term facts) and AGENDA (active working memory)
- Background process awareness (agenda reviews, nightly dreaming, context compaction)
- Tool usage patterns and categories
- Sub-session delegation and workflow DAGs with `depends_on`
- System event handling (sub-session results, routines, errors, timeouts)
- Behavioural guidelines (concise responses, credential policy, confidence)

#### Sectioned BASE_PROMPT

BASE_PROMPT.txt is divided into named sections using HTML comment markers:

```
<!-- section: knowledge_routing requires: append_memory,agenda,add_skill -->
```

Each section declares which tool(s) it requires. When a sub-session is spawned with a limited tool set, only sections whose required tools are available are included. Sections marked `always` are always included.

| Section | Required Tools | Content |
|---------|---------------|---------|
| `core` | always | Personality, environment, prompt note |
| `knowledge_routing` | `append_memory`, `agenda`, `add_skill` (any) | Memory/agenda/skill/routine routing |
| `delegation` | `spawn_sub_session` | Task delegation patterns |
| `routines` | `set_routine` | Scheduled task instructions |
| `system_events` | `spawn_sub_session` | System event handling |
| `guidelines` | always | Guidelines, critical rules, personality |

This reduces effective prompt size for minimal sub-sessions by ~800 tokens — significant for weak/small models where every token counts.

### 2. MEMORIES.txt — Long-Term Facts

Stores persistent facts about the user — preferences, biographical details, established decisions. Updated day-to-day via `append_memory` (preferred). Consolidated nightly by the dreaming loop to merge duplicates and prune outdated entries.

Key rule: if information would still matter in a month with no active project around it, it belongs in MEMORIES.

### 3. Agenda (SQLite) — Working Memory

Active goals, ongoing projects, time-sensitive tasks, and open questions. Managed via the `agenda` tool (add, complete, update, list actions). Each item has a priority (1=urgent, 10=low) and an auto-assigned ID. Reviewed autonomously every 60 minutes by the agenda loop and consolidated nightly by the dreaming loop.

Key rule: if it only matters because something is in progress right now, it belongs in Agenda.

### 4. skills/*.md — Learned Procedures

Each file in `data/skills/` is a Markdown document describing a reusable procedure the AI has learned. Created via the `add_skill` tool. The first line of each file serves as a one-line summary.

Only a table of contents (skill name + summary) is injected into the system prompt. The full skill content is loaded on demand by the LLM via `read_file` when relevant to the current task. This keeps the prompt lightweight even with many skills.

### 5. SEED_{language}.txt — Conversation Seed

Injected as a system event when a new conversation starts (first user message in an empty thread or after `/new`). Prompts the LLM to introduce itself, surface relevant memories and agendas, and explain its capabilities.

Language-specific files (`SEED_en.txt`, `SEED_de.txt`, `SEED_fr.txt`, `SEED_es.txt`, `SEED_it.txt`, `SEED_zh.txt`, `SEED_ja.txt`) are selected via the `seed.language` config option. Falls back to English if the configured language is missing. Add new languages by creating `data/prompts/SEED_{code}.txt`.

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

# Active Agenda
{formatted agenda items from DB, e.g. "[P2] #3: Fix auth bug"}

---

# Conversation Summary
{compaction summary, if context was compacted}

---

# Skills
Load a skill with read_file when relevant to the current task.
- data/skills/calendar.md — Google Calendar event management
- data/skills/deploy-docker.md — Docker container deployment workflow
```

Sections are separated by `---` dividers. Empty sections are omitted (except Skills, which always appears with at least the load instruction).

## Sub-session System Prompt Modes

When spawning sub-sessions, the system prompt varies by mode:

| Mode | System Prompt Content |
|------|----------------------|
| `minimal` | Lightweight execution agent instructions (default) |
| `full` | Full assembled prompt (sectioned BASE + MEMORIES + AGENDA + SKILLS) |
| `base_only` | Sectioned BASE_PROMPT.txt only |
| `none` | No system prompt (bare tool-use loop) |

Both `full` and `base_only` modes use the sectioned BASE_PROMPT — sections irrelevant to the worker's available tools are automatically omitted. All modes append the task context and objective at the end.

## Size Limits and Auto-summarisation

Each component has a configurable character limit (set in `config.yaml` under `context.component_size_limits`):

| Component | Default Limit | Action When Exceeded |
|-----------|---------------|---------------------|
| MEMORIES.txt | 10,000 chars | AI is asked to condense and prioritise |
| Agenda (DB) | 5,000 chars | AI is asked to condense and prioritise |
| skills/ (TOC) | 2,000 chars | AI is asked to reorganise |

When a component exceeds its limit after any inference, a system event is enqueued asking the AI to read the component, condense it, and update it using the appropriate tool.

The nightly dreaming loop also consolidates MEMORIES.txt and agenda items independently using a direct LLM call (no tool loop).

## Customisable Prompt Templates

The following prompt templates are stored as editable files in `data/` and shipped with the repository. They can be freely modified.

| File | Used By | Placeholder | Purpose |
|------|---------|-------------|---------|
| `DREAM_MEMORIES_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating MEMORIES.txt overnight |
| `DREAM_AGENDA_PROMPT.txt` | Dreaming loop | `{content}` | Instructions for consolidating agenda items overnight (LLM returns JSON actions) |
| `COMPACTION_PROMPT.txt` | Context compaction | `{history}` | Instructions for summarising old conversation history |

Templates support an optional placeholder (`{content}` or `{history}`). If present, the relevant text is substituted in. If absent, it is appended to the end of the prompt. This means you can write free-form instructions without worrying about placeholder syntax.
