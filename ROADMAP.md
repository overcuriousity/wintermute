# Roadmap: Autonomous Self-Improving Architecture

This document outlines the architectural evolution of Wintermute toward a truly autonomous, self-learning, self-improving system — while remaining auditable and optimized for small/local/weak LLMs.

## Competitive Analysis: Wintermute vs. OpenClaw

OpenClaw (formerly Clawdbot, by Peter Steinberger) is the current reference point for open-source autonomous AI assistants, having gained 100k+ GitHub stars within a week of launch. Understanding where it succeeds and where Wintermute can surpass it is the foundation for this roadmap.

### Where OpenClaw Leads

| Capability | OpenClaw | Wintermute |
|---|---|---|
| **Gateway architecture** | WebSocket control plane (`ws://127.0.0.1:18789`) as single source of truth for sessions, routing, channels | No unified control plane; interfaces connect directly to `LLMThread` queue |
| **Multi-channel reach** | 50+ integrations (WhatsApp, Telegram, Signal, Discord, iMessage, Slack) | Matrix + Web UI only |
| **Skill ecosystem** | ClawHub registry with install gating; agents auto-discover and pull skills | Local-only `data/skills/*.md`; no discovery or sharing |
| **Multi-agent routing** | `AGENTS.md` routes channels/peers to isolated agent instances with separate memory | Single agent instance; sub-sessions are workers, not independent agents |
| **Cron + wakeups + webhooks** | Three autonomous trigger types built into the Gateway | Event-driven + cron scheduling |
| **Community & ecosystem** | MIT license, massive contributor base, plugin marketplace | Single-developer project |

### Where Wintermute Already Leads

| Capability | Wintermute | OpenClaw |
|---|---|---|
| **Weak LLM optimization** | NL tool schemas, prompt assembly tuned for small models, tool profiles to limit cognitive load | "Recommends Anthropic Pro/Max + Opus 4.6 for long-context strength" — designed for frontier models |
| **Quality assurance** | Turing Protocol: 3-stage detect→validate→correct pipeline with 7 built-in hooks | AGENTS.md safety rules are static guidelines, not runtime enforcement |
| **Sub-session DAG** | Full dependency graph with `depends_on`, time gates, timeout continuation, nested workers | Flat task execution; no structured dependency resolution |
| **Context compaction** | Rolling summaries with chained compaction, 30-day archival, token budget awareness | Session-based; no documented compaction strategy |
| **Memory consolidation** | Nightly dreaming cycle: memory dedup, agenda review, skill condensation | Persistent markdown files; no automated consolidation |
| **Outcome tracking** | Structured recording of sub-session outcomes with historical feedback for future spawns | No documented outcome tracking |
| **E2E encryption** | Native Matrix E2E via mautrix | Channel-dependent; no universal encryption |
| **Event-driven architecture** | Async pub/sub event bus with debounce, history ring-buffer; memory harvest and dreaming are event-triggered | No documented event system |

### The Gap That Matters

OpenClaw's autonomy is **broad but shallow**: many integrations, cron triggers, skill discovery — but no feedback loops, no self-evaluation, no learning from outcomes. It is a sophisticated tool-use agent that can be scheduled.

Wintermute's autonomy is **narrow but deeper**: fewer interfaces, but the Turing Protocol, outcome tracking, dreaming cycle, and DAG sub-sessions provide the substrate for genuine self-improvement. The missing pieces are architectural, not capability gaps.

**The strategic bet:** OpenClaw scales by adding integrations and relying on frontier models. Wintermute scales by making weak models smarter through architectural scaffolding — feedback loops, structured self-reflection, and progressive skill evolution. This is a fundamentally different (and more defensible) approach.

---

## Completed Work

### ~~Phase A: Unified Task System~~ ✅ DONE

The former agenda/routine split has been resolved. A single `tasks` table in SQLite now handles both tracked items and scheduled actions:

- `task` tool with actions: `add`, `update`, `complete`, `pause`, `resume`, `delete`, `list`
- Schedule types: `once`, `daily`, `weekly`, `monthly`, `interval`
- `ai_prompt` + `background` flag enables autonomous sub-session execution on schedule
- APScheduler persists jobs; `schedule_config` JSON survives pause/resume
- Operational metrics tracked per-task: `run_count`, `last_run_at`, `last_result_summary`

**Remaining minor enhancement (not blocking):** Computed metrics from `interaction_log` + `sub_session_outcomes` could be surfaced in task list output (success rate, avg duration). This is a query, not a schema change.

### ~~Phase B: Event Bus~~ ✅ DONE

Full async pub/sub event bus (`wintermute/infra/event_bus.py`) wired into all components:

- `emit()`, `subscribe()`, `unsubscribe()`, `history()` with 1000-event ring-buffer
- Per-subscriber `debounce_ms` support for coalescing rapid-fire events
- Error isolation per subscriber

**Events emitted:**
- `message.received`, `message.sent` (LLMThread)
- `sub_session.started`, `sub_session.completed`, `sub_session.failed` (sub_session)
- `tool.executed` (main + sub-session scope)
- `task.created`, `task.completed`, `task.fired` (tools + scheduler)
- `memory.appended`, `skill.added` (tools)
- `dreaming.started`, `dreaming.completed` (dreaming)
- `harvest.started`, `harvest.completed` (memory_harvest)

**Event-driven behavior (replaces timer-based polling):**
- Memory harvest wakes on `message.received` count threshold
- Dreaming reacts to `memory.appended` (with 5-min debounce)
- Scheduler observes `task.created`

New events (e.g. `inference.completed` with token telemetry) can be added incrementally as later phases need them.

### ~~Phase C: Audit Infrastructure~~ ✅ DONE (via existing systems)

The `interaction_log` table comprehensively records every autonomous action post-hoc:

- Every LLM inference call (action, session, model, input, output, status, raw tool_calls)
- Every tool execution (tool name, arguments, result)
- Every Turing Protocol verdict (detection, validation, correction stages)
- Every sub-session round, dreaming cycle, memory harvest, compaction

The `sub_session_outcomes` table provides structured metrics per sub-session: status, tools_used, tool_call_count, duration, turing_verdict, continuation_count, backend_used, objective embedding for similarity search.

The `/debug` web panel surfaces sub-sessions, jobs, tasks, interaction log, and an SSE stream.

Pre-execution approval gating, if ever needed, is covered by extending the Turing Protocol (which already has `pre_execution` phase and scope-based filtering).

---

## Current Architectural Problems (Remaining)

### 1. No Feedback Loops

Sub-sessions complete, results are delivered, done. The system never asks:
- "Did that actually work?"
- "Should I try a different approach next time?"
- "This skill led to failure 3 times — is it outdated?"

Outcome tracking records data, but nothing consumes it for adaptation. It is write-only telemetry.

### 2. No Introspection

The system cannot query its own operational state:
- What sub-sessions failed today and why?
- Which memories were loaded but never relevant?
- What's my success rate by task type?

### 3. Skills Are Static Documentation

Skills are `.md` files the LLM reads. The system can't:
- Track which skills are actually used
- Test whether a skill's procedure still works
- Improve a skill based on observed outcomes
- Retire skills that are never loaded

---

## Roadmap Phases

### Phase 1: Reflection Cycle

**Priority: Highest — the single most differentiating capability**

**Close the feedback loop: Execute → Observe → Reflect → Adapt**

A new event-triggered worker (`reflection.py`) that:

1. **Reads recent outcomes** from `interaction_log`, `sub_session_outcomes`, and `event_bus.history()`
2. **Identifies patterns:** repeated failures, unused skills, tasks that stall
3. **Proposes adaptations:**
   - Adjust a task's cron schedule (back off on repeated NO_ACTION)
   - Pause a task that keeps failing
   - Update a skill based on what worked
   - Flag skills with high failure correlation
4. **Executes adaptations** via existing tools (task update/pause, skill edit) — all mutations are git-versioned in `data/`

**Trigger conditions:**
- `sub_session.failed` → immediate reflection on that failure
- Batch reflection after every N completed sub-sessions
- Periodic sweep (daily, during dreaming window)

**Weak-LLM optimization:** The reflection prompt is tightly scoped — it receives only the specific outcomes and context, not the full conversation. Uses the `compaction` backend pool (typically a cheaper/faster model). Reflection frequency auto-adjusts: fewer events = fewer reflection calls.

**Comparison to OpenClaw:** OpenClaw has no reflection mechanism. Its agents execute tasks and move on. This is the single most differentiating capability Wintermute can build.

### Phase 2: Self-Model

**Structured self-knowledge maintained by the reflection cycle:**

A `data/self_model.yaml` file containing:

```yaml
capabilities:
  shell_commands: {success_rate: 0.92, common_failures: ["timeout on network ops"]}
  web_search: {success_rate: 0.78, note: "SearXNG instance unreliable after midnight"}

performance:
  avg_sub_session_duration_s: 45
  avg_tool_calls_per_session: 3.2
  compaction_frequency: "every ~40 messages"
  token_budget_utilization: 0.73

operational:
  peak_hours: ["09:00-12:00", "14:00-17:00"]  # when user is active
  quiet_hours: ["01:00-07:00"]  # good for background work
  backend_latency:
    base: {p50_ms: 1200, p99_ms: 4500}
    sub_sessions: {p50_ms: 800, p99_ms: 3000}

skills:
  most_used: ["calendar.md", "deploy-docker.md"]
  never_used: ["legacy-backup.md"]

tasks:
  active: 12
  completion_rate_30d: 0.68
  avg_attempts_to_complete: 2.1
```

The reflection cycle updates this file. The prompt assembler includes a compact summary in the system prompt. The LLM can then make informed decisions: "my web search is unreliable at night, I'll schedule this for morning" or "this skill has never been used, I can skip loading it."

**Weak-LLM optimization:** The self-model is injected as structured YAML, not prose. Small models parse structured data more reliably than narrative self-descriptions. The self-model also enables the system to auto-tune its own parameters (compaction threshold, sub-session timeout defaults) without requiring the user to configure them.

**Prerequisite:** Phase 1 (reflection cycle generates self-model updates). Also requires adding `inference.completed` events with token count/duration to the event bus (incremental addition to existing infrastructure).

### Phase 3: Skill Evolution

**Transform skills from static docs to living, tested procedures:**

- **Usage tracking:** Every `read_file("data/skills/X.md")` call increments a counter. The reflection cycle uses this to identify hot vs. cold skills.
- **Outcome correlation:** When a sub-session loads a skill and succeeds/fails, the outcome is associated with the skill. Skills with high failure correlation get flagged for review.
- **Versioned updates with rationale:** Skill modifications (by dreaming or reflection) include a changelog entry. The `data/` git repo already provides history, but an in-file `## Changelog` section makes it visible to the LLM.
- **Retirement:** Skills unused for 90 days are auto-archived (moved to `data/skills/.archive/`). The reflection cycle can propose this; dreaming can execute it.
- **Skill synthesis:** When the reflection cycle observes a pattern across multiple successful sessions that isn't captured in any skill, it can propose a new skill. This is genuine self-improvement — the system learns procedures from its own experience.

**Prerequisite:** Phase 1 (reflection cycle drives all skill evolution decisions).

---

## Implementation Order & Dependencies

```
Completed:
  ✅ Unified Task System
  ✅ Event Bus
  ✅ Audit Infrastructure (interaction_log + sub_session_outcomes + /debug)

Remaining:
  Phase 1: Reflection Cycle ─→ Phase 2: Self-Model
                             ─→ Phase 3: Skill Evolution
```

Phase 1 is the critical path. Phases 2 and 3 can be developed in parallel once Phase 1 is operational.

## Design Principles

1. **Autonomy increases monotonically with auditability.** Every new capability comes with a corresponding visibility mechanism.
2. **Optimize for token poverty.** Every architectural decision should reduce, not increase, the number of inference calls needed. If a feature requires an extra LLM call, it must justify its cost.
3. **Structured over narrative.** Small models handle YAML, JSON, and schemas better than prose instructions. Prefer structured data formats for all system-internal communication.
4. **Fail gracefully, not silently.** Failed sub-sessions, stalled tasks, and broken skills should surface — not disappear into logs.
5. **Git is the undo button.** All autonomous mutations to `data/` are auto-committed. The user can always `cd data && git log` to see what changed and revert.
6. **Don't increase LLM cognitive load.** New capabilities should be backend features that enrich existing tool outputs, not new tool schemas the LLM must learn. The task tool schema stays simple; computed metrics are injected into responses.
