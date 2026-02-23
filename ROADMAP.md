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
| **Cron + wakeups + webhooks** | Three autonomous trigger types built into the Gateway | Timer-based polling only (agenda loop, memory harvest, scheduler) |
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

### The Gap That Matters

OpenClaw's autonomy is **broad but shallow**: many integrations, cron triggers, skill discovery — but no feedback loops, no self-evaluation, no learning from outcomes. It is a sophisticated tool-use agent that can be scheduled.

Wintermute's autonomy is **narrow but deeper**: fewer interfaces, but the Turing Protocol, outcome tracking, dreaming cycle, and DAG sub-sessions provide the substrate for genuine self-improvement. The missing pieces are architectural, not capability gaps.

**The strategic bet:** OpenClaw scales by adding integrations and relying on frontier models. Wintermute scales by making weak models smarter through architectural scaffolding — feedback loops, structured self-reflection, and progressive skill evolution. This is a fundamentally different (and more defensible) approach.

---

## Current Architectural Problems

### 1. Agenda vs. Routines: A False Dichotomy

An agenda item is "something that should be done" (SQLite). A routine is "something that fires on a schedule" (APScheduler). But:
- An agenda review *is* a routine (fixed interval timer)
- A routine with `ai_prompt` *is* a task that should track completion — i.e., an agenda item
- Neither system knows about the other's state

They are the same concept split across two storage backends with incompatible semantics.

### 2. No Feedback Loops

Sub-sessions complete, results are delivered, done. The system never asks:
- "Did that actually work?"
- "Should I try a different approach next time?"
- "This skill led to failure 3 times — is it outdated?"

Outcome tracking records data, but nothing consumes it for adaptation. It is write-only telemetry.

### 3. No Introspection

The system cannot query its own operational state:
- What sub-sessions failed today and why?
- Which memories were loaded but never relevant?
- How many NO_ACTION agenda reviews ran (wasted tokens)?
- What's my success rate by task type?

### 4. Timer-Based Everything

Agenda reviews, memory harvesting, and dreaming all run on fixed timers regardless of whether there's anything to do. This wastes tokens on weak/local models where every inference call is expensive.

### 5. Skills Are Static Documentation

Skills are `.md` files the LLM reads. The system can't:
- Track which skills are actually used
- Test whether a skill's procedure still works
- Improve a skill based on observed outcomes
- Retire skills that are never loaded

---

## Roadmap Phases

### Phase 1: Intent Log & Audit Infrastructure

**Priority: Highest — prerequisite for all autonomy increases**

Every autonomous action (agenda review, routine execution, memory consolidation, skill modification) produces an **intent record** before execution:

```
IntentRecord:
  id, timestamp, actor (agenda_review | reflection | goal_pursuit | dreaming)
  action: structured description of what it wants to do
  reasoning: LLM-generated explanation
  status: proposed → approved → executed | vetoed
  approval_policy: auto | user_required
  cost_estimate: estimated token spend
```

**Approval tiers** (configurable):
- **Auto-approve:** read_file, search_web, append_memory (low risk, reversible)
- **Auto-approve + log:** set_routine, agenda complete, update skill (medium risk, git-reversible)
- **Require approval:** delete_skill, external-facing actions, self-schedule modification (high risk)

**Audit UI:** Extend the existing `/debug` web panel with an Intent Feed — chronological log of all autonomous decisions with reasoning, outcome, and cost.

**Weak-LLM optimization:** Intent records are generated by the acting sub-session as structured tool output, not by a separate evaluation call. Zero additional inference cost.

### Phase 2: Unified Goal System

**Replace both agenda items and routines with a single Goal abstraction:**

```
Goal:
  id, description, status (active | paused | completed | failed | abandoned)
  strategy: nullable — can be discovered through reflection
  trigger: cron | event | condition | manual
  success_criteria: how to verify completion
  parent_goal_id: nullable (hierarchical decomposition)
  priority: integer
  metrics: {attempts, successes, failures, avg_duration, last_outcome}
  thread_id: nullable (scoped to a conversation, or global)
```

**Migration path:**
- Existing agenda items become Goals with `trigger: manual` or `trigger: condition`
- Existing routines become Goals with `trigger: cron`
- The `set_routine` and `agenda` tools merge into a single `goal` tool with actions: `create`, `update`, `complete`, `pause`, `list`, `decompose`
- APScheduler remains the execution backend for cron-triggered goals, but goal state lives in SQLite

**Why this matters for weak LLMs:** Currently the LLM must decide at creation time whether something is an "agenda item" or a "routine" — a premature classification that small models get wrong. A Goal is just "something to be done" with optional scheduling. The LLM's cognitive load decreases.

### Phase 3: Event Bus

**Replace timer-based polling with in-process async pub/sub:**

```python
class EventBus:
    async def emit(self, event: str, payload: dict)
    def subscribe(self, event: str, handler: Callable)
```

**Core events:**
- `message.received`, `message.sent`
- `sub_session.completed`, `sub_session.failed`
- `goal.created`, `goal.completed`, `goal.stalled`
- `memory.appended`, `memory.consolidated`
- `skill.loaded`, `skill.updated`
- `inference.completed` (with token count, duration)

**What changes:**
- Memory harvest triggers on `message.received` count threshold instead of polling every 60s
- Agenda review triggers on `goal.created` or `goal.stalled` instead of fixed interval
- Dreaming still runs on cron (it's genuinely time-based) but can also trigger on `memory.appended` count threshold
- Goals with `trigger: event` subscribe to specific events

**Weak-LLM optimization:** Eliminates wasted inference on empty polling cycles. A local 8B model running on CPU should only fire when there's actual work to do.

### Phase 4: Reflection Cycle

**Close the feedback loop: Execute → Observe → Reflect → Adapt**

A new periodic process (event-triggered, not timer-based) that:

1. **Reads recent outcomes** from the intent log and goal metrics
2. **Identifies patterns:** repeated failures, unused skills, goals that stall
3. **Proposes adaptations:**
   - Update a Goal's strategy
   - Adjust a Goal's cron schedule (back off on repeated NO_ACTION)
   - Pause a Goal that keeps failing
   - Update a skill based on what worked
   - Create a new sub-goal to unblock a stalled parent
4. **Records proposals as intent records** (subject to approval policy)

**Trigger conditions:**
- `sub_session.failed` → immediate reflection on that failure
- `goal.stalled` (no progress in N cycles) → strategy review
- Batch reflection after every N completed goals

**Weak-LLM optimization:** The reflection prompt is tightly scoped — it receives only the specific outcomes and goal context, not the full conversation. Uses the `compaction` backend pool (typically a cheaper/faster model). Reflection frequency auto-adjusts: fewer events = fewer reflection calls.

**Comparison to OpenClaw:** OpenClaw has no reflection mechanism. Its agents execute tasks and move on. This is the single most differentiating capability Wintermute can build.

### Phase 5: Self-Model

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

goals:
  active: 12
  completion_rate_30d: 0.68
  avg_attempts_to_complete: 2.1
```

The reflection cycle updates this file. The prompt assembler includes a compact summary in the system prompt. The LLM can then make informed decisions: "my web search is unreliable at night, I'll schedule this for morning" or "this skill has never been used, I can skip loading it."

**Weak-LLM optimization:** The self-model is injected as structured YAML, not prose. Small models parse structured data more reliably than narrative self-descriptions. The self-model also enables the system to auto-tune its own parameters (compaction threshold, sub-session timeout defaults) without requiring the user to configure them.

### Phase 6: Skill Evolution

**Transform skills from static docs to living, tested procedures:**

- **Usage tracking:** Every `read_file("data/skills/X.md")` call increments a counter. The reflection cycle uses this to identify hot vs. cold skills.
- **Outcome correlation:** When a sub-session loads a skill and succeeds/fails, the outcome is associated with the skill. Skills with high failure correlation get flagged for review.
- **Versioned updates with rationale:** Skill modifications (by dreaming or reflection) include a changelog entry. The `data/` git repo already provides history, but an in-file `## Changelog` section makes it visible to the LLM.
- **Retirement:** Skills unused for 90 days are auto-archived (moved to `data/skills/.archive/`). The reflection cycle can propose this; dreaming can execute it.
- **Skill synthesis:** When the reflection cycle observes a pattern across multiple successful sessions that isn't captured in any skill, it can propose a new skill. This is genuine self-improvement — the system learns procedures from its own experience.

---

## Implementation Order & Dependencies

```
Phase 1: Intent Log ──────────────────────────────┐
                                                   │
Phase 2: Unified Goals ───────┐                    │
                              ├─→ Phase 4: Reflection Cycle ─→ Phase 5: Self-Model
Phase 3: Event Bus ───────────┘                    │
                                                   │
                                        Phase 6: Skill Evolution
```

Phases 1-3 can be developed in parallel. Phase 4 requires all three. Phases 5-6 build on Phase 4.

## Design Principles

1. **Autonomy increases monotonically with auditability.** Every new capability comes with a corresponding visibility mechanism.
2. **Optimize for token poverty.** Every architectural decision should reduce, not increase, the number of inference calls needed. If a feature requires an extra LLM call, it must justify its cost.
3. **Structured over narrative.** Small models handle YAML, JSON, and schemas better than prose instructions. Prefer structured data formats for all system-internal communication.
4. **Fail gracefully, not silently.** Failed sub-sessions, stalled goals, and broken skills should surface — not disappear into logs.
5. **Git is the undo button.** All autonomous mutations to `data/` are auto-committed. The user can always `cd data && git log` to see what changed and revert.
