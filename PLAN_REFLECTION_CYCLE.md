# Implementation Plan: Reflection Cycle

## Context

Wintermute records extensive telemetry (`interaction_log`, `sub_session_outcomes`, `event_bus` history) but nothing consumes this data for adaptation. Sub-sessions fail, tasks stall, skills rot — and the system never learns. The reflection cycle closes this feedback loop: **observe → analyze → act**.

This is Phase 1 of the remaining roadmap — the critical path to self-improvement.

---

## Design

Three-tier architecture, optimized for token poverty:

1. **Rule engine** (zero LLM cost) — Programmatic pattern detection on DB/event data. Auto-applies simple actions (pause failing tasks, flag stale skills). Runs on every trigger.
2. **LLM analysis** (cheap, one-shot) — Direct `pool.call()` with a prose prompt summarizing recent outcomes. Produces human-readable observations logged to `interaction_log`. No structured output required. Runs only when the rule engine finds something interesting.
3. **Sub-session mutations** (expensive, rare) — Spawned only when the LLM analysis recommends a creative change (e.g. rewriting a skill). Uses constrained tool set. Runs only on explicit LLM recommendation.

---

## Triggers (event-driven, no polling)

- `sub_session.failed` → immediate rule-engine check on that failure + related history
- `sub_session.completed` count threshold (configurable, default every 10 completions) → batch analysis
- No fixed timer. Zero cost when idle.

Wake-up mechanism: `asyncio.Event` (same pattern as `MemoryHarvestLoop`). Event bus callbacks set the event; the main loop blocks on `asyncio.wait_for(event.wait(), timeout=fallback_poll)`.

---

## Tier 1: Rule Engine (programmatic, no LLM)

Checks run against `database.get_outcomes_since()` + `event_bus.history()`:

| Rule | Condition | Action |
|---|---|---|
| **Consecutive failures** | A scheduled task's sub-sessions failed 3+ times in a row | `database.pause_task(task_id)` + remove APScheduler job + log finding |
| **Timeout pattern** | A scheduled task's sub-sessions timed out 3+ times consecutively | Log warning (may need longer timeout or simpler ai_prompt) |
| **Stale task** | Scheduled task with `run_count > 0` but all recent runs produced no meaningful output (result_length < threshold) | Log warning |
| **Skill failure correlation** | A skill file was loaded (via `read_file` tool call in `interaction_log`) in 3+ failed sub-sessions within the lookback window | Emit `reflection.skill_flagged` event + log |

Each rule produces a `ReflectionFinding` dataclass:

```python
@dataclass
class ReflectionFinding:
    rule: str              # "consecutive_failures", "timeout_pattern", etc.
    severity: str          # "warning" | "action_taken"
    subject_type: str      # "task" | "skill" | "sub_session"
    subject_id: str        # task_id, skill filename, or session_id
    detail: str            # human-readable description
    action_taken: str      # "" if warning-only, or "paused_task" etc.
```

Findings are logged to `interaction_log` with `action="reflection_rule"`, `session="system:reflection"`, `output=json.dumps(finding)`.

### How consecutive failures are tracked

The key challenge: linking sub-session outcomes back to the task that triggered them. The `scheduler_thread._fire_task()` spawns sub-sessions with objectives derived from `task.ai_prompt`. The `sub_session_outcomes` table records `objective` but not `task_id`.

**Solution:** Add `task_id` as an optional field to `sub_session_outcomes` (via inline `ALTER TABLE`). When `_fire_task()` spawns a sub-session, pass the `task_id` through to `spawn()` as a new optional parameter, which flows through to `_persist_outcome()`. This is the cleanest approach — avoids fragile objective-text matching.

For existing outcomes (before migration): fall back to matching `objective LIKE '%' || task.ai_prompt || '%'` with a length guard.

---

## Tier 2: LLM Analysis (event-triggered, cheap)

Only fires when the rule engine produced findings. A direct `pool.call()` with a single-turn prompt.

### Prompt template: `data/prompts/REFLECTION_ANALYSIS.txt`

```
You are reviewing the recent operational history of an AI assistant system.

## Rule Engine Findings
{findings}

## Recent Failed Sub-Sessions
{failed_sessions}

## Active Tasks
{active_tasks}

Based on this data, provide brief observations:
1. What patterns do you see in the failures?
2. Are any tasks misconfigured or need adjustment?
3. Should any skills be updated or retired?

If you recommend updating or creating a skill, start the relevant paragraph with "SKILL_ACTION:" followed by your recommendation. Keep your response concise.
```

The LLM responds with prose. Logged to `interaction_log` with `action="reflection_analysis"`.

### Parsing for actionable recommendations

After logging, a simple check: if the response contains `"SKILL_ACTION:"`, extract that paragraph and spawn a Tier 3 sub-session with it as the objective. This is intentionally low-tech — we're not asking the LLM for structured output, just using a keyword marker that even weak models can produce reliably.

---

## Tier 3: Sub-session Mutations (rare, creative)

Spawned only when Tier 2 produces a `SKILL_ACTION:` recommendation.

```python
self._sub_sessions.spawn(
    objective=skill_action_text,
    tool_names=["read_file", "add_skill", "append_memory"],
    system_prompt_mode="none",
    pool=self._pool,
    parent_thread_id=None,    # fire-and-forget, silent
    skip_tp_on_exit=True,
    max_rounds=5,             # hard cap — skill edits shouldn't take many rounds
)
```

All skill mutations are auto-committed by the existing `data/` git versioning. The user can review via `cd data && git log`.

---

## Implementation Details

### New file: `wintermute/workers/reflection.py`

```python
@dataclass
class ReflectionConfig:
    enabled: bool = True
    batch_threshold: int = 10           # trigger batch analysis every N completions
    consecutive_failure_limit: int = 3  # auto-pause after N consecutive failures
    lookback_seconds: int = 86400       # 24h window for pattern detection
    min_result_length: int = 50         # below this = "no meaningful output" for stale detection

class ReflectionLoop:
    def __init__(self, config, sub_session_manager, pool, event_bus):
        self._cfg = config
        self._sub_sessions = sub_session_manager
        self._pool = pool
        self._event_bus = event_bus
        self._running = False
        self._completed_count = 0
        self._check_event = asyncio.Event()
        self._event_bus_subs: list[str] = []

    async def run(self):
        # Subscribe to sub_session.failed and sub_session.completed
        # Main loop: wait for _check_event, run _run_rules(), optionally _run_analysis()

    def stop(self):
        # Set _running = False, set _check_event, unsubscribe all

    async def _on_sub_session_failed(self, event):
        # Set _check_event immediately

    async def _on_sub_session_completed(self, event):
        # Increment _completed_count, set _check_event if >= batch_threshold

    async def _run_rules(self) -> list[ReflectionFinding]:
        # Query DB for recent outcomes, check each rule, return findings
        # For consecutive failures: query get_task_failure_streak()
        # For skill correlation: query interaction_log for read_file tool calls in failed sessions

    async def _run_analysis(self, findings: list[ReflectionFinding]):
        # Build prompt from template + findings + context
        # Direct pool.call()
        # Log to interaction_log
        # Check for SKILL_ACTION: marker → spawn mutation sub-session

    async def _spawn_mutation(self, objective: str):
        # Spawn constrained sub-session for skill changes
```

### Modifications to existing files

#### `wintermute/core/llm_thread.py`

Add `reflection` field to `MultiProviderConfig`:

```python
@dataclass
class MultiProviderConfig:
    main: list[ProviderConfig]
    compaction: list[ProviderConfig]
    sub_sessions: list[ProviderConfig]
    dreaming: list[ProviderConfig]
    turing_protocol: list[ProviderConfig]
    memory_harvest: list[ProviderConfig]
    nl_translation: list[ProviderConfig]
    reflection: list[ProviderConfig]       # ← NEW, fallback to compaction
```

#### `wintermute/main.py`

1. Import `ReflectionConfig, ReflectionLoop` from `wintermute.workers.reflection`
2. In `_build_multi_provider_config()`: parse `llm.reflection` with fallback to `compaction` pool (following the `memory_harvest` pattern):
   ```python
   refl_raw = llm_raw.get("reflection")
   compaction_configs = _get_role("compaction")
   if refl_raw is None:
       refl_configs = list(compaction_configs)
   elif isinstance(refl_raw, list):
       refl_configs = _resolve_role("reflection", refl_raw, backends) if refl_raw else list(compaction_configs)
   ```
3. Build pool: `reflection_pool = _build_pool(multi_cfg.reflection, client_cache)`
4. Parse config:
   ```python
   reflection_raw = cfg.get("reflection", {})
   reflection_cfg = ReflectionConfig(
       enabled=reflection_raw.get("enabled", True),
       batch_threshold=reflection_raw.get("batch_threshold", 10),
       consecutive_failure_limit=reflection_raw.get("consecutive_failure_limit", 3),
       lookback_seconds=reflection_raw.get("lookback_seconds", 86400),
   )
   ```
5. Instantiate (after sub_sessions manager):
   ```python
   reflection_loop = ReflectionLoop(
       config=reflection_cfg,
       sub_session_manager=sub_sessions,
       pool=reflection_pool,
       event_bus=event_bus,
   )
   ```
6. Launch: `asyncio.create_task(reflection_loop.run(), name="reflection")`
7. Shutdown: `reflection_loop.stop()`

#### `wintermute/infra/database.py`

Add two new query functions:

```python
def get_outcomes_since(
    since: float,
    status_filter: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Return sub-session outcomes newer than `since` timestamp."""
    where = "WHERE timestamp > ?"
    params: list = [since]
    if status_filter:
        where += " AND status = ?"
        params.append(status_filter)
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM sub_session_outcomes {where} "
            f"ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def get_task_failure_streak(task_id: str, limit: int = 10) -> int:
    """Count consecutive recent failures/timeouts for a task.
    Returns the streak length (0 if the most recent outcome was a success)."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT status FROM sub_session_outcomes "
            "WHERE task_id = ? ORDER BY timestamp DESC LIMIT ?",
            (task_id, limit),
        ).fetchall()
    streak = 0
    for row in rows:
        if row["status"] in ("failed", "timeout"):
            streak += 1
        else:
            break
    return streak
```

Add `task_id` column to `sub_session_outcomes` (inline migration in `_ensure_tables()`):

```python
_add_column("sub_session_outcomes", "task_id", "TEXT")
```

#### `wintermute/core/sub_session.py`

Add `task_id: Optional[str] = None` parameter to `spawn()`. Flow it through `_SubSessionState` into `_persist_outcome()` where it gets written to the new column.

#### `wintermute/workers/scheduler_thread.py`

In `_fire_task()`, pass `task_id=task_id` when calling `self._sub_sessions.spawn(...)`.

#### `wintermute/infra/prompt_loader.py`

Add `"REFLECTION_ANALYSIS"` to the optional prompts list (not required files — reflection degrades gracefully if missing, falling back to a hardcoded default).

#### `config.yaml.example`

Add:
```yaml
# Reflection cycle — autonomous pattern detection and adaptation
reflection:
  enabled: true
  batch_threshold: 10            # analyze after every N sub-session completions
  consecutive_failure_limit: 3   # auto-pause tasks after N consecutive failures
  lookback_seconds: 86400        # 24h window for pattern detection

# Under llm: section
# reflection:                    # LLM backend for reflection analysis
#   - your-backend-name          # Falls back to compaction backends if omitted
```

---

## Event Bus Integration

### New events emitted by reflection:

| Event | When | Data |
|---|---|---|
| `reflection.finding` | Rule engine produces a finding | `rule`, `severity`, `subject_id`, `action_taken` |
| `reflection.analysis_completed` | LLM analysis finishes | `findings_count`, `actions_recommended` |
| `reflection.skill_flagged` | Skill correlated with failures | `skill_name`, `failure_count` |

These events are available in `event_bus.history()` for the self-model (Phase 2) and are visible in the `/debug` SSE stream.

---

## Verification

1. **Startup:** `uv sync && uv run wintermute` — app starts, logs show `"Reflection loop started"` (or `"Reflection loop disabled"` if `enabled: false`)
2. **Single failure:** Trigger a failing sub-session → reflection fires on `sub_session.failed`, rule engine checks for patterns, logs finding if applicable
3. **Consecutive failures:** Create a scheduled task with a bad `ai_prompt`, let it fire 3 times → rule engine auto-pauses the task, `interaction_log` shows `action="reflection_rule"` with `action_taken="paused_task"`
4. **Batch analysis:** After 10 sub-session completions → LLM analysis runs, prose logged as `action="reflection_analysis"`
5. **Debug panel:** All reflection entries visible in `/debug` interaction log tab
6. **Graceful degradation:** If no reflection pool is configured, falls back to compaction pool. If no prompt template exists, uses hardcoded default. If `enabled: false`, loop exits immediately.

---

## Dependency Graph

```
                      ┌── Phase 2: Self-Model
Phase 1: Reflection ──┤   (reflection updates self_model.yaml)
                      └── Phase 3: Skill Evolution
                          (reflection drives skill tracking/retirement)
```

Phase 1 is self-contained and can ship independently. Phases 2 and 3 consume reflection's output but don't block it.
