# Plan: Weak-Model Resilience Features

Four features to improve Wintermute's reliability with small/local LLMs (9B class).
No graduated feature tiers. No keyword-based NL parsing.

**Implementation order:** Feature 4 → 2 → 1 → 3 (lowest risk first, most data-dependent last).

---

## Feature 1: Confidence Gating on Dreaming Output

**Problem:** Creative phases (prediction, schema, association) trust the LLM's self-reported `"confidence"` field. A weak model claims `"high"` for garbage output. The only current check filters out `"low"` — it does not validate content coherence.

**Approach:** Programmatic structural validators that run AFTER JSON parsing but BEFORE writing to memory store. These check measurable properties of the output, not the model's opinion of itself.

### Files to modify

**`wintermute/workers/dreaming.py`**

Add a validation function around line 95 (after helpers):

```python
def _validate_dreaming_output(text: str, phase: str, parsed: dict, context: dict) -> tuple[bool, str]:
```

#### Prediction validation (lines ~1036-1073, `_phase_prediction`):

1. `text` must be >= 20 characters (reject trivially short outputs)
2. `text` must not be a near-duplicate of the input activity summary (character overlap ratio > 0.7 means the model is parroting input)
3. For `temporal` type: structured suffix `||hours=...||` must have valid hour ranges (0-23) and valid day abbreviations. If malformed, downgrade to `behavioral` type rather than rejecting.
4. Reject if `text` contains 3+ consecutive identical words (common small-model hallucination)
5. Reject if `text` is identical (case-insensitive, stripped) to an existing prediction — fetch existing predictions via `memory_store.get_by_source("dreaming_prediction", 50, False)` before the loop, build a set of normalized texts.

Insert at line ~1041 before the `if text:` check:

```python
valid, reason = _validate_dreaming_output(text, "prediction", pred, {"existing": existing_texts})
if not valid:
    logger.info("Dreaming prediction rejected: %s", reason)
    continue
```

#### Schema validation (lines ~874-907, `_phase_schema`):

1. `schema_text` must be >= 15 characters
2. `schema_text` must not be a substring of any single cluster member (the model should generalize, not copy)
3. If `replaces_entries` is True but `confidence` is not `"high"`, force `replaces_entries = False` (programmatic override)
4. Reject if `schema_text` contains JSON fragments or markdown headers (`{"`, `##`, triple backticks — signs the model returned formatting instead of content)

Insert at line ~879 after `schema_text = parsed.get("schema", "").strip()`.

#### Association validation (lines ~761-773, `_phase_association`):

1. `text` must be >= 15 characters
2. `source_indices` must reference valid indices within the seed set (reject if any index >= len(seed_texts))
3. `source_indices` must have >= 2 entries
4. Reject if `text` is a near-substring of any single seed entry (>0.6 overlap ratio)

Insert at line ~766 after current confidence check.

### Config additions (`config.yaml.example`)

Under `memory.dreaming`:
```yaml
# ── Confidence Gating ──
confidence_min_text_length: 20      # Reject dreaming outputs shorter than this
confidence_max_input_overlap: 0.7   # Reject outputs that parrot >70% of input
```

### No new database tables. No new dependencies.

---

## Feature 2: Memory Write Validation

**Problem:** `append_memory` (used by harvest and direct AI calls) has no content validation. Weak models write garbage: single words, JSON fragments, system prompt echoes, looping hallucinations.

**Approach:** Validation gate in `memory_io.append_memory()` — the single chokepoint for all memory writes.

### Files to modify

**`wintermute/infra/memory_io.py`**

Add `_validate_memory_entry(entry: str, cfg: dict) -> tuple[bool, str]` before `append_memory()` (around line 24).

#### Validation rules:

1. **Minimum length:** `len(entry.strip()) < 10` → reject
2. **Maximum length:** `len(entry.strip()) > 2000` → reject (model dumping conversation/system prompt)
3. **JSON/code detection:** Reject if entry starts with `{` or `[` and is valid JSON (`json.loads` succeeds)
4. **System prompt echo:** Reject if entry contains any 2+ of: `"you are"`, `"your task"`, `"return json"`, `"convergence protocol"`, `"system event"`, `"sub-session"` (case-insensitive). These indicate the model is echoing instructions.
5. **Repetitive content:** Reject if any single word appears > 5 times (split whitespace, casefold, count). Catches looping hallucinations.
6. **Encoding artifacts:** Reject if entry contains `\x00` or 3+ consecutive backslashes.

In `append_memory()` at line ~45 (before `status = "ok"`):
```python
valid, reason = _validate_memory_entry(entry, validation_cfg)
if not valid:
    logger.warning("Memory entry rejected: %s (entry: %s)", reason, entry[:100])
    return memory_store.count(), "rejected"
```

The `"rejected"` status propagates to the tool response via `tool_append_memory` in `wintermute/tools/memory_tools.py` — no changes needed there.

**Fail-open:** Wrap the entire validation in try/except. If validation itself throws, log the error and allow the write.

### Config additions (`config.yaml.example`)

Under `memory_harvest`:
```yaml
validation:
  min_length: 10
  max_length: 2000
  reject_json: true
  reject_system_echoes: true
```

### No new database tables.

---

## Feature 3: Dreaming Quality Metrics

**Problem:** No feedback loop to detect when dreaming phases degrade memory quality. Bad schemas or noisy associations accumulate with no mechanism to detect the pattern and disable the offending phase.

**Approach:** Track dreaming output survival rate. After each cycle, record what was added. In the next cycle, check how many previous additions still exist (weren't deleted by dedup, user, or contradiction resolution). If survival rate drops below threshold over N cycles, auto-disable the phase.

### Database schema

**`wintermute/infra/database.py`**

New table in `run_migrations()` (after `dreaming_state`, around line 197):

```sql
CREATE TABLE IF NOT EXISTS dreaming_quality (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_timestamp REAL NOT NULL,
    phase_name      TEXT NOT NULL,
    entry_ids       TEXT NOT NULL,    -- JSON array of memory store IDs added
    entries_count   INTEGER NOT NULL,
    survived_count  INTEGER,          -- filled in next cycle
    checked_at      REAL              -- when survival was checked
);
```

New query functions:

1. `record_dreaming_entries(phase_name: str, entry_ids: list[str]) -> None`
2. `get_unchecked_dreaming_entries(phase_name: str) -> list[dict]` — rows where `survived_count IS NULL` and `cycle_timestamp` older than 24h
3. `update_dreaming_survival(row_id: int, survived_count: int) -> None`
4. `get_phase_survival_rate(phase_name: str, lookback_cycles: int) -> float | None` — `SUM(survived_count) / SUM(entries_count)` for last N checked cycles. Returns None if < 2 data points.

### Memory store changes

**`wintermute/infra/memory_store.py`**

Add `exists(entry_id: str) -> bool` to both `LocalVectorStore` and `QdrantStore`:
- LocalVectorStore: `SELECT 1 FROM local_vectors WHERE entry_id = ?`
- QdrantStore: `qdrant_client.retrieve(ids=[entry_id])`

### Dreaming integration

**`wintermute/workers/dreaming.py`**

In each creative phase function (`_phase_association`, `_phase_schema`, `_phase_prediction`):
- Collect returned `entry_id` from each `memory_store.add()` call
- After the loop: `database.record_dreaming_entries(phase_name, collected_ids)`

New function `_check_survival(cfg: dict) -> dict[str, bool]`:
1. For each phase ("association", "schema", "prediction"), fetch unchecked rows
2. For each row, check which entry_ids still exist in memory_store
3. Update survival count
4. Compute survival rate
5. Return `{phase_name: should_run}` — `False` if rate < threshold (default 0.3) over >= 3 cycles

Call `_check_survival` at the beginning of `run_dream_cycle()` (around line 1178). Use the returned dict to skip phases. Log clearly when a phase is auto-disabled.

### Config additions (`config.yaml.example`)

Under `memory.dreaming`:
```yaml
# ── Quality Metrics ──
quality_survival_threshold: 0.3     # Auto-disable creative phases below this survival rate
quality_min_cycles: 3               # Require N cycles before auto-disabling
quality_lookback_cycles: 5          # Compute survival over last N cycles
quality_force_enable_phases: []     # Override auto-disable for specific phases, e.g. ["association"]
```

### Risk notes
- `exists()` calls should be batched where possible (collect all IDs, query once)
- `quality_force_enable_phases` provides a manual override to re-enable phases

---

## Feature 4: Compaction Prompt Examples

**Problem:** `data/prompts/COMPACTION_PROMPT.txt` has zero few-shot examples. Small models produce much better structured summaries when given concrete examples.

**Approach:** Add 2 compact examples directly into the prompt template. No code changes needed.

### File to modify

**`data/prompts/COMPACTION_PROMPT.txt`**

Insert between the formatting instructions (line ~17) and the `{history}` placeholder:

**Example 1 (simple merge):**
```
--- example input ---
[PRIOR SUMMARY]
- User is developing a Python CLI tool called "netwatch"

[NEW MESSAGES]
USER: Can you add a --timeout flag to the CLI?
ASSISTANT: Done. Added --timeout to netwatch/cli.py with a default of 30 seconds.
USER: Perfect, make it 60 seconds default instead.
ASSISTANT: Updated the default timeout to 60 seconds.

--- example output ---
**netwatch CLI tool**
- User is developing a Python CLI tool called "netwatch"
- Added --timeout flag to netwatch/cli.py (default: 60 seconds)
```

**Example 2 (prior summary supersession):**
```
--- example input ---
[PRIOR SUMMARY]
- User prefers dark mode in all applications
- Working on a React dashboard for inventory management
- Database is PostgreSQL 15 on localhost

[NEW MESSAGES]
USER: Actually let's switch to MySQL, the hosting provider doesn't support Postgres.
ASSISTANT: I've updated the database connection in config.ts to use MySQL.

--- example output ---
**User preferences**
- Prefers dark mode in all applications

**Inventory dashboard (React)**
- Switched from PostgreSQL to MySQL (hosting constraint)
- Database connection updated in config.ts
```

These demonstrate: merging prior summary with new info, superseding old information, grouped bulleted format, concise output.

**Token impact:** ~150 tokens added per compaction call. Negligible given that compaction already sends the full conversation history.

### No code changes needed. The `{history}` placeholder remains at the end.

---

## Summary of all changes

| File | Feature(s) | Type |
|---|---|---|
| `data/prompts/COMPACTION_PROMPT.txt` | 4 | Prompt-only |
| `wintermute/infra/memory_io.py` | 2 | New validation function + gate |
| `wintermute/workers/dreaming.py` | 1, 3 | Validation functions + survival tracking |
| `wintermute/infra/database.py` | 3 | New table + 4 query functions |
| `wintermute/infra/memory_store.py` | 3 | New `exists()` method on both backends |
| `config.yaml.example` | 1, 2, 3 | New config keys documented |
