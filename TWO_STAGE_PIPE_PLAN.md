# Two-Stage Inference Pipe for Complex Tool Calls

**Target tools:** `set_reminder`, `spawn_sub_session`
**Goal:** Allow weak/small main LLMs to invoke complex tools by writing a plain-English
description, with a dedicated translator LLM converting that description to the full
structured schema.

---

## Problem Statement

`set_reminder` has 11 properties with conditional interdependencies (e.g. `schedule_type`
determines which of `at`, `day_of_week`, `day_of_month`, `interval_seconds` are required).
`spawn_sub_session` has complex DAG semantics (`depends_on`, `depends_on_previous`,
`not_before`). Small 8B models frequently produce partial or malformed arguments.

The `tool_schema_validation` Turing Protocol hook (already implemented) catches and
corrects these errors with a one-shot retry. The pipe described here is a complementary
approach: eliminate the schema complexity from the main LLM entirely by presenting it a
simplified single-field schema, then using a second LLM call (the "translator") to
expand the natural-language description into fully-structured arguments.

---

## High-Level Architecture

```
Main LLM
  └─ calls set_reminder(description="daily 9am check email")
       │
       ▼
  [NL Translation Layer]  ─── translator LLM call ───► structured args dict
       │                                                 {"message": ...,
       │                                                  "schedule_type": "daily",
       │                                                  "at": "09:00"}
       ▼
  execute_tool("set_reminder", structured_args)
       │
       ▼
  Tool result injected into history
  (includes translation summary so main LLM has full context)
```

The main LLM **only sees the simplified schema**. The real schema is held entirely
inside the translator prompt. The translator is purpose-built for one narrow task and
can be heavily few-shot prompted.

---

## Simplified Schemas

Two simplified schemas replace the full ones when NL translation is enabled. They are
defined as an alternative export from `tools.py` and selected by `get_tool_schemas()`
based on config.

### `set_reminder` simplified

```json
{
  "type": "function",
  "function": {
    "name": "set_reminder",
    "description": "Schedule a reminder or recurring task. Describe what you want in natural language — include when, how often, and what action to take or message to send.",
    "parameters": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "Full natural-language description of the reminder. Examples: 'daily at 09:00 to check email', 'every Monday at 14:00 to review pulse', 'once on 2026-03-15 at 10:00 — dentist appointment', 'every 3600 seconds between 08:00-18:00 to check server status'. Include the ai_prompt if an autonomous action should run instead of just a notification."
        }
      },
      "required": ["description"]
    }
  }
}
```

### `spawn_sub_session` simplified

```json
{
  "type": "function",
  "function": {
    "name": "spawn_sub_session",
    "description": "Spawn an autonomous background worker. Describe the task in natural language — the worker will have no conversation access, so be specific about what it should do and what the success condition is.",
    "parameters": {
      "type": "object",
      "properties": {
        "description": {
          "type": "string",
          "description": "Full natural-language description of the task. Include: what to do, what to produce, any ordering constraints (e.g. 'after the previous session finishes'), timeout if relevant."
        }
      },
      "required": ["description"]
    }
  }
}
```

---

## New Module: `wintermute/nl_translator.py`

```python
"""
Natural-language to structured-args translator for complex tool calls.

Called by the tool execution loops in llm_thread.py and sub_session.py
when the main LLM invokes a tool using the simplified NL schema.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from wintermute.llm_thread import BackendPool

from wintermute import prompt_loader

logger = logging.getLogger(__name__)

# Tools that have NL variants and require translation.
NL_TOOLS: frozenset[str] = frozenset({"set_reminder", "spawn_sub_session"})


async def translate_nl_tool_call(
    pool: "BackendPool",
    tool_name: str,
    description: str,
    *,
    thread_id: str = "unknown",
) -> dict | None:
    """Translate a natural-language description into structured tool arguments.

    Parameters
    ----------
    pool : BackendPool
        Backend pool to use for the translator inference call.
    tool_name : str
        Name of the real tool to translate for (e.g. "set_reminder").
    description : str
        Natural-language description provided by the main LLM.
    thread_id : str
        For logging only.

    Returns
    -------
    dict
        Structured argument dict ready to pass to ``execute_tool()``.
    None
        If translation failed unrecoverably (caller should return an error
        tool result so the main LLM can request clarification).
    """
    prompt_key = f"NL_TRANSLATOR_{tool_name.upper()}.txt"
    try:
        system_prompt = prompt_loader.load(prompt_key)
    except FileNotFoundError:
        logger.error("nl_translator: no prompt file %r — cannot translate", prompt_key)
        return None

    try:
        response = await pool.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("nl_translator: inference call failed for tool %r", tool_name)
        return None

    # Strip markdown fences.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "nl_translator: unparseable response for %r: %r", tool_name, raw[:300]
        )
        return None

    if not isinstance(result, dict):
        logger.warning("nl_translator: expected dict, got %s", type(result).__name__)
        return None

    # Ambiguity signal: translator returns {"error": ..., "clarification_needed": ...}
    if "error" in result:
        logger.info(
            "nl_translator: translator flagged ambiguity for %r: %s",
            tool_name, result.get("clarification_needed", "?"),
        )
        # Return the error dict; caller converts it to a tool result error message.
        return result

    logger.info(
        "nl_translator: translated %r successfully (thread=%s): %s",
        tool_name, thread_id, list(result.keys()),
    )
    return result


def is_nl_tool_call(tool_name: str, tool_args: dict) -> bool:
    """Return True if this tool call used the simplified NL schema.

    Detection criterion: the tool is in NL_TOOLS and the args dict contains
    exactly a "description" key (and nothing else, since the simplified
    schema has only that one property).
    """
    return tool_name in NL_TOOLS and "description" in tool_args and len(tool_args) == 1
```

---

## Prompt Template Files

Two new files in `data/prompts/`:

### `data/prompts/NL_TRANSLATOR_SET_REMINDER.txt`

```
You are a tool-call translator. You receive a natural-language description of a
reminder and output ONLY a JSON object containing the arguments for the
set_reminder tool.

Full tool schema:
{set_reminder_schema}

Rules:
- Output ONLY valid JSON. No prose, no markdown explanation.
- If the description is ambiguous and you cannot infer a required field,
  output: {"error": "missing_info", "clarification_needed": "<specific question>"}
- "message" must always be a concise reminder text (the human-readable notification).
- "schedule_type" must be one of: once, daily, weekly, monthly, interval.
- "at" is required for all schedule types except interval. Use HH:MM for recurring,
  ISO-8601 for once.
- "day_of_week" is required for weekly. Use: mon tue wed thu fri sat sun.
- "day_of_month" is required for monthly. Integer 1–31.
- "interval_seconds" is required for interval. Convert minutes/hours to seconds.
- "ai_prompt" should be set whenever the description says to "do X" or "run Y" rather
  than just send a notification.
- Do not include fields not needed for the schedule type.

Examples:

User: daily at 09:00 to check email
Output: {"message": "Check email", "schedule_type": "daily", "at": "09:00"}

User: once on 2026-03-15 at 10:00 — dentist appointment
Output: {"message": "Dentist appointment", "schedule_type": "once", "at": "2026-03-15T10:00:00"}

User: every Monday at 14:00 to review pulse
Output: {"message": "Review pulse", "schedule_type": "weekly", "at": "14:00", "day_of_week": "mon"}

User: 1st of every month at 09:00 send monthly report
Output: {"message": "Send monthly report", "schedule_type": "monthly", "at": "09:00", "day_of_month": 1}

User: every hour between 08:00 and 18:00 check server status and alert if down
Output: {"message": "Check server status", "schedule_type": "interval", "interval_seconds": 3600, "window_start": "08:00", "window_end": "18:00", "ai_prompt": "Check server status using execute_shell. If any service is down, append a critical alert to MEMORIES.txt."}

User: remind me daily at noon to drink water
Output: {"message": "Drink water", "schedule_type": "daily", "at": "12:00"}

User: remind me to call John
Output: {"error": "missing_info", "clarification_needed": "When should the reminder fire? Please specify a date/time or recurrence pattern."}
```

The template string `{set_reminder_schema}` is filled at load time by `prompt_loader` with
the JSON of the real `set_reminder` schema from `TOOL_SCHEMAS`. See
**Schema Embedding** below.

### `data/prompts/NL_TRANSLATOR_SPAWN_SUB_SESSION.txt`

```
You are a tool-call translator. You receive a natural-language description of a
background task and output ONLY a JSON object containing the arguments for the
spawn_sub_session tool.

Full tool schema:
{spawn_sub_session_schema}

Rules:
- Output ONLY valid JSON. No prose, no markdown explanation.
- "objective" is always required. Write it as a complete, standalone task instruction
  that a worker with no conversation access can execute.
- Omit optional fields unless the description explicitly requires them.
- "depends_on_previous" should be true if the description says "after the previous
  task" or "once the last session finishes".
- "system_prompt_mode" defaults to "minimal". Use "full" only if the task explicitly
  needs long-term memories or skills (e.g. "using your knowledge of the project").
- "timeout" in seconds. Default 300 if not specified.
- Do not invent constraints not mentioned in the description.

Examples:

User: search for the latest news about AI regulation and summarize in a file
Output: {"objective": "Search the web for latest news about AI regulation. Write a concise summary (300–500 words) to data/ai_regulation_summary.md."}

User: research the Python asyncio docs and write a cheatsheet to data/asyncio_cheatsheet.md, then after that's done, review the cheatsheet and add three practical examples
Output: [{"objective": "Research the Python asyncio documentation. Write a concise cheatsheet to data/asyncio_cheatsheet.md covering the most important patterns.", "depends_on_previous": false}, {"objective": "Read data/asyncio_cheatsheet.md. Add three practical, runnable code examples to the file.", "depends_on_previous": true}]

User: check if the wintermute service is running and restart it if not, timeout 60s
Output: {"objective": "Run 'systemctl --user status wintermute'. If it is not running, run 'systemctl --user restart wintermute' and verify it started.", "timeout": 60}
```

> **Note:** The multi-session example (`[{...}, {...}]`) is a JSON array. When the
> translator returns an array, the integration layer calls `spawn_sub_session` once
> per element in order. See **Integration** below.

---

## Schema Embedding in Prompt Templates

`prompt_loader.load()` currently supports `{key}` substitution. The NL translator
prompts need the real tool schema embedded. Two approaches:

**Option A (preferred): Static embedding at file write time.**
The prompt files contain the schema verbatim. Whenever `TOOL_SCHEMAS` changes, the
files need regenerating. A helper script `scripts/regen_nl_prompts.py` does this.

**Option B: Dynamic substitution via prompt_loader.**
`prompt_loader.load("NL_TRANSLATOR_SET_REMINDER.txt", set_reminder_schema=schema_json)`.
This requires `prompt_loader` to not fail on unknown substitution keys. Currently
it passes kwargs directly to `str.format_map()`, so this already works — but the
prompt file must escape all literal curly braces.

**Recommendation: Option A.** Prompt files are developer-maintained artifacts;
embedding the schema statically makes them self-contained and easier to inspect/edit.
The `regen_nl_prompts.py` script is a one-liner.

---

## Config Additions

In `config.yaml` (and the config dataclass):

```yaml
nl_translation:
  enabled: false          # master switch; default false (opt-in)
  tools:                  # which tools to translate; default: both
    - set_reminder
    - spawn_sub_session
  backend_role: nl_translation  # BackendPool role; falls back to sub_sessions if absent
```

Config dataclass additions (wherever LLMConfig or equivalent is defined):

```python
@dataclass
class NLTranslationConfig:
    enabled: bool = False
    tools: list[str] = field(default_factory=lambda: ["set_reminder", "spawn_sub_session"])
    backend_role: str = "nl_translation"
```

---

## BackendPool Routing

The translator uses the same `BackendPool` mechanism as Turing Protocol. A new role
`"nl_translation"` is added. If no provider declares `role: nl_translation`, the pool
falls back to `role: sub_sessions`. The config plumbing:

In `config.yaml` providers section, a provider can declare:
```yaml
  - name: strong-model
    roles: [nl_translation]
    ...
```

The `BackendPool` passed to `translate_nl_tool_call()` is the existing
`self._turing_protocol_pool` (reuse it; same lightweight inference pattern) OR a
dedicated `self._nl_translation_pool` created by `LLMThread.__init__`. The latter is
cleaner — create it once and pass it down.

---

## Changes to `tools.py`

### New export: `NL_TOOL_SCHEMAS`

```python
NL_TOOL_SCHEMAS: list[dict] = [
    _fn(
        "set_reminder",
        "Schedule a reminder or recurring task. Describe what you want in natural "
        "language — include when, how often, and what action to take.",
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Natural-language description. Examples: "
                        "'daily at 09:00 to check email', "
                        "'every Monday at 14:00 to review pulse', "
                        "'once on 2026-03-15 at 10:00 — dentist', "
                        "'every 3600 seconds 08:00-18:00 to check servers'."
                    ),
                }
            },
            "required": ["description"],
        },
    ),
    _fn(
        "spawn_sub_session",
        "Spawn an autonomous background worker. Describe the task in natural language.",
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What the worker should do. Include success criteria, "
                        "ordering constraints, and timeout if relevant."
                    ),
                }
            },
            "required": ["description"],
        },
    ),
]

# Map from tool name to its NL-simplified schema entry (for O(1) lookup).
_NL_SCHEMA_MAP: dict[str, dict] = {
    s["function"]["name"]: s for s in NL_TOOL_SCHEMAS
}
```

### Updated `get_tool_schemas()`

```python
def get_tool_schemas(
    categories: set[str] | None = None,
    *,
    nl_tools: set[str] | None = None,
) -> list[dict]:
    """Return tool schemas filtered by category.

    Parameters
    ----------
    categories : set[str] | None
        If None, return all schemas. Otherwise filter by category.
    nl_tools : set[str] | None
        Tool names for which the NL-simplified schema should be substituted.
        When a tool name is in this set, its full schema is replaced by the
        corresponding entry from NL_TOOL_SCHEMAS. Pass None (default) to
        always use full schemas.
    """
    base = TOOL_SCHEMAS if categories is None else [
        s for s in TOOL_SCHEMAS
        if TOOL_CATEGORIES.get(s["function"]["name"]) in categories
    ]
    if not nl_tools:
        return base
    return [
        _NL_SCHEMA_MAP[s["function"]["name"]]
        if s["function"]["name"] in nl_tools and s["function"]["name"] in _NL_SCHEMA_MAP
        else s
        for s in base
    ]
```

---

## Integration Points

### `llm_thread.py` — `_run_inference_loop()`

The tool-call loop already has `pre_execution` Turing hooks. NL translation is inserted
**before** the Turing Protocol check (since we want to validate the *translated* args,
not the raw description):

```python
# In the tool-call for loop, after:
#   inputs = json.loads(tc.function.arguments) / inputs = {}

# -- NL Translation (before Turing Protocol pre_execution) --
if self._nl_translation_enabled and nl_translator.is_nl_tool_call(tc.function.name, inputs):
    translated = await nl_translator.translate_nl_tool_call(
        self._nl_translation_pool,
        tc.function.name,
        inputs["description"],
        thread_id=thread_id,
    )
    if translated is None:
        full_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": (
                f"[TRANSLATION ERROR] Failed to translate your natural-language "
                f"description for {tc.function.name!r}. Please try again with more "
                f"detail, or call the tool with explicit arguments."
            ),
        })
        continue
    if "error" in translated:
        # Ambiguity: relay clarification request to main LLM.
        full_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": (
                f"[TRANSLATION AMBIGUITY] {translated.get('clarification_needed', 'Please provide more detail.')}"
            ),
        })
        continue
    inputs = translated
    # Fall through to Turing Protocol pre_execution and execute_tool.

# -- Turing Protocol: pre_execution phase (unchanged) --
if tp_enabled:
    pre_result = await self._run_phase_check(...)
    ...
```

**`_run_inference_loop` signature change:**

```python
async def _run_inference_loop(
    self,
    system_prompt: str,
    messages: list,
    thread_id: str,
    disable_tools: bool = False,
    nl_tool_mode: bool = False,     # NEW: pass NL schemas instead of full schemas
) -> LLMReply:
```

When `nl_tool_mode=True`, `tools` is set using the `nl_tools` kwarg of `get_tool_schemas()`.

The `nl_tool_mode` flag is set by the caller based on `self._cfg.nl_translation.enabled`
and whether the tool is in `self._cfg.nl_translation.tools`.

### `sub_session.py` — `TaskNode._run()`

Same pattern, applied in the sub-session tool loop at lines ~1086–1113 (the equivalent
`for tc in choice.message.tool_calls` block). The sub-session runner also uses a
`_run_tp_phase()` call; NL translation is inserted before it in the same way.

The sub-session `WorkerState` already carries `pool`; the NL translation pool is passed
in at construction time (analogous to how `turing_protocol_pool` is passed).

### Multi-session Spawn

When the translator returns a JSON **array** for `spawn_sub_session`, the integration
layer iterates over the list and calls `execute_tool("spawn_sub_session", item)` once
per entry, collecting the session IDs. All results are concatenated into a single tool
result message. This enables the translator to decompose compound tasks automatically.

```python
if isinstance(translated, list):
    results = []
    for item in translated:
        r = tool_module.execute_tool("spawn_sub_session", item, ...)
        results.append(r)
        tool_calls_made.append("spawn_sub_session")
    # Inject combined tool result.
    full_messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": "\n---\n".join(results),
    })
    continue
```

---

## Tool Result Transparency

After translation the main LLM receives the tool result but **does not see** the
translated args directly. To preserve the reasoning loop, prepend a translation summary
to the tool result:

```
[Translated to: message="Check email", schedule_type="daily", at="09:00"]
Reminder set: job_id=reminder_a1b2c3d4, fires daily at 09:00.
```

This is done in the integration layer after calling `execute_tool()`:

```python
translation_summary = (
    f"[Translated to: {', '.join(f'{k}={v!r}' for k, v in inputs.items())}]\n"
)
result = translation_summary + tool_module.execute_tool(tc.function.name, inputs, ...)
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Translator LLM call fails (network/timeout) | Return error tool result; main LLM retries or asks user |
| Translator returns unparseable JSON | Return error tool result with raw output (truncated) |
| Translator returns `{"error": "missing_info", ...}` | Return clarification request as tool result |
| Translator returns array with one bad entry | Skip bad entry, execute good ones, include error notice in result |
| `prompt_loader` can't find NL_TRANSLATOR_*.txt | Log error, disable NL translation for this tool in this session |
| Translated args fail Turing `tool_schema_validation` | Blocked by existing hook; correction injected; LLM retries |

---

## Implementation Steps (ordered)

1. **Create `wintermute/nl_translator.py`** — exactly as specified above.

2. **Add `NL_TOOL_SCHEMAS`, `_NL_SCHEMA_MAP`, updated `get_tool_schemas()`** to `tools.py`.

3. **Create translator prompt files:**
   - `data/prompts/NL_TRANSLATOR_SET_REMINDER.txt` (static, schema embedded)
   - `data/prompts/NL_TRANSLATOR_SPAWN_SUB_SESSION.txt` (static, schema embedded)
   - Run `scripts/regen_nl_prompts.py` (one-liner) to regenerate if schemas change.

4. **Add `NLTranslationConfig` dataclass** and parse `nl_translation:` section in
   config loading (wherever `config.yaml` is parsed into dataclasses).

5. **Add `_nl_translation_pool`** to `LLMThread.__init__` — built with role
   `"nl_translation"` falling back to `"sub_sessions"`.

6. **Update `LLMThread._run_inference_loop()`:**
   - Add `nl_tool_mode: bool = False` parameter.
   - When `nl_tool_mode=True`, use `get_tool_schemas(nl_tools=...)`.
   - Insert NL translation block after `json.loads(tc.function.arguments)`.
   - Prepend translation summary to tool result.

7. **Update the callers of `_run_inference_loop()`** in `LLMThread` to pass
   `nl_tool_mode=self._cfg.nl_translation.enabled`.

8. **Update `sub_session.py` `TaskNode._run()`:**
   - Pass `nl_translation_pool` into `WorkerState` or `TaskNode`.
   - Insert the same NL translation block in the sub-session tool loop.
   - Respect `nl_tools` set when building the sub-session tool schema list
     (already goes through `get_tool_schemas(categories=...)`; add `nl_tools=` kwarg).

9. **Handle multi-session spawn array** in both integration points.

10. **Smoke test manually:**
    - Main LLM calls `set_reminder(description="daily at 09:00 check email")`.
    - Verify translator fires, structured args produced, tool executes, result returned.
    - Verify `tool_schema_validation` hook validates the *translated* args (not the
      raw description).
    - Verify ambiguity path: `set_reminder(description="remind me about the meeting")`
      returns a clarification request.

---

## Scope Boundaries

**Not covered by this plan:**
- Making the translator aware of the current date/time for relative time expressions
  like "tomorrow at 3pm". The main LLM should resolve relative times before calling
  the tool (it has the current date in its system prompt via `PromptAssembler`).
- Translator caching / deduplication (not needed for latency-sensitive personal assistant).
- Streaming the translator response (unnecessary — translator outputs short JSON).
- Disabling the `tool_schema_validation` hook for translated calls (the hook provides
  a useful safety net even when translation is enabled).

---

## Open Questions (decide before implementing)

1. **Which BackendPool instance for translation?** Reuse `_turing_protocol_pool` (saves
   code) or create a dedicated `_nl_translation_pool` (allows different model/provider
   per role). Recommendation: dedicated pool with fallback to sub_sessions role.

2. **Should `prompt_loader` validate NL translator prompt files at startup?** Currently
   `prompt_loader` only validates files listed in its required set. Add
   `NL_TRANSLATOR_SET_REMINDER.txt` and `NL_TRANSLATOR_SPAWN_SUB_SESSION.txt` to the
   required set only when `nl_translation.enabled: true` — to avoid startup failure for
   users who don't enable the feature.

3. **Multi-session array from translator:** Should this be supported for `set_reminder`
   too (schedule multiple reminders from one description)? Probably yes, same logic.
