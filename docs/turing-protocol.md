# Turing Protocol

The Turing Protocol is Wintermute's post-inference validation framework. It addresses one of the core challenges of running small LLMs as autonomous agents: the model's tendency to claim it did things it didn't do, fabricate tool output, or commit to actions without executing them.

Rather than requiring a larger or more capable model, the protocol catches these failure modes programmatically and injects concise corrections so the model can self-correct in the same conversation turn. It is designed to be fast, cheap to run, and low-overhead — a secondary small model is the recommended (and default) backend.

---

## Three-Stage Design

Each validation round runs three stages:

```
Stage 1 (Detection)
  └─ A single LLM call analyses the context against ALL enabled hooks
     for the current phase. Purely programmatic hooks skip this stage.

Stage 2 (Validation)
  └─ Per-violation dispatch:
       programmatic → calls a registered Python function (no LLM, instant)
       llm          → dedicated LLM call per hook
     False positives are eliminated here.

Stage 3 (Correction)
  └─ All confirmed violations are aggregated into a single correction
     prompt injected into the conversation (main thread) or appended
     to the sub-session message list (sub-sessions).
```

Each hook fires **at most once per turn**. There is no escalation — one clear, concise correction is issued and the protocol moves on.

---

## Phases

Hooks are associated with one of three **phases** in the inference cycle:

| Phase | When it fires |
|-------|---------------|
| `post_inference` | After the LLM produces a response (text or tool calls), before delivery |
| `pre_execution` | After the LLM requests a tool call, before `execute_tool()` runs |
| `post_execution` | After `execute_tool()` returns, before the result is appended to history |

---

## Scopes

Each hook declares which execution context it applies to:

| Scope | Where |
|-------|-------|
| `main` | User-facing conversation thread only |
| `sub_session` | Background worker sub-sessions only |
| `["main", "sub_session"]` | Both |

Correction injection differs by scope:

- **main thread**: correction is injected as an async system event, which causes the LLM to produce a new response
- **sub_session**: correction is appended synchronously to the worker's message list and the inference loop continues inline

---

## Built-in Hooks

### `workflow_spawn` — Hallucinated session claim

**Phase:** `post_inference` | **Scope:** `main` | **Type:** programmatic

Detects when the model's text response claims that a background session, workflow, or task was started during this turn, but `spawn_sub_session` was never actually called.

Stage 2 check: verifies `spawn_sub_session` is absent from `tool_calls_made`. If it is present, the detection is treated as a false positive and suppressed.

Correction: instructs the model to either call `spawn_sub_session` now or acknowledge no session was started; includes the full tool schema.

---

### `phantom_tool_result` — Fabricated tool output

**Phase:** `post_inference` | **Scope:** `main` | **Type:** programmatic

Detects when the model presents specific data (file contents, search results, command output, directory listings) as if already obtained from a tool during this exchange, without any tool having been called.

Stage 2 check: if any tool was called this turn, the detection is treated as a false positive (the data may be grounded in a real tool result).

Correction: instructs the model to call the appropriate tool or clearly state it does not have the information.

---

### `empty_promise` — Action committed but not executed

**Phase:** `post_inference` | **Scope:** `main` | **Type:** programmatic

Detects when the model commits to performing an action as its final response ("I'll do X", "Let me check", "Ich werde das prüfen") without having made any tool call or spawned a sub-session.

Stage 2 checks:
- If any tool was called, treated as a false positive.
- If the response ends with a question (`?`), treated as a false positive — the model is deliberately seeking user confirmation before acting, not making an empty promise.

Correction: instructs the model to call the appropriate tool or explain why it cannot and ask how to proceed.

---

### `objective_completion` — Sub-session exit gate

**Phase:** `post_inference` | **Scope:** `sub_session` | **Type:** LLM

Fires on the final text-only response of a sub-session (when the worker produces a response with no further tool calls). Uses a dedicated LLM call to evaluate whether the response genuinely satisfies the stated objective.

If the objective is not met, a correction is injected instructing the worker to continue using available tools. The worker loop continues — this hook effectively prevents premature sub-session exit.

Configuration supports granular scope control:

```yaml
turing_protocol:
  validators:
    objective_completion:
      enabled: true
      scope: "sub_session"   # "main", "sub_session", or both
```

---

### `agenda_complete` — Agenda item completion guard

**Phase:** `pre_execution` | **Scope:** `sub_session` | **Type:** programmatic

Always-on. Fires when the model calls `agenda(action='complete')` without a substantive `reason` field (missing or fewer than 10 characters). Prevents the model from blindly marking tasks as done without evidence.

Correction: instructs the model to provide concrete, verifiable proof the task is finished before completing the item.

Not configurable via the `validators` map. Always active in sub-session scope.

---

### `tool_schema_validation` — Pre-execution argument validation

**Phase:** `pre_execution` | **Scope:** `main` + `sub_session` | **Type:** programmatic

Always-on. Fires before every tool call. Validates the LLM-supplied arguments against the tool's JSON Schema: required fields, types, enum values, minimum/maximum constraints, and unknown properties.

When a validation error is found, the tool call is blocked and a correction is injected with a bullet list of errors and the full schema for the tool. The model retries the call with corrected arguments.

Not configurable via the `validators` map. Always active in all scopes.

---

## Hook Behavior Flags

Each hook has two optional behavior flags (both default to `false`):

| Flag | Effect |
|------|--------|
| `halt_inference` | When `true`, the inference thread pauses before broadcasting the reply and waits for validation to complete. When `false`, validation runs asynchronously after the reply is already sent. |
| `kill_on_detect` | When `true`, a confirmed violation causes the response to be discarded entirely (not broadcast, not saved to DB). The correction prompt is still injected. |

All built-in hooks currently have both flags set to `false`. These are primarily useful for custom hooks (see below).

---

## Configuration

Basic configuration in `config.yaml`:

```yaml
turing_protocol:
  backends: ["local_small"]         # small/fast model recommended; defaults to base if omitted
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
    objective_completion:
      enabled: true
      scope: "sub_session"
```

**Disabling entirely:** Set `backends: []`.

**Disabling individual hooks:** Set the validator to `false`:

```yaml
turing_protocol:
  validators:
    empty_promise: false            # disable this hook only
```

**Granular scope override:**

```yaml
turing_protocol:
  validators:
    workflow_spawn:
      enabled: true
      scope: ["main", "sub_session"]   # extend to sub-sessions too
```

The `turing_protocol` role in the `llm` role mapping also controls the backend:

```yaml
llm:
  turing_protocol: ["local_small", "local_large"]   # failover list
```

---

## Custom Hooks

Custom hooks are loaded from `data/TURING_PROTOCOL_HOOKS.txt` (a JSON array). File entries override built-ins by hook name. Missing or malformed — built-in defaults are used.

Example custom hook:

```json
[
  {
    "name": "my_custom_check",
    "phase": "post_inference",
    "scope": ["main"],
    "detection_prompt": "- **my_custom_check**: The assistant claims X without evidence Y.",
    "validator_type": "programmatic",
    "validator_fn_name": "validate_workflow_spawn",
    "correction_template": "[TURING CORRECTION] Issue: {reason}\n\nPlease correct this.",
    "halt_inference": false,
    "kill_on_detect": false
  }
]
```

Hook fields:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Unique hook identifier |
| `phase` | yes | `post_inference`, `pre_execution`, or `post_execution` |
| `scope` | yes | `"main"`, `"sub_session"`, or `["main", "sub_session"]` |
| `detection_prompt` | yes (for Stage 1 hooks) | Bullet text added to the universal Stage 1 detection prompt |
| `validator_type` | yes | `"programmatic"` or `"llm"` |
| `validator_fn_name` | if programmatic | Name of a registered Python validator function |
| `correction_template` | yes | Correction text; supports `{reason}`, `{tool_schema}`, `{tool_name}`, `{objective}` placeholders |
| `halt_inference` | no | Block inference until validation completes (default: `false`) |
| `kill_on_detect` | no | Discard the response on confirmed violation (default: `false`) |

Custom `"llm"` type hooks currently require a Python implementation for the dedicated LLM handler. `"programmatic"` hooks can reuse any registered validator function.

---

## Logging

All validation rounds are written to the `interaction_log` table in SQLite and visible via the debug panel at `/debug` → (interaction log endpoint at `/api/debug/interaction-log`). Each entry records the input context, raw detection output, and final status (`ok` or `violation_detected`) for stages 1, 2, and 3 separately.

---

## Why It Matters for Small LLMs

Autonomous tool-use agents fail in predictable ways when the underlying model is small:

1. **Hallucinated actions** — the model reports doing something it didn't (most common)
2. **Phantom tool results** — the model invents tool output rather than calling the tool
3. **Empty promises** — the model acknowledges a task but forgets to call the tool
4. **Premature completion** — a sub-session declares success before the objective is met
5. **Schema drift** — the model produces malformed tool arguments that silently fail

The Turing Protocol catches all five failure modes programmatically and corrects them within the same inference loop. A separate small validator model (recommended: 3B–7B, fast) handles Stage 1 detection, making the overhead low even on modest hardware. The result is usable, reliable autonomous behaviour from models that would otherwise be too unreliable for agentic tasks.
