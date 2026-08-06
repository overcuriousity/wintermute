# Fresh Redeploy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make wintermute safe to redeploy fresh and run in production: a clean lint baseline, blocking CI with a real test floor over the highest-risk pure logic, no dead dependencies, no nightly skill-documentation destruction, and memory that works without a configured embeddings endpoint.

**Architecture:** Six tasks landing in order. Mechanical changes (lint, dead deps) land before behavioral ones so every later diff stays reviewable. Tests target only synchronous, IO-free functions — no mocks, no fixtures beyond `monkeypatch`, no `pytest-asyncio`. The local embedding fallback integrates at exactly one choke point, `infra/llm_utils.py:embed()`, which every one of the 15 `_embed()` call sites already routes through.

**Tech Stack:** Python 3.12, `uv`, `pytest` 9.x, `ruff` 0.9.x, GitHub Actions, `onnxruntime` + `tokenizers` (new optional extra), SQLite.

**Source spec:** `docs/superpowers/specs/2026-08-06-fresh-redeploy-hardening-design.md`

## Global Constraints

- Python floor is `requires-python = ">=3.12"`. Do not raise or lower it.
- Dependency budget is explicit in `ROADMAP.md`: keep direct dependencies near 17. This plan removes two and adds zero required ones; `onnxruntime` and `tokenizers` go in an **optional** extra only.
- `ruff` config lives in `pyproject.toml`: `target-version = "py312"`, `line-length = 100`, `select = ["E", "W", "F", "I", "UP", "B", "C4", "SIM"]`, `ignore = ["E501"]`, `quote-style = "double"`.
- Never run `ruff` with `--unsafe-fixes`.
- `[tool.pytest.ini_options] testpaths = ["tests"]` — all tests go in `tests/`.
- No telemetry, no network calls in tests.
- Commit messages use conventional-commit prefixes (`feat:`, `fix:`, `chore:`, `test:`, `docs:`) — the repo runs release-please, which parses them.
- The project has no test suite habit yet. Every test in this plan must pass with a bare `uv run pytest`, with no config file, no service, and no network.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | ruff ignores with rationale; drop `asyncpg`/`aiosqlite`; add `local-embeddings` extra | 1, 4, 6 |
| `.github/workflows/ci.yml` | Blocking lint + test on push and PR | 2 |
| `tests/test_convergence_validators.py` | The 8 stage-2 validators | 2 |
| `tests/test_redaction.py` | `redact_credentials`, `extract_config_secrets` | 2 |
| `tests/test_tool_call_rescue.py` | All 6 rescue parsers + `rescue_tool_calls` | 3 |
| `tests/test_nl_translator.py` | `is_nl_tool_call`, `_fix_unescaped_control_chars`, `_task_list_fastpath` | 3 |
| `CLAUDE.md`, `ROADMAP.md` | Correct stale claims | 4 |
| `wintermute/workers/dreaming.py` | Remove skill condensation | 5 |
| `wintermute/infra/prompt_loader.py` | Drop condensation template from `REQUIRED_FILES` | 5 |
| `wintermute/infra/local_embeddings.py` | **New.** Lazy ONNX MiniLM provider; the only file that knows about ONNX | 6 |
| `wintermute/infra/llm_utils.py` | Route to local provider when no endpoint configured | 6 |
| `wintermute/infra/memory_store.py`, `skill_store.py` | Relax the hard fail; persist provider + dimension; guard mismatches | 6 |
| `config.yaml.example`, `docs/installation.md` | Document the fallback | 6 |

---

### Task 1: Lint baseline

Purely mechanical. Land and merge this alone — it touches most of the tree, and mixing it with behavior changes makes both unreviewable.

**Files:**
- Modify: most of `wintermute/` (formatting and auto-fixes only)
- Modify: `pyproject.toml` (`[tool.ruff.lint] ignore`)

**Interfaces:**
- Consumes: nothing.
- Produces: a tree where `ruff check .` and `ruff format --check .` both exit 0. Task 2's CI workflow depends on this.

- [ ] **Step 1: Record the starting state**

```bash
cd /home/user01/Projekte/wintermute
uvx ruff check . 2>&1 | tail -3
uvx ruff format --check . 2>&1 | tail -1
```

Expected, as audited on 2026-08-06: `Found 581 errors.` / `513 fixable` and `47 files would be reformatted, 31 files already formatted`. If the numbers differ substantially, the tree has moved since the audit — proceed anyway, but note the new numbers in the commit message.

- [ ] **Step 2: Verify the test suite passes before touching anything**

```bash
uv run pytest -q
```

Expected: PASS. This is the baseline the reformat must not break.

- [ ] **Step 3: Apply safe auto-fixes**

```bash
uv run ruff check --fix .
uv run ruff format .
```

Do not add `--unsafe-fixes`. It rewrites semantics (for example, turning `except Exception: pass` into `contextlib.suppress`, which changes traceback behavior).

- [ ] **Step 4: Confirm the auto-fix changed no behavior**

```bash
uv run pytest -q
uv run python -c "import wintermute.main; import wintermute.core.llm_thread; import wintermute.workers.dreaming; print('imports ok')"
```

Expected: tests PASS, `imports ok`. The import check matters because `ruff check --fix` reorders imports (rule `I`), which can expose a circular import that was previously masked by ordering.

- [ ] **Step 5: Commit the mechanical pass separately**

```bash
git add -A
git commit -m "style: apply ruff auto-fixes and formatter

Mechanical only: ruff check --fix (no --unsafe-fixes) plus ruff format.
No behavior change; test suite and module imports verified before and after."
```

Committing before the manual triage keeps the reviewable-by-machine part isolated from the part that needs judgment.

- [ ] **Step 6: List what auto-fix could not resolve**

```bash
uv run ruff check . --statistics
```

This prints a count per rule code. Expect roughly 68 remaining errors.

- [ ] **Step 7: Triage the remainder**

Apply this decision rule per rule code:

- Fewer than ~5 occurrences, and the fix is local and obvious: fix the code.
- Many occurrences arising from a deliberate codebase-wide pattern: add the rule to `[tool.ruff.lint] ignore` in `pyproject.toml` with a comment stating why.

The known case is `SIM105` ("use `contextlib.suppress`"), which fires against the codebase's 311 deliberate broad `except Exception` handlers. Add it as:

```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "C4", "SIM"]
ignore = [
    "E501",
    # SIM105 wants contextlib.suppress in place of try/except/pass. The
    # codebase uses ~311 deliberate broad handlers to keep the agent loop
    # alive through tool and backend failures; suppress() hides the
    # exception object and makes those sites harder to instrument later.
    "SIM105",
]
```

Add further ignores in the same style only where the same reasoning applies. Prefer one documented ignore over scattered `# noqa` comments.

- [ ] **Step 8: Verify the tree is clean**

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Expected: `All checks passed!`, `78 files already formatted` (or equivalent — zero files needing reformat), tests PASS. All three must be clean; Task 2 makes them blocking.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "style: resolve remaining ruff findings and document rule ignores"
```

---

### Task 2: CI workflow, validator tests, redaction tests

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_convergence_validators.py`
- Create: `tests/test_redaction.py`

**Interfaces:**
- Consumes: the lint-clean tree from Task 1.
- Produces: a green blocking CI job named `ci`. Task 3 adds more test files to the same job with no workflow change.

Functions under test, all in `wintermute/core/convergence_protocol.py`, all synchronous with signature `(context: dict, detection_result: dict) -> bool`. `True` means the violation is **confirmed**; `False` means Stage 1's detection was a **false positive**. The false-positive cases are the important ones: a validator that over-fires blocks legitimate tool calls.

- [ ] **Step 1: Write the failing validator tests**

Create `tests/test_convergence_validators.py`:

```python
"""Tests for the Convergence Protocol stage-2 programmatic validators.

Each validator returns True when the violation is confirmed and False when
Stage 1's detection was a false positive.  The false-positive cases matter
most: an over-firing validator blocks legitimate agent actions.
"""

import pytest

from wintermute.core import convergence_protocol as cp
from wintermute.core.tool_schemas import TOOL_CATEGORIES


# ── workflow_spawn ──────────────────────────────────────────────────

def test_workflow_spawn_confirmed_when_not_called():
    assert cp.validate_workflow_spawn({"tool_calls_made": []}, {}) is True


def test_workflow_spawn_false_positive_called_this_turn():
    ctx = {"tool_calls_made": ["worker_delegation"]}
    assert cp.validate_workflow_spawn(ctx, {}) is False


def test_workflow_spawn_false_positive_called_prior_turn():
    ctx = {"tool_calls_made": [], "prior_tool_calls_made": ["worker_delegation"]}
    assert cp.validate_workflow_spawn(ctx, {}) is False


# ── phantom_tool_result ─────────────────────────────────────────────

def test_phantom_confirmed_when_claimed_tool_never_called():
    ctx = {"tool_calls_made": []}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is True


def test_phantom_false_positive_claimed_tool_was_called():
    ctx = {"tool_calls_made": ["read_file"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_execute_shell_substitutes():
    """execute_shell can read files, list dirs, fetch URLs — it covers for any
    file/data claim."""
    ctx = {"tool_calls_made": ["execute_shell"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_claimed_tool_in_prior_turn():
    ctx = {"tool_calls_made": [], "prior_tool_calls_made": ["read_file"]}
    assert cp.validate_phantom_tool_result(ctx, {"tool": "read_file"}) is False


def test_phantom_false_positive_no_tool_named_but_tools_ran():
    ctx = {"tool_calls_made": ["append_memory"]}
    assert cp.validate_phantom_tool_result(ctx, {}) is False


# ── empty_promise ───────────────────────────────────────────────────

def test_empty_promise_confirmed_when_no_tools_called():
    ctx = {
        "tool_calls_made": [],
        "assistant_response": "I will now read the config file and report back.",
    }
    assert cp.validate_empty_promise(ctx, {}) is True


def test_empty_promise_false_positive_when_tools_called():
    ctx = {"tool_calls_made": ["read_file"], "assistant_response": "Reading it now."}
    assert cp.validate_empty_promise(ctx, {}) is False


def test_empty_promise_false_positive_pure_confirmation_question():
    """A response that is only a question is waiting for approval, not
    promising anything."""
    ctx = {"tool_calls_made": [], "assistant_response": "Should I restart the service?"}
    assert cp.validate_empty_promise(ctx, {}) is False


def test_empty_promise_confirmed_despite_trailing_question():
    """A generic closer appended to a real commitment does not excuse it.
    The line before the question is substantive (>20 chars)."""
    ctx = {
        "tool_calls_made": [],
        "assistant_response": (
            "I will now read the config file and report the results back to you.\n"
            "Anything else?"
        ),
    }
    assert cp.validate_empty_promise(ctx, {}) is True


# ── task_complete ───────────────────────────────────────────────────

def test_task_complete_confirmed_on_short_reason():
    ctx = {"tool_name": "task", "tool_args": {"action": "complete", "reason": "ok"}}
    assert cp.validate_task_complete(ctx, {}) is True
    assert "insufficient reason" in ctx["_convergence_hook_reason"]


def test_task_complete_passes_with_substantive_reason():
    ctx = {
        "tool_name": "task",
        "tool_args": {"action": "complete", "reason": "verified all tests pass"},
    }
    assert cp.validate_task_complete(ctx, {}) is False


def test_task_complete_ignores_other_actions():
    ctx = {"tool_name": "task", "tool_args": {"action": "list"}}
    assert cp.validate_task_complete(ctx, {}) is False


def test_task_complete_ignores_other_tools():
    ctx = {"tool_name": "read_file", "tool_args": {"action": "complete"}}
    assert cp.validate_task_complete(ctx, {}) is False


# ── repetition_loop ─────────────────────────────────────────────────

def test_repetition_loop_confirmed_on_near_identical_response():
    text = "I checked the scheduler and it reports three pending jobs right now."
    ctx = {"assistant_response": text, "recent_assistant_messages": [text]}
    assert cp.validate_repetition_loop(ctx, {}) is True
    assert "similar" in ctx["_convergence_hook_reason"]


def test_repetition_loop_passes_on_different_response():
    ctx = {
        "assistant_response": "I checked the scheduler and it reports three pending jobs.",
        "recent_assistant_messages": [
            "The memory store contains four hundred entries after consolidation."
        ],
    }
    assert cp.validate_repetition_loop(ctx, {}) is False


def test_repetition_loop_ignores_short_responses():
    ctx = {"assistant_response": "ok", "recent_assistant_messages": ["ok"]}
    assert cp.validate_repetition_loop(ctx, {}) is False


def test_repetition_loop_passes_with_no_history():
    ctx = {
        "assistant_response": "I checked the scheduler and it reports three pending jobs.",
        "recent_assistant_messages": [],
    }
    assert cp.validate_repetition_loop(ctx, {}) is False


# ── inline_tool_limit ───────────────────────────────────────────────

def _exec_tools(n: int) -> list[str]:
    """Return *n* tool names whose category is execution or research."""
    names = [t for t, c in sorted(TOOL_CATEGORIES.items()) if c in ("execution", "research")]
    assert len(names) >= n, "not enough execution/research tools to build the fixture"
    return names[:n]


@pytest.fixture()
def inline_limit_two():
    """Set the module-level inline tool limit to 2 and restore it after."""
    original = cp._max_inline_tool_rounds
    cp.set_max_inline_tool_rounds(2)
    yield
    cp.set_max_inline_tool_rounds(original)


def test_inline_tool_limit_confirmed_at_limit(inline_limit_two):
    tools = _exec_tools(3)
    ctx = {"scope": "main", "tool_name": tools[2], "tool_calls_made": tools[:2]}
    assert cp.validate_inline_tool_limit(ctx, {}) is True


def test_inline_tool_limit_passes_below_limit(inline_limit_two):
    tools = _exec_tools(2)
    ctx = {"scope": "main", "tool_name": tools[1], "tool_calls_made": tools[:1]}
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_never_blocks_orchestration_tools(inline_limit_two):
    """The model must always be able to delegate, however many tools it ran."""
    ctx = {
        "scope": "main",
        "tool_name": "worker_delegation",
        "tool_calls_made": _exec_tools(3),
    }
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_scoped_to_main(inline_limit_two):
    tools = _exec_tools(3)
    ctx = {"scope": "sub_session", "tool_name": tools[2], "tool_calls_made": tools[:2]}
    assert cp.validate_inline_tool_limit(ctx, {}) is False


def test_inline_tool_limit_disabled_when_zero():
    original = cp._max_inline_tool_rounds
    cp.set_max_inline_tool_rounds(0)
    try:
        tools = _exec_tools(3)
        ctx = {"scope": "main", "tool_name": tools[2], "tool_calls_made": tools[:2]}
        assert cp.validate_inline_tool_limit(ctx, {}) is False
    finally:
        cp.set_max_inline_tool_rounds(original)


def test_inline_tool_limit_per_thread_override_wins(inline_limit_two):
    """extra_context.max_inline_tool_rounds takes precedence over the global."""
    tools = _exec_tools(3)
    ctx = {
        "scope": "main",
        "tool_name": tools[2],
        "tool_calls_made": tools[:2],
        "extra_context": {"max_inline_tool_rounds": 10},
    }
    assert cp.validate_inline_tool_limit(ctx, {}) is False


# ── credential_redaction ────────────────────────────────────────────

def test_credential_redaction_confirmed_when_placeholder_present():
    ctx = {"assistant_response": f"Your key is {cp._SECRET_PLACEHOLDER} now."}
    assert cp.validate_credential_redaction(ctx, {}) is True


def test_credential_redaction_passes_on_clean_response():
    ctx = {"assistant_response": "Everything looks fine."}
    assert cp.validate_credential_redaction(ctx, {}) is False


# ── tool_schema ─────────────────────────────────────────────────────

def test_tool_schema_skips_when_not_pre_execution():
    """tool_args is None outside the pre_execution phase."""
    assert cp.validate_tool_schema({"tool_name": "task", "tool_args": None}, {}) is False


def test_tool_schema_skips_unknown_tool():
    ctx = {"tool_name": "no_such_tool_exists", "tool_args": {}}
    assert cp.validate_tool_schema(ctx, {}) is False


def test_tool_schema_flags_invalid_args_and_records_reason():
    ctx = {"tool_name": "task", "tool_args": {"totally_bogus_field": 1}}
    assert cp.validate_tool_schema(ctx, {}) is True
    assert ctx["_convergence_hook_reason"]


def test_tool_schema_registry_covers_every_validator():
    """Guard against a validator being added without registration."""
    assert set(cp._PROGRAMMATIC_VALIDATORS) == {
        "validate_workflow_spawn",
        "validate_phantom_tool_result",
        "validate_empty_promise",
        "validate_tool_schema",
        "validate_task_complete",
        "validate_repetition_loop",
        "validate_inline_tool_limit",
        "validate_credential_redaction",
    }
```

- [ ] **Step 2: Run the validator tests**

```bash
uv run pytest tests/test_convergence_validators.py -v
```

Expected: all PASS. These test existing code, so failures mean the test encodes a wrong assumption, not a product bug — **except** in three places where the exact behavior must be confirmed rather than assumed:

1. `test_tool_schema_flags_invalid_args_and_records_reason` depends on the `task` tool's JSON Schema rejecting an unknown property or a missing required one. If it passes validation instead, open `wintermute/core/tool_schemas.py`, find the `task` schema's `parameters`, and change the test's args to something the schema genuinely rejects (a missing required field is the most robust choice). Do not delete the test.
2. `_exec_tools` asserts that at least three tools are categorised `execution` or `research`. If that assert trips, print `TOOL_CATEGORIES` and adjust.
3. `test_empty_promise_confirmed_despite_trailing_question` depends on the pre-question line exceeding the 20-character `_TRIVIAL_LINE_THRESHOLD`. The line used is 67 characters, so it should hold.

- [ ] **Step 3: Write the failing redaction tests**

Create `tests/test_redaction.py`:

```python
"""Tests for credential redaction in the Convergence Protocol.

redact_credentials reads a module-level global, so every test that touches it
must restore the previous value — otherwise state leaks between tests.
"""

import pytest

from wintermute.core import convergence_protocol as cp


@pytest.fixture()
def secrets():
    """Install redaction secrets for one test, then restore the originals."""
    original = cp._redaction_secrets

    def _install(*values: str) -> None:
        cp.set_redaction_secrets(frozenset(values))

    yield _install
    cp._redaction_secrets = original


def test_redact_replaces_known_secret(secrets):
    secrets("sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("The key is sk-abcdefghijklmnop okay")
    assert was_redacted is True
    assert "sk-abcdefghijklmnop" not in text
    assert cp._SECRET_PLACEHOLDER in text


def test_redact_leaves_clean_text_untouched(secrets):
    secrets("sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("Nothing sensitive here.")
    assert was_redacted is False
    assert text == "Nothing sensitive here."


def test_redact_handles_overlapping_secrets_longest_first(secrets):
    """A short secret that is a substring of a long one must not cause the
    long one to leak its tail."""
    secrets("sk-abcdefgh", "sk-abcdefghijklmnop")
    text, was_redacted = cp.redact_credentials("key sk-abcdefghijklmnop end")
    assert was_redacted is True
    assert "ijklmnop" not in text


def test_redact_noop_when_no_secrets_configured(secrets):
    secrets()
    text, was_redacted = cp.redact_credentials("sk-abcdefghijklmnop")
    assert was_redacted is False
    assert text == "sk-abcdefghijklmnop"


def test_redact_handles_empty_text(secrets):
    secrets("sk-abcdefghijklmnop")
    assert cp.redact_credentials("") == ("", False)


def test_extract_config_secrets_collects_every_known_path():
    cfg = {
        "inference_backends": [{"api_key": "backend-key-123456"}],
        "matrix": {"password": "matrix-pass-123456", "access_token": "matrix-token-123456"},
        "whisper": {"api_key": "whisper-key-123456"},
        "memory": {
            "embeddings": {"api_key": "embed-key-123456"},
            "qdrant": {"api_key": "qdrant-key-123456"},
        },
        "skills": {"qdrant": {"api_key": "skills-qdrant-key-123456"}},
    }
    found = cp.extract_config_secrets(cfg)
    assert "backend-key-123456" in found
    assert "matrix-pass-123456" in found
    assert "matrix-token-123456" in found
    assert "whisper-key-123456" in found
    assert "embed-key-123456" in found
    assert "qdrant-key-123456" in found
    assert "skills-qdrant-key-123456" in found


def test_extract_config_secrets_drops_short_and_placeholder_values():
    cfg = {
        "inference_backends": [{"api_key": "short"}, {"api_key": "none"}],
        "whisper": {"api_key": "whisper-1"},
        "memory": {"embeddings": {"api_key": "llama-server"}},
    }
    assert cp.extract_config_secrets(cfg) == frozenset()


def test_extract_config_secrets_tolerates_empty_config():
    assert cp.extract_config_secrets({}) == frozenset()
```

- [ ] **Step 4: Run the redaction tests**

```bash
uv run pytest tests/test_redaction.py -v
```

Expected: all PASS. If `cp._redaction_secrets` does not exist at import time, check its initialisation near `set_redaction_secrets` in `convergence_protocol.py` and adjust the fixture to match the actual name.

- [ ] **Step 5: Run the whole suite together**

```bash
uv run pytest -q
```

Expected: PASS, including the pre-existing `tests/test_task_tools.py`. Run it as one command — the `secrets` fixture and the `inline_limit_two` fixture both mutate module globals, and running the files together is what proves they restore them.

- [ ] **Step 6: Add the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-groups

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Test
        run: uv run pytest -q
```

All three checks block. No coverage measurement: a percentage target invites tests written to move a number.

- [ ] **Step 7: Verify the workflow file parses**

```bash
uv run python -c "import yaml,pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('workflow yaml ok')"
```

Expected: `workflow yaml ok`.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/ci.yml tests/test_convergence_validators.py tests/test_redaction.py
git commit -m "test: cover convergence protocol validators and credential redaction

Adds blocking CI (ruff check, ruff format --check, pytest) on push and PR.
Covers all eight stage-2 validators with true-positive and false-positive
cases, plus redaction and config secret extraction."
```

---

### Task 3: Tool-call rescue and NL translator tests

**Files:**
- Create: `tests/test_tool_call_rescue.py`
- Create: `tests/test_nl_translator.py`

**Interfaces:**
- Consumes: the CI job from Task 2 (picks these files up automatically via `testpaths`).
- Produces: nothing other tasks depend on.

This is the weak-model parsing surface. `rescue_tool_calls(content, known_tool_names)` returns a list of `SyntheticToolCall`, each with `.function.name` (str) and `.function.arguments` (a JSON string). An empty list means no rescue was possible.

- [ ] **Step 1: Write the failing rescue tests**

Create `tests/test_tool_call_rescue.py`:

```python
"""Tests for weak-model tool-call rescue.

rescue_tool_calls salvages tool calls that a model emitted as text instead of
through the tool-calling API.  The negative cases matter as much as the
positive ones: rescuing prose into a phantom tool call is worse than rescuing
nothing.
"""

import json

from wintermute.core.tool_call_rescue import (
    _parse_arrow_style,
    _parse_cli_args,
    _parse_minimax_kv,
    _parse_yaml_like_kv,
    _try_parse_invoke_body,
    _try_parse_json_body,
    rescue_tool_calls,
)

KNOWN = {"read_file", "write_file", "worker_delegation", "task"}


def args_of(call) -> dict:
    """Decode a synthetic call's JSON argument string."""
    return json.loads(call.function.arguments)


# ── _try_parse_json_body ────────────────────────────────────────────

def test_json_body_flat_arguments():
    body = '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_parameters_alias():
    body = '{"name": "read_file", "parameters": {"path": "/tmp/x"}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_nested_function_shape():
    body = '{"function": {"name": "read_file", "arguments": {"path": "/tmp/x"}}}'
    calls = _try_parse_json_body(body, KNOWN)
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_json_body_array_of_calls():
    body = (
        '[{"name": "read_file", "arguments": {"path": "/a"}},'
        ' {"name": "write_file", "arguments": {"path": "/b"}}]'
    )
    calls = _try_parse_json_body(body, KNOWN)
    assert [c.function.name for c in calls] == ["read_file", "write_file"]


def test_json_body_rejects_unknown_tool_name():
    body = '{"name": "definitely_not_a_tool", "arguments": {}}'
    assert _try_parse_json_body(body, KNOWN) is None


def test_json_body_rejects_object_without_arguments_field():
    """Plain JSON in a discussion is not a tool call."""
    body = '{"name": "read_file", "description": "a tool that reads files"}'
    assert _try_parse_json_body(body, KNOWN) is None


def test_json_body_rejects_invalid_json():
    assert _try_parse_json_body("{not json at all", KNOWN) is None


def test_json_body_rejects_empty():
    assert _try_parse_json_body("   ", KNOWN) is None


# ── _try_parse_invoke_body ──────────────────────────────────────────

def test_invoke_body_single_block():
    body = (
        '<invoke name="read_file">'
        '<parameter name="path">/tmp/x</parameter>'
        "</invoke>"
    )
    calls = _try_parse_invoke_body(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_invoke_body_multiple_blocks():
    body = (
        '<invoke name="read_file"><parameter name="path">/a</parameter></invoke>'
        '<invoke name="write_file"><parameter name="path">/b</parameter></invoke>'
    )
    calls = _try_parse_invoke_body(body, KNOWN)
    assert [c.function.name for c in calls] == ["read_file", "write_file"]


def test_invoke_body_skips_unknown_tool():
    body = '<invoke name="nope"><parameter name="path">/a</parameter></invoke>'
    assert _try_parse_invoke_body(body, KNOWN) is None


# ── _parse_minimax_kv ───────────────────────────────────────────────

def test_minimax_kv_quoted_and_bare():
    assert _parse_minimax_kv('path="/tmp/x" limit=10') == {"path": "/tmp/x", "limit": "10"}


def test_minimax_kv_unescapes_inner_quotes():
    assert _parse_minimax_kv(r'text="say \"hi\""') == {"text": 'say "hi"'}


# ── _parse_cli_args ─────────────────────────────────────────────────

def test_cli_args_quoted_value_and_numeric_coercion():
    parsed = _parse_cli_args('--operation "interaction_log"\n--limit 10')
    assert parsed == {"operation": "interaction_log", "limit": 10}


def test_cli_args_normalises_kebab_case_keys():
    assert _parse_cli_args("--sub-session-id abc123") == {"sub_session_id": "abc123"}


def test_cli_args_coerces_float():
    assert _parse_cli_args("--threshold 0.85") == {"threshold": 0.85}


def test_cli_args_single_quoted_value():
    assert _parse_cli_args("--path '/tmp/x'") == {"path": "/tmp/x"}


# ── _parse_yaml_like_kv ─────────────────────────────────────────────

def test_yaml_like_kv_dash_prefixed_lines():
    body = (
        '- objective: "Implement GUI functionality"\n'
        "- timeout: 600\n"
        '- sub_session_id: "sub_aa62f3c8"'
    )
    assert _parse_yaml_like_kv(body) == {
        "objective": "Implement GUI functionality",
        "timeout": 600,
        "sub_session_id": "sub_aa62f3c8",
    }


def test_yaml_like_kv_unquoted_values():
    assert _parse_yaml_like_kv("action: list\nlimit: 5") == {"action": "list", "limit": 5}


# ── _parse_arrow_style ──────────────────────────────────────────────

def test_arrow_style_with_json_args():
    body = '{tool => "read_file", args => {"path": "/tmp/x"}}'
    calls = _parse_arrow_style(body, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"
    assert args_of(calls[0]) == {"path": "/tmp/x"}


def test_arrow_style_rejects_unknown_tool():
    body = '{tool => "not_a_tool", args => {"path": "/tmp/x"}}'
    assert _parse_arrow_style(body, KNOWN) == []


# ── rescue_tool_calls, end to end ───────────────────────────────────

def test_rescue_generic_xml_wrapper():
    content = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"


def test_rescue_fenced_json_block():
    content = '```json\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 1
    assert calls[0].function.name == "read_file"


def test_rescue_bare_invoke_block():
    content = (
        'Here you go:\n<invoke name="read_file">'
        '<parameter name="path">/tmp/x</parameter></invoke>'
    )
    calls = rescue_tool_calls(content, KNOWN)
    assert [c.function.name for c in calls] == ["read_file"]


def test_rescue_deduplicates_identical_calls():
    one = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    calls = rescue_tool_calls(one + one, KNOWN)
    assert len(calls) == 1


def test_rescue_ignores_plain_prose():
    """The most important negative case: prose must never become a tool call."""
    content = "I read the file and it contained the database configuration."
    assert rescue_tool_calls(content, KNOWN) == []


def test_rescue_ignores_markup_without_a_known_tool():
    content = '<tool_call>{"name": "not_a_real_tool", "arguments": {}}</tool_call>'
    assert rescue_tool_calls(content, KNOWN) == []


def test_rescue_returns_empty_for_empty_content():
    assert rescue_tool_calls("", KNOWN) == []


def test_rescue_returns_empty_when_no_tools_available():
    """An explicitly empty tool set means nothing can be rescued."""
    content = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
    assert rescue_tool_calls(content, set()) == []


def test_rescue_assigns_unique_ids():
    content = (
        '<tool_call>{"name": "read_file", "arguments": {"path": "/a"}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "/b"}}</tool_call>'
    )
    calls = rescue_tool_calls(content, KNOWN)
    assert len(calls) == 2
    assert calls[0].id != calls[1].id
    assert all(c.type == "function" for c in calls)
```

- [ ] **Step 2: Run the rescue tests**

```bash
uv run pytest tests/test_tool_call_rescue.py -v
```

Expected: all PASS. Two are worth checking carefully if they fail:

- `test_arrow_style_with_json_args` — `_parse_arrow_style` returns a plain list (not `None`) when nothing matches, unlike the `_try_parse_*` helpers. If the hash-rocket regex expects different spacing, read the `name_m = re.search(...)` block at `tool_call_rescue.py:261` and match the test input to it.
- `test_minimax_kv_unescapes_inner_quotes` — uses a raw string so the backslashes reach the parser literally, which is what a model actually emits.

- [ ] **Step 3: Write the failing NL translator tests**

Create `tests/test_nl_translator.py`:

```python
"""Tests for natural-language tool-call detection and JSON repair.

These cover the synchronous helpers only; translate_nl_tool_call itself
requires a backend and is out of scope.
"""

import json

from wintermute.core.nl_translator import (
    NL_TOOLS,
    _fix_unescaped_control_chars,
    _task_list_fastpath,
    is_nl_tool_call,
)


# ── is_nl_tool_call ─────────────────────────────────────────────────

def test_is_nl_tool_call_true_for_nl_tool_with_description():
    assert is_nl_tool_call("task", {"description": "list my tasks"}) is True


def test_is_nl_tool_call_tolerates_extra_keys():
    """The translator uses only the description; extra keys are discarded."""
    assert is_nl_tool_call("worker_delegation", {"description": "do it", "timeout": 600}) is True


def test_is_nl_tool_call_false_for_structured_args():
    assert is_nl_tool_call("task", {"action": "list"}) is False


def test_is_nl_tool_call_false_for_non_nl_tool():
    assert is_nl_tool_call("read_file", {"description": "read something"}) is False


def test_is_nl_tool_call_false_when_description_is_not_a_string():
    assert is_nl_tool_call("task", {"description": {"nested": "object"}}) is False


def test_nl_tools_membership_is_stable():
    assert NL_TOOLS == frozenset({"task", "worker_delegation", "skill"})


# ── _task_list_fastpath ─────────────────────────────────────────────

def test_task_list_fastpath_default_list():
    assert _task_list_fastpath("list tasks") == {"action": "list"}


def test_task_list_fastpath_is_case_and_punctuation_insensitive():
    assert _task_list_fastpath("  Show Me All Active Tasks!  ") == {"action": "list"}


def test_task_list_fastpath_status_variant():
    assert _task_list_fastpath("show completed tasks") == {
        "action": "list",
        "status": "completed",
    }


def test_task_list_fastpath_all_variant():
    assert _task_list_fastpath("list all tasks") == {"action": "list", "status": "all"}


def test_task_list_fastpath_returns_none_for_unknown_phrasing():
    """Anything unrecognised must fall through to the LLM translator."""
    assert _task_list_fastpath("create a task to water the plants") is None


def test_task_list_fastpath_returns_a_fresh_dict_each_call():
    """The default must be copied, or a caller mutating it corrupts the table."""
    first = _task_list_fastpath("list")
    first["action"] = "mutated"
    assert _task_list_fastpath("list") == {"action": "list"}


# ── _fix_unescaped_control_chars ────────────────────────────────────

def test_fix_control_chars_escapes_raw_newline_in_string():
    broken = '{"text": "line one\nline two"}'
    fixed = _fix_unescaped_control_chars(broken)
    assert json.loads(fixed) == {"text": "line one\nline two"}


def test_fix_control_chars_escapes_tab_and_carriage_return():
    broken = '{"text": "a\tb\rc"}'
    assert json.loads(_fix_unescaped_control_chars(broken)) == {"text": "a\tb\rc"}


def test_fix_control_chars_leaves_structural_whitespace_alone():
    valid = '{\n  "a": 1,\n  "b": 2\n}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"a": 1, "b": 2}


def test_fix_control_chars_preserves_existing_escapes():
    valid = '{"text": "already \\n escaped"}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"text": "already \n escaped"}


def test_fix_control_chars_preserves_escaped_quote():
    valid = '{"text": "he said \\"hi\\""}'
    assert json.loads(_fix_unescaped_control_chars(valid)) == {"text": 'he said "hi"'}


def test_fix_control_chars_is_a_noop_on_clean_input():
    clean = '{"a": "b"}'
    assert _fix_unescaped_control_chars(clean) == clean
```

- [ ] **Step 4: Run the NL translator tests**

```bash
uv run pytest tests/test_nl_translator.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the whole suite and the linters**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Expected: tests PASS, lint clean. The new test files must satisfy the same lint rules as everything else — that is what CI will run.

- [ ] **Step 6: Commit**

```bash
git add tests/test_tool_call_rescue.py tests/test_nl_translator.py
git commit -m "test: cover tool-call rescue parsers and NL translator helpers

Parametrised coverage of all six rescue parser paths plus rescue_tool_calls
end to end, including the negative case that prose must never be rescued
into a phantom tool call.  Adds NL detection and JSON control-character
repair tests."
```

---

### Task 4: Dead dependencies and stale documentation

**Files:**
- Modify: `pyproject.toml` (dependency list)
- Modify: `CLAUDE.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing later tasks depend on. Task 6 edits `pyproject.toml` again, so land this first to avoid a conflict.

- [ ] **Step 1: Confirm the dependencies are genuinely unused**

```bash
grep -rn "asyncpg\|aiosqlite\|sqlalchemy" --include=*.py wintermute/
```

Expected output: three hits, all in `wintermute/workers/scheduler_thread.py`, all for `sqlalchemy` (`SQLAlchemyJobStore`). Zero hits for `asyncpg` and `aiosqlite`. If `asyncpg` or `aiosqlite` appears, stop and do not remove that one.

- [ ] **Step 2: Remove the two dead dependencies**

In `pyproject.toml`, delete these two lines from `[project] dependencies`:

```toml
    "asyncpg>=0.30.0",
    "aiosqlite>=0.20.0",
```

Keep `"sqlalchemy>=2.0.51"` — `scheduler_thread.py:28` imports `SQLAlchemyJobStore` from it, so it is a direct dependency despite also arriving via APScheduler.

- [ ] **Step 3: Verify the environment still resolves and the app still imports**

```bash
uv sync
uv run python -c "import wintermute.main; import wintermute.workers.scheduler_thread; print('imports ok')"
uv run pytest -q
```

Expected: sync succeeds, `imports ok`, tests PASS.

- [ ] **Step 4: Fix the stale claims in CLAUDE.md**

Two edits:

1. In the "Commands" section, delete the sentence `No test suite exists.` from the line reading `No test suite exists. Configuration: copy config.yaml.example to config.yaml.`, leaving `Configuration: copy \`config.yaml.example\` to \`config.yaml\`.` Add a line documenting how to run tests:

```markdown
uv run pytest                         # Run the test suite
uv run ruff check . && uv run ruff format --check .   # Lint
```

2. In the "Key modules" table, the `Module` column lists flat filenames (`llm_thread.py`, `tools.py`, `dreaming.py`, …) but the tree is organised into packages. Correct each path to its real location. Verify each one before writing it:

```bash
ls wintermute/core/ wintermute/infra/ wintermute/tools/ wintermute/workers/ wintermute/interfaces/
```

Map every row of the table to the path that command shows. For example `dreaming.py` becomes `workers/dreaming.py`, `database.py` becomes `infra/database.py`, `convergence_protocol.py` becomes `core/convergence_protocol.py`. Do not guess: if a row's module no longer exists under any package, remove the row.

- [ ] **Step 5: Mark the completed roadmap item**

In `ROADMAP.md`, under "Phase 1 — Session store hardening", change:

```markdown
- [ ] SQLite WAL mode for the main database (concurrent readers during long
  inference calls).
```

to:

```markdown
- [x] SQLite WAL mode for the main database (concurrent readers during long
  inference calls). Done: `infra/database.py:88`, `infra/memory_store.py:113`,
  `infra/skill_store.py:134`.
```

Also, under "Phase 0", check the box on the pytest/CI and `asyncpg` items that Tasks 2, 3, and 4 complete, and update the "Wintermute currently has one 251-line test file and no test CI" sentence to reflect the new state.

- [ ] **Step 6: Verify**

```bash
uv run pytest -q
grep -c "test suite exists" CLAUDE.md || true
```

Expected: tests PASS; the grep prints `0` (or exits non-zero with no output), meaning the stale claim is gone.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml CLAUDE.md ROADMAP.md
git commit -m "chore: drop unused asyncpg and aiosqlite deps, refresh stale docs

Neither package is imported anywhere in wintermute/. sqlalchemy stays —
scheduler_thread.py imports SQLAlchemyJobStore directly.

CLAUDE.md: correct the module table to the core/infra/tools/workers/
interfaces package layout and drop the 'no test suite' claim.
ROADMAP.md: WAL mode was already implemented."
```

---

### Task 5: Remove skill documentation condensation (issue #236)

**Files:**
- Modify: `wintermute/workers/dreaming.py:697-747`
- Modify: `wintermute/infra/prompt_loader.py:22`
- Delete: `data/prompts/DREAM_SKILLS_CONDENSATION_PROMPT.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing later tasks depend on.

**Why:** `prompt_assembler` injects only a skills table of contents (name plus one-line summary), never the full documentation. The nightly condensation therefore buys no prompt-size reduction while LLM-rewriting every skill's documentation on every cycle — a skill that survives five nights has been through five lossy rewrites. Deletion is the whole fix.

**Order matters.** `DREAM_SKILLS_CONDENSATION_PROMPT.txt` is listed in `prompt_loader.REQUIRED_FILES`, and `validate()` raises `FileNotFoundError` at startup for any missing required file. Delete the registry entry before or with the file, never after.

- [ ] **Step 1: Confirm the file is a startup requirement**

```bash
grep -n "DREAM_SKILLS_CONDENSATION_PROMPT" -r wintermute/ data/prompts/ --include=*.py --include=*.txt -l
grep -n "DREAM_SKILLS_CONDENSATION_PROMPT" wintermute/infra/prompt_loader.py
```

Expected: a hit in `prompt_loader.py` inside `REQUIRED_FILES`, and a hit in `workers/dreaming.py`. This confirms the deletion order.

- [ ] **Step 2: Remove the entry from REQUIRED_FILES**

In `wintermute/infra/prompt_loader.py`, delete this line from the `REQUIRED_FILES` list:

```python
    "DREAM_SKILLS_CONDENSATION_PROMPT.txt",
```

- [ ] **Step 3: Remove the condensation block from the dreaming phase**

In `wintermute/workers/dreaming.py`, delete everything from the comment

```python
    # Condense each surviving skill (less aggressive — only if doc > 600 chars).
    condensed = 0
```

through the end of the `for name, rec in list(skills.items()):` loop that follows it — that is, the `try/except FileNotFoundError` template load, the `if condense_template is None:` early-return block, and the whole condensation loop.

Then replace the phase's trailing result lines:

```python
    result.items_processed = merged_skills + condensed
    result.summary = f"merged {merged_skills}, condensed {condensed} skills"
    logger.info("Dreaming phase skill_consolidation: %s", result.summary)
    return result
```

with:

```python
    result.items_processed = merged_skills
    result.summary = f"merged {merged_skills} skills"
    logger.info("Dreaming phase skill_consolidation: %s", result.summary)
    return result
```

Leave stage 1 (stale retirement) and stage 2 (dedup/merge) untouched — they run before this block and remain valuable.

- [ ] **Step 4: Delete the prompt template**

```bash
git rm data/prompts/DREAM_SKILLS_CONDENSATION_PROMPT.txt
```

- [ ] **Step 5: Verify no references survive**

```bash
grep -rn "DREAM_SKILLS_CONDENSATION_PROMPT\|condense_template\|condensed" wintermute/ || echo "no references remain"
```

Expected: `no references remain`. A leftover `condensed` variable means the deletion in Step 3 was partial and the module will raise `NameError` at runtime.

- [ ] **Step 6: Verify startup prompt validation still passes**

```bash
uv run python -c "
from wintermute.infra import prompt_loader
prompt_loader.validate()
print('prompt validation ok')
"
uv run python -c "import wintermute.workers.dreaming; print('dreaming imports ok')"
uv run pytest -q
uv run ruff check .
```

Expected: `prompt validation ok`, `dreaming imports ok`, tests PASS, lint clean. The first check is the one that catches a forgotten `REQUIRED_FILES` entry — without it, the failure appears only when the daemon next starts.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "fix: remove skill documentation condensation from dreaming (#236)

prompt_assembler injects only the skills table of contents, so condensing
documentation reduced no prompt size while lossily LLM-rewriting every
skill each night — compounding across cycles and flattening the embedding
signal, since skill_store embeds summary + documentation.

Removes the condensation loop, its prompt template, and the template's
entry in prompt_loader.REQUIRED_FILES. Stale retirement and dedup/merge
are unchanged.

Closes #236"
```

---

### Task 6: Zero-config local embedding fallback (issue #180)

**Files:**
- Create: `wintermute/infra/local_embeddings.py`
- Create: `tests/test_local_embeddings.py`
- Modify: `wintermute/infra/llm_utils.py:217-221` (the `if not endpoint: raise` branch)
- Modify: `wintermute/infra/memory_store.py` (`init()` hard fail; `LocalVectorBackend.init()` schema)
- Modify: `wintermute/infra/skill_store.py` (`init()` hard fail)
- Modify: `pyproject.toml` (optional extra)
- Modify: `config.yaml.example`, `docs/installation.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (the public surface of the new module, relied on by `llm_utils` and both stores):
  - `local_embeddings.DIMENSIONS: int = 384`
  - `local_embeddings.MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"`
  - `local_embeddings.is_available() -> bool` — True when `onnxruntime` and `tokenizers` import successfully. Never raises, never downloads.
  - `local_embeddings.embed_local(texts: list[str]) -> list[list[float]]` — returns one 384-float L2-normalised vector per input. Raises `RuntimeError` when the extra is not installed or the model cannot be fetched.
  - `local_embeddings.provider_name() -> str` — returns `"local:all-MiniLM-L6-v2"`, the value persisted for the dimension guard.

**Scope boundary:** zero-config only. The local model activates **only** when `embeddings.endpoint` is unset. A configured endpoint stays authoritative; if it errors, memory fails loudly exactly as today. Silent runtime fallback is out of scope, because writing 384-dimensional vectors into a store built at 1536 corrupts the index.

- [ ] **Step 1: Add the optional dependency extra**

In `pyproject.toml`, after the `[project.scripts]` block, add:

```toml
[project.optional-dependencies]
# Zero-config memory: bundles a local ONNX embedding model so a fresh
# deployment works with no embeddings endpoint configured. Optional by
# design — the roadmap caps direct dependencies near 17.
local-embeddings = [
    "onnxruntime>=1.20.0",
    "tokenizers>=0.21.0",
]
```

- [ ] **Step 2: Install the extra and confirm it resolves**

```bash
uv sync --extra local-embeddings
uv run python -c "import onnxruntime, tokenizers; print('extra ok')"
```

Expected: `extra ok`.

- [ ] **Step 3: Write the failing tests for the new module**

Create `tests/test_local_embeddings.py`. These must pass whether or not the extra is installed and must never hit the network:

```python
"""Tests for the local ONNX embedding fallback.

No test here downloads a model or touches the network.  The one test that
would need real inference is skipped unless the model is already cached.
"""

import pytest

from wintermute.infra import local_embeddings


def test_constants_are_stable():
    """These values are persisted in the store's dimension guard; changing
    them invalidates existing local-vector databases."""
    assert local_embeddings.DIMENSIONS == 384
    assert local_embeddings.MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert local_embeddings.provider_name() == "local:all-MiniLM-L6-v2"


def test_is_available_returns_a_bool_without_raising():
    assert isinstance(local_embeddings.is_available(), bool)


def test_is_available_reports_false_when_extra_is_missing(monkeypatch):
    def _boom():
        raise ImportError("no onnxruntime")

    monkeypatch.setattr(local_embeddings, "_import_backend", _boom)
    local_embeddings.reset_cache()
    try:
        assert local_embeddings.is_available() is False
    finally:
        # is_available memoises; clear the poisoned False so later tests
        # (and the skipif above) see the real environment again.
        local_embeddings.reset_cache()


def test_embed_local_raises_a_clear_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(local_embeddings, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as exc:
        local_embeddings.embed_local(["hello"])
    assert "local-embeddings" in str(exc.value)


def test_embed_local_returns_empty_for_empty_input():
    assert local_embeddings.embed_local([]) == []


@pytest.mark.skipif(
    not local_embeddings.is_model_cached(),
    reason="model not downloaded; run once online to enable this test",
)
def test_embed_local_produces_normalised_vectors_of_the_right_shape():
    vectors = local_embeddings.embed_local(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert all(len(v) == local_embeddings.DIMENSIONS for v in vectors)
    norms = [sum(x * x for x in v) ** 0.5 for v in vectors]
    assert all(abs(n - 1.0) < 1e-3 for n in norms)
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
uv run pytest tests/test_local_embeddings.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wintermute.infra.local_embeddings'`.

- [ ] **Step 5: Implement the local embedding provider**

Create `wintermute/infra/local_embeddings.py`:

```python
"""Local ONNX embedding fallback for zero-config memory.

Used only when ``memory.embeddings.endpoint`` is unset.  A configured
endpoint always wins; this module never silently substitutes for a
configured-but-failing endpoint, because mixing vector dimensions would
corrupt an existing store.

The model (all-MiniLM-L6-v2, 384-dimensional) and its tokenizer are fetched
over HTTPS on first use and cached under ``data/.embedding_cache/``.  We
fetch by URL rather than depend on ``huggingface_hub`` to keep the
dependency budget flat.
"""

import logging
import threading
from pathlib import Path

from wintermute.infra.paths import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_SEQ_LENGTH = 256

CACHE_DIR: Path = DATA_DIR / ".embedding_cache"
_MODEL_PATH = CACHE_DIR / "model.onnx"
_TOKENIZER_PATH = CACHE_DIR / "tokenizer.json"

_BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"
_MODEL_URL = f"{_BASE_URL}/onnx/model.onnx"
_TOKENIZER_URL = f"{_BASE_URL}/tokenizer.json"

_lock = threading.Lock()
_session = None      # onnxruntime.InferenceSession
_tokenizer = None    # tokenizers.Tokenizer
_available: "bool | None" = None


def provider_name() -> str:
    """Identifier persisted alongside stored vectors for the dimension guard."""
    return "local:all-MiniLM-L6-v2"


def _import_backend():
    """Import the optional extra.  Raises ImportError when it is absent."""
    import onnxruntime
    import tokenizers
    return onnxruntime, tokenizers


def reset_cache() -> None:
    """Drop memoised availability and loaded model state (used by tests)."""
    global _session, _tokenizer, _available
    with _lock:
        _session = None
        _tokenizer = None
        _available = None


def is_available() -> bool:
    """True when the optional extra is importable.  Never raises, never
    downloads."""
    global _available
    if _available is None:
        try:
            _import_backend()
            _available = True
        except ImportError:
            _available = False
    return _available


def is_model_cached() -> bool:
    """True when both model artefacts are already on disk."""
    return _MODEL_PATH.is_file() and _TOKENIZER_PATH.is_file()


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading embedding artefact %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    tmp.rename(dest)


def _ensure_loaded():
    """Load (downloading on first use) the ONNX session and tokenizer."""
    global _session, _tokenizer
    if _session is not None and _tokenizer is not None:
        return _session, _tokenizer

    if not is_available():
        raise RuntimeError(
            "Local embeddings are not installed. Either configure "
            "memory.embeddings.endpoint in config.yaml, or install the "
            "optional extra: uv sync --extra local-embeddings"
        )

    onnxruntime, tokenizers = _import_backend()

    with _lock:
        if _session is not None and _tokenizer is not None:
            return _session, _tokenizer
        try:
            if not _MODEL_PATH.is_file():
                _download(_MODEL_URL, _MODEL_PATH)
            if not _TOKENIZER_PATH.is_file():
                _download(_TOKENIZER_URL, _TOKENIZER_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch the local embedding model from {_BASE_URL}. "
                f"The first use requires outbound HTTPS. Original error: {exc}"
            ) from exc

        _tokenizer = tokenizers.Tokenizer.from_file(str(_TOKENIZER_PATH))
        _tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        _tokenizer.enable_padding(length=None)
        _session = onnxruntime.InferenceSession(
            str(_MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        logger.info("Local embedding model loaded (%s, %d-dim)", MODEL_ID, DIMENSIONS)
    return _session, _tokenizer


def embed_local(texts: list[str]) -> list[list[float]]:
    """Embed *texts* with the cached local model.

    Returns one L2-normalised 384-float vector per input, using mean pooling
    over the token embeddings — the pooling all-MiniLM-L6-v2 was trained with.
    """
    if not texts:
        return []

    import numpy as np

    session, tokenizer = _ensure_loaded()
    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    expected = {i.name for i in session.get_inputs()}
    if "token_type_ids" in expected:
        inputs["token_type_ids"] = np.zeros_like(input_ids)
    inputs = {k: v for k, v in inputs.items() if k in expected}

    token_embeddings = session.run(None, inputs)[0]  # (batch, seq, 384)

    # Mean pooling over non-padding tokens.
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    pooled = summed / counts

    # L2 normalise so cosine similarity reduces to a dot product.
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    normalised = pooled / norms

    return normalised.astype(np.float32).tolist()
```

If `wintermute/infra/paths.py` does not export `DATA_DIR`, open it and use whatever constant points at the `data/` directory; the cache must live under it so the existing `data/` git auto-versioning ignores or tracks it consistently.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest tests/test_local_embeddings.py -v
```

Expected: PASS, with `test_embed_local_produces_normalised_vectors_of_the_right_shape` SKIPPED (no model cached, no network in tests).

- [ ] **Step 7: Exclude the model cache from version control**

Add to `.gitignore`:

```gitignore
data/.embedding_cache/
```

Model artefacts are ~90 MB; the `data/` directory has its own auto-committing git repo, and committing binary blobs into it on every dreaming cycle would be a problem.

- [ ] **Step 8: Commit the provider**

```bash
git add wintermute/infra/local_embeddings.py tests/test_local_embeddings.py pyproject.toml .gitignore
git commit -m "feat: add local ONNX embedding provider (#180)

all-MiniLM-L6-v2 (384-dim) behind an optional local-embeddings extra,
cached under data/.embedding_cache/. Not wired in yet."
```

- [ ] **Step 9: Route `embed()` to the local provider when no endpoint is configured**

In `wintermute/infra/llm_utils.py`, replace:

```python
    if not endpoint:
        raise RuntimeError("embeddings.endpoint is not configured")
```

with:

```python
    if not endpoint:
        # Zero-config fallback: no endpoint configured at all.  A configured
        # endpoint always takes precedence and is never silently replaced —
        # mixing vector dimensions would corrupt an existing store.
        from wintermute.infra import local_embeddings

        if local_embeddings.is_available():
            return local_embeddings.embed_local(texts)
        raise RuntimeError(
            "embeddings.endpoint is not configured and the local embedding "
            "fallback is unavailable. Either set memory.embeddings.endpoint "
            "in config.yaml, or install the optional extra: "
            "uv sync --extra local-embeddings"
        )
```

Place this **before** the prefix handling and truncation blocks: the local model applies neither task prefixes (which are model-specific, for example EmbeddingGemma's `search_query: `) nor the endpoint's `max_text_chars` cap, since it truncates at `MAX_SEQ_LENGTH` tokens itself. The import is function-local to keep `onnxruntime` off the startup path.

This one edit serves all 15 `_embed()` call sites across `memory_store`, `skill_store`, and `database`.

- [ ] **Step 10: Relax the two startup hard-fails**

In `wintermute/infra/memory_store.py`, inside `init()`, replace the `if not has_embeddings: raise ValueError(...)` block with:

```python
    if not has_embeddings:
        from wintermute.infra import local_embeddings

        if not local_embeddings.is_available():
            raise ValueError(
                "memory.embeddings.endpoint is required. "
                "Configure an OpenAI-compatible /v1/embeddings endpoint in config.yaml,\n"
                "  or install the zero-config local fallback: "
                "uv sync --extra local-embeddings\n"
                "  Example:\n"
                "    memory:\n"
                "      embeddings:\n"
                "        endpoint: \"http://localhost:8080/v1\"\n"
                "        model: \"text-embedding-3-small\"\n"
                "        dimensions: 1536"
            )
        logger.warning(
            "No embeddings endpoint configured — using the local %s fallback "
            "(%d-dim). Quality is below a hosted model; configure "
            "memory.embeddings.endpoint for better recall.",
            local_embeddings.MODEL_ID, local_embeddings.DIMENSIONS,
        )
```

Apply the same change to the corresponding `if not _embed_cfg.get("endpoint"): raise ValueError(...)` block in `wintermute/infra/skill_store.py:1068`, adapting the message to say "skill store" as the existing text does.

- [ ] **Step 11: Add the dimension guard to the local vector backend**

`LocalVectorBackend` stores bare float32 blobs with no recorded dimension, so switching providers would surface as an opaque numpy shape error mid-query. In `LocalVectorBackend.init()` in `memory_store.py`, after the existing `CREATE TABLE` and inline column migrations and before `conn.commit()`, add:

```python
            # Record which embedding provider built this store.  Switching
            # providers changes the vector dimension, which silently breaks
            # similarity search — fail loudly instead.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS store_meta ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL"
                ")"
            )
```

Then, still inside `init()` and after the commit, add:

```python
        self._check_provider_compatibility()
```

And add this method to `LocalVectorBackend`:

```python
    def _active_provider(self) -> tuple[str, int]:
        """Return (provider_name, dimension) for the currently configured
        embedding source."""
        if self._embed_cfg.get("endpoint"):
            model = self._embed_cfg.get("model", "text-embedding-3-small")
            return f"endpoint:{model}", int(self._embed_cfg.get("dimensions", 0))
        from wintermute.infra import local_embeddings
        return local_embeddings.provider_name(), local_embeddings.DIMENSIONS

    def _check_provider_compatibility(self) -> None:
        """Refuse to start when the stored vectors were built by a provider
        with a different dimension."""
        provider, dimension = self._active_provider()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM store_meta WHERE key = 'embedding_dimension'"
                ).fetchone()
                stored_dim = int(row[0]) if row else None
                has_vectors = conn.execute(
                    "SELECT 1 FROM local_vectors LIMIT 1"
                ).fetchone() is not None

                if stored_dim is None:
                    if has_vectors and dimension:
                        # Pre-existing store from before the guard existed:
                        # adopt its dimension from an actual stored vector.
                        blob = conn.execute(
                            "SELECT vector FROM local_vectors LIMIT 1"
                        ).fetchone()[0]
                        stored_dim = len(blob) // 4  # float32
                    else:
                        stored_dim = dimension
                    conn.execute(
                        "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                        ("embedding_dimension", str(stored_dim)),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO store_meta (key, value) VALUES (?, ?)",
                        ("embedding_provider", provider),
                    )
                    conn.commit()

                if dimension and stored_dim and stored_dim != dimension:
                    raise ValueError(
                        f"Memory store was built with {stored_dim}-dimensional "
                        f"vectors, but the active embedding provider "
                        f"({provider}) produces {dimension}-dimensional ones. "
                        f"Similarity search would be meaningless.\n"
                        f"  Either restore the previous embeddings "
                        f"configuration, or delete {self._db_path} to start "
                        f"fresh (existing memories will be lost)."
                    )
            finally:
                conn.close()
```

A re-embed migration is deliberately out of scope for this batch.

- [ ] **Step 12: Write a test for the dimension guard**

Append to `tests/test_local_embeddings.py`:

```python
def test_dimension_mismatch_is_refused(tmp_path, monkeypatch):
    """A store built at one dimension must refuse a provider with another."""
    import sqlite3

    from wintermute.infra import memory_store

    db_path = tmp_path / "local_vectors.db"
    monkeypatch.setattr(memory_store, "LOCAL_VECTOR_DB_PATH", db_path)

    backend = memory_store.LocalVectorBackend({"embeddings": {}})
    backend._db_path = db_path
    backend.init()

    # Rewrite the recorded dimension to simulate a store built elsewhere.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('embedding_dimension', '1536')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ValueError) as exc:
        backend._check_provider_compatibility()
    assert "1536" in str(exc.value)
    assert "384" in str(exc.value)
```

- [ ] **Step 13: Run the tests**

```bash
uv run pytest tests/test_local_embeddings.py -v
uv run pytest -q
```

Expected: all PASS (the inference test still SKIPPED). If `LOCAL_VECTOR_DB_PATH` is read at import time into `self._db_path`, the test sets `backend._db_path` directly to cover that; keep both lines.

- [ ] **Step 14: Verify the fallback end to end, offline first**

```bash
uv run python -c "
from wintermute.infra import llm_utils
try:
    llm_utils.embed(['hello'], {})
    print('embedded via fallback')
except RuntimeError as exc:
    print('RuntimeError:', exc)
"
```

With the extra installed and the model cached, expect `embedded via fallback`. With the extra installed but no cache and no network, expect a `RuntimeError` naming the download failure. Without the extra, expect the message pointing at `uv sync --extra local-embeddings`. All three are correct outcomes; confirm the message you get matches your environment.

- [ ] **Step 15: Document the fallback**

In `config.yaml.example`, under the `memory.embeddings` block, add:

```yaml
    # Optional. When endpoint is unset and the local-embeddings extra is
    # installed (uv sync --extra local-embeddings), wintermute falls back to
    # a bundled 384-dimensional all-MiniLM-L6-v2 model, downloaded once to
    # data/.embedding_cache/ and run on CPU. Recall quality is below a hosted
    # model — the fallback is a floor that keeps memory working, not a
    # recommended default. A configured endpoint always takes precedence and
    # is never silently replaced.
```

In `docs/installation.md`, add a short subsection under the memory/embeddings setup covering: the extra's install command, the one-time ~90 MB HTTPS download on first use, the cache location, and the fact that switching between the local model and a hosted endpoint later is refused at startup because the dimensions differ.

- [ ] **Step 16: Full verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python -c "import wintermute.main; print('imports ok')"
```

Expected: tests PASS, lint clean, `imports ok`.

- [ ] **Step 17: Commit**

```bash
git add -A
git commit -m "feat: zero-config memory via local embedding fallback (#180)

A fresh deployment with no embeddings endpoint no longer fails at startup.
llm_utils.embed() routes to the local ONNX provider when no endpoint is
configured, which covers all 15 _embed() call sites at once. A configured
endpoint stays authoritative and is never silently replaced.

Adds a dimension guard to LocalVectorBackend: the store records the
provider and vector dimension that built it and refuses to open under a
provider with a different dimension, rather than degrading similarity
search silently.

Closes #180"
```

---

## Verification of the whole batch

After Task 6, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -c "import wintermute.main; print('imports ok')"
grep -rn "asyncpg\|aiosqlite" --include=*.py --include=*.toml . | grep -v '\.venv' || echo "dead deps gone"
grep -rn "DREAM_SKILLS_CONDENSATION" wintermute/ data/prompts/ || echo "condensation gone"
```

Expected: lint clean, tests pass, imports ok, `dead deps gone`, `condensation gone`.

Then open a pull request and let CI confirm the same on a clean checkout.
