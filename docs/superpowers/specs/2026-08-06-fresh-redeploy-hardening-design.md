# Fresh Redeploy Hardening — Design

**Date:** 2026-08-06
**Status:** Approved, pending implementation plan

## Goal

Make wintermute safe to redeploy fresh and run in production. This batch covers
ROADMAP Tier 1 (production-blocking fixes) and Tier 2 (a minimal test and CI
floor). It deliberately excludes the roadmap's benchmark, eval-harness, and
memory-confidence work: those are high effort with no effect on uptime.

Scope: five commits, landed in the order below. Mechanical changes precede
behavioral ones so that every later diff stays reviewable.

## Verified starting state

Audited against the working tree on 2026-08-06:

- `tests/` contains one file (`test_task_tools.py`). Only `release-please.yml`
  runs in CI; no lint or test workflow exists.
- `ruff check .` reports 581 errors (513 auto-fixable, no unsafe fixes needed);
  `ruff format --check .` would reformat 47 of 78 files.
- `asyncpg` and `aiosqlite` are declared in `pyproject.toml` with zero imports
  anywhere in `wintermute/`. `sqlalchemy` **is** used, via
  `workers/scheduler_thread.py:28` (`SQLAlchemyJobStore`), and stays.
- `DREAM_SKILLS_CONDENSATION_PROMPT.txt` is listed in
  `infra/prompt_loader.py:REQUIRED_FILES`. Deleting the template without editing
  that list turns startup into a `FileNotFoundError`.
- All embedding calls funnel through one function, `infra/llm_utils.py:embed()`,
  which raises `"embeddings.endpoint is not configured"` when the endpoint is
  empty. There are 15 `_embed()` call sites across `memory_store`, `skill_store`,
  and `database`; all of them reach the endpoint through that single function.
- `LocalVectorStore` persists embeddings as raw float32 blobs with no recorded
  dimension. `QdrantStore` tracks `self._dimensions` (default 1536).
- WAL mode is already enabled (`infra/database.py:88`, `memory_store.py:113`,
  `skill_store.py:134`). The ROADMAP's Phase 1 WAL item is stale.

## Commit 1 — Lint baseline

No behavior change. Establishes the clean tree that blocking CI requires.

1. Run `ruff check --fix` and `ruff format`. Do **not** pass `--unsafe-fixes`.
2. Triage the ~68 errors that remain. Fix the trivial ones.
3. Where a rule fights the codebase systematically — `SIM105` versus the
   deliberate broad `except Exception` handlers is the known case — add a
   per-rule entry to `[tool.ruff.lint] ignore` in `pyproject.toml` with a
   comment stating why. Prefer one documented ignore over scattered `# noqa`.

**Verification:** `ruff check .` and `ruff format --check .` both clean;
`uv run pytest` passes; `git diff --stat` reviewed for any hunk that is not
purely mechanical.

## Commit 2 — Test floor and CI

### Tests

All targets are synchronous and perform no IO, so no mocks, no fixtures, and no
`pytest-asyncio` are required. `pytest` and `ruff` are already in the dev group;
this commit adds no dependencies.

- `tests/test_convergence_validators.py` — the eight stage-2 validators in
  `core/convergence_protocol.py`: `validate_workflow_spawn`,
  `validate_phantom_tool_result`, `validate_empty_promise`,
  `validate_tool_schema`, `validate_task_complete`, `validate_repetition_loop`,
  `validate_inline_tool_limit`, `validate_credential_redaction`. Each gets at
  least one true positive and one false positive; the false-positive cases are
  the point, since a validator that over-fires blocks legitimate tool calls. The
  documented example — a response ending in `?` is not an empty promise — is one
  such case.
- `tests/test_tool_call_rescue.py` — parametrized over every parser path in
  `core/tool_call_rescue.py`: `_try_parse_json_body`, `_try_parse_invoke_body`,
  `_parse_minimax_kv`, `_parse_cli_args`, `_parse_arrow_style`,
  `_parse_yaml_like_kv`. Plus `rescue_tool_calls` end to end over synthetic
  malformed strings, and a negative case asserting that ordinary prose is not
  rescued into a phantom tool call.
- `tests/test_redaction.py` — `redact_credentials` and `extract_config_secrets`.
- `tests/test_nl_translator.py` — `is_nl_tool_call`,
  `_fix_unescaped_control_chars`, `_task_list_fastpath`.

Async orchestrators (`run_convergence_protocol`, `process_tool_call`) are out of
scope for this batch; testing them needs a stubbed backend pool, which roughly
doubles the work.

### CI

New `.github/workflows/ci.yml`, triggered on push and pull request:
`uv sync`, then `ruff check .`, `ruff format --check .`, `uv run pytest`. All
three block the merge. No coverage measurement and no coverage threshold — a
percentage target would invite tests written to move a number.

## Commit 3 — Dead dependencies and stale docs

- Remove `asyncpg` and `aiosqlite` from `pyproject.toml`. Keep `sqlalchemy`.
- `CLAUDE.md`: the module table lists flat filenames, but the tree is organized
  into `core/`, `infra/`, `tools/`, `workers/`, and `interfaces/` packages.
  Correct the paths and delete the "No test suite exists" line.
- `ROADMAP.md`: mark the Phase 1 WAL item done.

**Verification:** `uv sync` succeeds against the trimmed dependency list and the
application still starts.

## Commit 4 — Issue #236, remove skill condensation

The nightly dreaming cycle LLM-rewrites every skill's documentation. Because
`prompt_assembler` injects only a table of contents, that rewriting buys no
prompt-size reduction while compounding information loss on every cycle. The fix
is deletion.

1. Delete the condensation loop in `workers/dreaming.py` (roughly lines
   698-732) and the `condensed` counter in the phase's result summary.
2. Delete `data/prompts/DREAM_SKILLS_CONDENSATION_PROMPT.txt`.
3. Remove `"DREAM_SKILLS_CONDENSATION_PROMPT.txt"` from
   `infra/prompt_loader.py:REQUIRED_FILES`. Skipping this step makes startup
   fail.

Stage 1 stale retirement and stage 2 dedup/merge are unaffected.

**Verification:** startup validates prompts without error; a dreaming run
completes and its summary no longer mentions condensation.

## Commit 5 — Issue #180, zero-config local embeddings

A fresh deployment with no embeddings endpoint currently fails at startup
(`memory_store.py:1206`). Scope for this batch is **zero-config only**: the
local model is used when no endpoint is configured. A configured endpoint stays
authoritative, and if it errors, memory fails loudly exactly as it does today.
Silent runtime fallback is explicitly rejected, because writing 384-dimensional
vectors into a store built at 1536 dimensions corrupts the index.

- New `wintermute/infra/local_embeddings.py`: a lazily loaded ONNX
  `all-MiniLM-L6-v2` (384-dimensional), with model and tokenizer cached under
  `data/.embedding_cache/`. Fetch over plain HTTPS on first use rather than
  adding a `huggingface_hub` dependency; the roadmap's dependency budget is
  explicit.
- Integrate at the single choke point. `llm_utils.embed()` currently raises when
  the endpoint is empty; route to the local provider there instead. All 15
  `_embed()` call sites inherit the behavior with no change.
- `memory_store`'s factory stops hard-failing when no endpoint is configured
  **if** the optional extra is installed. When neither an endpoint nor the extra
  is available, keep today's loud, instructive error.
- `pyproject.toml`: add
  `[project.optional-dependencies] local-embeddings = ["onnxruntime", "tokenizers"]`.
- **Dimension guard.** `LocalVectorStore` stores bare float32 blobs and records
  no dimension, so a provider switch would surface as a numpy shape error deep
  in a query. Persist the active provider name and vector dimension in each
  store's metadata. On startup, if the stored dimension disagrees with the
  active provider's, refuse to start with a message naming both values and the
  remedy. A re-embed migration is out of scope for this batch.
- Document the fallback and its one-time network fetch in `config.yaml.example`
  and `docs/installation.md`.

**Verification:** with no endpoint configured and the extra installed, a memory
can be written and retrieved; with a dimension mismatch, startup refuses with
the intended message.

## Risks

- Commit 1 touches most of the tree. Land and merge it alone, before anything
  else, so it never obscures a behavioral diff.
- The local embedding model needs one-time outbound HTTPS from the deployment
  host. Wintermute runs on a remote server, so confirm that egress before
  relying on the fallback.
- Local embedding quality is below that of a hosted model. The fallback is a
  floor that keeps memory working, not a recommended default.

## Out of scope

Deferred deliberately: issues #248 and #247 (memory confidence and
episodic/semantic separation), the evaluation harness and published benchmark,
FTS5 search, session export/import, the Phase 3 god-file splits, and the
mechanical refactors #204 through #210.
