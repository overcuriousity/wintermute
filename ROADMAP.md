# Wintermute Roadmap

Written after a source-level comparison against Nous Research's `hermes-agent`
(v0.20.0, cloned and audited 2026-08). This roadmap takes what hermes
demonstrably does well, explicitly rejects its bloat and marketing, and doubles
down on the areas where wintermute is already architecturally ahead.

## Positioning

Wintermute is not hermes and should not try to become it. Hermes is an
interface-and-ecosystem play: 22 chat-platform adapters, Electron, TUI, web,
MCP, ~1.5M LOC, real test discipline, and a "learning loop" that is, by its
own docstring's admission, prompt-file generation ("There is no separate
distillation engine"). Wintermute is an intelligence-depth play: a single-node
runtime where the investment went into outcome feedback, memory consolidation,
and making weak models reliable.

The strategy: stay small, become trustworthy, and make the learning/memory
machinery measurably real instead of advertised.

---

## Phase 0 — Engineering rigor (the existential gap)

Wintermute currently has one 251-line test file and no test CI. Everything else
in this roadmap is unsafe without fixing this first. This is the single largest
lesson from hermes, which has a genuine ~640k-LOC test suite on its Python core.

- [ ] Set up pytest + pytest-asyncio with coverage reporting.
- [ ] CI workflow (GitHub Actions): lint (ruff, config already in
  `pyproject.toml`) + tests on every PR. Currently only release-please runs.
- [ ] Test priority order, by risk:
  1. `core/convergence_protocol.py` — it gates tool execution; a regression
     here silently corrupts or blocks agent actions.
  2. `core/inference_engine.py::process_tool_call` — the shared tool pipeline
     (JSON repair, NL translation, dispatch).
  3. `infra/memory_store.py` / `infra/skill_store.py` — vector math, dedup,
     outcome recording.
  4. `core/sub_session.py` — DAG resolution, failure propagation, timeout
     continuation.
  5. `core/tool_call_rescue.py` / `core/nl_translator.py` — weak-model parsing;
     golden-file tests with real malformed outputs.
- [ ] Coverage gates only on the above modules first; do not chase a global
  percentage.
- [ ] Audit the 311 broad `except Exception` handlers: narrow or annotate each
  in the core loop and tool pipeline.
- [ ] Remove dead `asyncpg` dependency (declared, zero imports).
- [ ] Update `CLAUDE.md`: stale module table (flat filenames vs. `core/`,
  `tools/` packages) and the outdated "no test suite" claim.

## Phase 1 — Session store hardening (take from hermes)

Hermes's `hermes_state.py` is the best-engineered part of their repo. Steal the
techniques, not the code:

- [ ] SQLite WAL mode for the main database (concurrent readers during long
  inference calls).
- [ ] FTS5 full-text search over conversation history — hermes's trigram/CJK
  tokenizer approach is worth copying. This gives cheap, local, deterministic
  recall that complements (not replaces) vector memory.
- [ ] Session export/import (portability) — single-file archive of a thread's
  messages, memories, and outcomes. Enables backup, migration, and sharing
  interesting trajectories.
- [ ] Compaction session chaining: link post-compaction threads to their
  predecessors (`parent_thread_id`) so long-running work survives context
  compaction as a navigable chain rather than an amnesiac fresh start.

## Phase 2 — Make the differentiators measurable (strengthen what is better)

These are the subsystems where wintermute already does what hermes only
markets. The gap is proof, not features.

### 2a. Outcome tracking → closed-loop skill selection

`sub_session_outcomes` is written at three exit points and injected as
similarity-matched feedback (`tools/session_tools.py:93`). Close the loop:

- [ ] Skill selection bias: use `skill_store.record_outcome` stats to rank
  skills at injection time (success rate weighted by recency and similarity),
  not just similarity alone.
- [ ] Negative feedback: outcomes below a quality threshold should *demote*
  skills and memories, not just accumulate.
- [ ] An evaluation harness: fixed task suite run against a fixed weak model,
  reporting skill-hit-rate and outcome-quality over time. This is the honest
  version of hermes's `mcp-research-data` bench JSONs — a number we can cite
  instead of an adjective.

### 2b. Dreaming pipeline — prove quality

The quality survival tracking (`dreaming_quality` table, `_check_survival`)
exists but has no visibility.

- [ ] Dreaming dashboard/report: memories created vs. survived, contradiction
  rate, stale-prune rate, per-phase quality scores. Surface in the web debug
  panel.
- [ ] A/B toggle: run the eval harness with dreaming enabled/disabled to
  demonstrate (or refute) that consolidation improves outcomes. If it doesn't
  help, simplify it — the honest answer is the point.

### 2c. Weak-model resilience — the benchmark as the headline

`PLAN_WEAK_MODEL_RESILIENCE.md` is fully implemented; nobody can see that from
outside.

- [ ] A reproducible benchmark: N standard tasks × M weak models (3B–8B local
  models), measuring tool-call success rate, rescue rate
  (`core/tool_call_rescue.py`), convergence-protocol correction rate, and task
  completion. Publish results in docs.
- [ ] Golden corpus of real malformed tool calls (collected from production
  logs, credential-redacted) as regression tests for rescue/NL-translation.
- [ ] Expand programmatic validators in the Convergence Protocol wherever a
  check can be deterministic instead of LLM-based — cheaper and more reliable,
  especially with weak CP models.
- [ ] A minimal `wintermute` CLI for running the eval harness and inspecting
  status/threads/memories from the terminal. This is ops tooling for the
  benchmark work above, not an interface play — the chat surface stays
  Signal/Matrix/web.

### 2d. Honest naming

Do not adopt hermes's vocabulary. Wintermute has outcome feedback and memory
consolidation, not "self-improvement" or a "learning loop." Docs should
describe mechanisms, not aspirations. This is a differentiator: the project
that underclaims and overdelivers stands out in this space.

## Phase 3 — Structural cleanup (only after Phase 0 tests exist)

- [ ] Split the god files along existing seam lines:
  - `core/sub_session.py` (2168 LOC) → DAG engine / worker lifecycle / scratchpad
  - `workers/dreaming.py` (1674 LOC) → housekeeping phases / creative phases /
    quality gating
  - `interfaces/matrix_thread.py` (1674 LOC) → E2E crypto / event handling /
    command routing
  - `core/convergence_protocol.py` (1667 LOC) → detection / validation /
    correction / scope
- [ ] Consolidate prompt text: `data/prompts/` already exists; move remaining
  inline prompts there so prompt changes are reviewable diffs.
- [ ] Keep `data/CONVERGENCE_PROTOCOL_HOOKS.txt` honest — it's an empty `[]`
  while built-in defaults carry the load; either document that or remove the
  file.

## Explicit non-goals (dropping the bullshit)

Learned from hermes's repo audit. Do not do any of these, ever:

- **No ecosystem interop play** (MCP servers, ACP adapters, cross-harness
  skill formats). That's hermes's game — breadth of surface area — and we can
  only lose it. Wintermute's chat surface is Signal/Matrix/web by decision,
  not by omission.
- **No vendored marketing website** in the repo. Docs live in `docs/`.
- **No Electron/desktop app, no TUI.** Web interface is sufficient.
- **No salvaged skill bundles** (hermes ships 182, many lifted from
  claude-code/codex/opencode ecosystems). Skills must be earned by the agent's
  own outcomes or written deliberately.
- **No telemetry of any kind**, including "pseudonymous install metrics" —
  hermes's "no telemetry" claim now carries an asterisk for their hosted
  relay. Wintermute's answer: nothing to asterisk.
- **No "learning loop" marketing language.** See 2d.
- **No god-file growth.** New subsystems start as packages, not 2000-line
  modules. The existing splits happen in Phase 3, not opportunistically.
- **No fine-tuning claims.** If trajectory export for external training is
  ever added (hermes's `trace_upload.py` pattern), it is explicit, opt-in,
  offline, and credential-redacted — and documented as data export, not as
  the agent learning.
- **No dependency bloat.** 17 direct deps today; keep it near that. Optional
  integrations (Qdrant already is one) stay optional. Adopt hermes's one good
  habit here: pin versions with a written rationale.

## Issue backlog (prioritized)

All open issues were audited against the code on 2026-08-05 and tagged with
`criticality`, `effort`, and `benefit` labels (high/medium/low each). This
ordering maps them onto the roadmap phases. Two issues were closed in the
audit: #58 (session auto-reset — implemented, notification gap noted) and #122
(L4 autonomy meta-plan — superseded by this document).

**Do first — high benefit, low effort:**

- **#236** Remove skill documentation condensation from dreaming
  (`criticality: high`) — the only actively harmful open issue: nightly
  LLM-rewrites of skill docs cause compounding information loss for zero
  prompt-size benefit. Fix is deletion. Precedes Phase 2b (dreaming quality
  work assumes the pipeline stops damaging its inputs).
- **#268** Hermes Sessions API compatibility (sync `/chat`, `POST
  /api/sessions`, bearer auth) — unblocks external clients (hermes-pebble);
  the issue contains near-complete implementation code.
- **#180** Local embedding fallback — memory currently hard-fails at startup
  without a configured endpoint; zero-config memory via an optional
  onnxruntime extra. Fits Phase 1 (store hardening).

**Strategic — Phase 2 alignment, high effort:**

- **#248** Confidence scores on memories — the highest-value open feature;
  implements the roadmap's outcome→memory-quality loop (2a/2b). Schema
  migration + reflection integration + confidence-weighted ranking.
- **#247** Episodic/semantic memory separation — same direction; design
  should follow #248 rather than precede it.

**When convenient:**

- **#42** Dynamic backend selection by historical success — data
  infrastructure exists (`get_outcome_stats()`), routing untouched. Only
  relevant with 3+ backends, per its own prerequisite.
- **#178** Progressive tool disclosure — fits weak-model resilience, but
  needs evaluation against CP-hook interference first (noted in the issue).
- **#207** Consolidate interface ACL/dispatch — rated above the other
  refactors because duplicated auth-path logic is a drift risk, not just
  aesthetics. Natural fit for Phase 3.
- **#259** Config hot-reload — high effort (per-component `reconfigure()`)
  for a gain `restart_self` mostly covers.

**Mechanical refactors — good first contributions, no urgency:**

- #210 (`tool_error()` helper; grew from 42 to 47 instances), #209 (Qdrant
  payload index dedup), #208 (SQLite connect dedup), #206
  (`retry_with_backoff`), #205 (OAuth refresh mixin), #204 (credential I/O
  util). All confirmed still duplicated as of the audit. Bundle with Phase 3
  or pick off individually; none should land before Phase 0 tests exist.

**Deferred:**

- **#249** Counterfactual dreaming phase — `benefit: low` on purpose: the
  roadmap requires proving dreaming's value with the eval harness (2b) before
  adding more speculative phases to it.

## Sequencing rationale

Phase 0 first because nothing else is safe to change without tests. Phase 1 is
low-risk, high-value engineering with a proven template. Phase 2 is the actual
strategy — converting existing architectural leads into demonstrable ones —
and depends on Phase 1's search/portability for the eval harness. Phase 3 is
pure maintainability and is gated on Phase 0 so the splits are verified.

Success metric for the whole roadmap: a published benchmark where wintermute,
running a weak local model, completes tasks that a stock frontier-model
harness fails on malformed-output grounds — with the tests to prove the
machinery behind it works.
