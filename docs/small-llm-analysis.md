# Small/Weak LLM Compatibility Analysis

> Audit date: 2026-02-19
> Scope: Full codebase review — architecture, prompts, tools, inference loop, sub-sessions, Turing Protocol

## Executive Summary

Wintermute's architecture has good foundations for small model support (role-based backend routing, tool category filtering, NL translation, programmatic validators). The gaps are mostly about **defaults** (too permissive for small models), **missing guard rails** (no loop limits, no JSON error feedback), and **tool schema design** (optimized for strong models that handle conditional logic well).

---

## P0 — Critical Blockers

### 1. Tool Schema Complexity

**Files:** `tools.py:68-349`

The main agent receives **all 12 tools at once** (`llm_thread.py:662`). The three most problematic:

- **`set_reminder`** (10 properties): Conditional-required fields based on `schedule_type` value. A state machine disguised as a flat schema. Small models mix up ISO-8601 vs HH:MM formats, forget conditionally-required fields, hallucinate enum values.

- **`spawn_sub_session`** (7 properties): Arrays of session IDs for `depends_on`, 4-value enum for `system_prompt_mode`, ISO-8601 strings. Requires cross-turn working memory for session ID tracking.

- **`pulse`** (6 properties): Action-dispatch pattern where `action` determines which other params are required.

**NL translation** (`nl_translator.py`) exists for `set_reminder` and `spawn_sub_session` but is **disabled by default**.

**Recommendations:**
- Enable NL translation by default when model is small
- Extend NL translation to `pulse`
- Consider splitting `set_reminder` into per-schedule-type tools
- Progressive tool disclosure: start with 5-6 core tools, expand on demand

### 2. No Inference Loop Max-Rounds Limit

**File:** `llm_thread.py:641-868`

The inference loop is `while True` with no cap. A small model stuck in a tool-call loop burns unlimited tokens. Malformed JSON arguments silently become `{}` (line 719-721) with no explicit error feedback to help the model self-correct.

**Recommendations:**
- Add configurable `max_inference_rounds` (8-10 for main, 5 for sub-sessions)
- Detect repetitive tool calls (same tool + same args N times) and break
- On JSON parse failure, inject explicit error with the raw text so model can fix syntax

---

## P1 — High Impact

### 3. System Prompt Size Defaults

**Files:** `prompt_assembler.py:17-19`, `data/prompts/BASE_PROMPT.txt`

Component size limits default to MEMORIES=10K, PULSE=5K, SKILLS=20K. Worst-case system prompt: ~37KB. On a 32K context with 1024 max_tokens, this leaves ~60% for conversation before tool schemas are even counted.

BASE_PROMPT.txt packs dense multi-concept instructions (two-layer memory, DAG planning, reminder semantics, system events, abstract prohibitions) — each concept requires working memory that 8B models don't have.

**Recommendations:**
- Lower default limits for small models (3K/1.5K/5K)
- Create a "lite" BASE_PROMPT variant without DAG planning, `not_before`, `background=true` nuances
- Consider injecting memories/skills only on the first round of multi-round tool loops

### 4. NL Translation Disabled by Default

**File:** `config.yaml.example` line 106

The primary mitigation for tool schema complexity is opt-in. Small model users must know to enable it.

**Recommendation:** Auto-enable when model name or context_size suggests a small model.

---

## P2 — Medium Impact

### 5. Turing Protocol Call Multiplication

**Files:** `llm_thread.py:870-912`, `turing_protocol.py`

A single sub-session tool round can trigger 4-5 LLM calls (inference + pre_execution detection + post_execution detection + objective_completion evaluation). With 3-4 tool rounds per sub-session, this means 12-20+ calls.

The `objective_completion` evaluator requires strict JSON output from a small model — and charitably assumes "met" on parse failure, making it unreliable.

The **programmatic validators** (`turing_protocol.py:405-555`) are excellent and don't need LLM calls. They catch: hallucinated spawns, phantom tool results, empty promises, schema violations, premature pulse completion.

**Recommendations:**
- Lean harder into programmatic validators, fewer LLM-based validators
- Make `objective_completion` programmatic for sub-sessions
- Add a "lightweight Turing" mode: programmatic Stage 2 only, skip Stage 1 LLM detection

### 6. Sub-Session Continuation Overhead

**Files:** `sub_session.py:792-852`

Continuations pass full prior message history (up to 3 hops). Small models must parse all prior messages and understand they're continuing.

**Recommendations:**
- For small models, pass structured summary instead of full history
- Cap continuation depth at 1 for small models

---

## P3 — Lower Impact

### 7. Token Counting Inaccuracy

**File:** `llm_thread.py:43-57`

Uses tiktoken with `cl100k_base` fallback for unknown models. Ministral, Qwen, etc. get GPT-4's tokenizer, causing overestimation and premature compaction.

**Recommendation:** Allow configuring tokenizer per backend, or at minimum a `tokens_per_char` ratio.

### 8. Sparse "minimal" Worker Prompt

**File:** `data/prompts/WORKER_MINIMAL.txt`

Only ~170 chars — too sparse for small models that need more scaffolding. The gap between "minimal" and "base_only" (full BASE_PROMPT) is too large.

**Recommendation:** Create an intermediate "small_model" prompt mode with concise essential instructions.

---

## What's Already Working Well

- **Role-based backend routing**: Different models for compaction, sub-sessions, dreaming, Turing
- **Tool category filtering**: Sub-sessions in "minimal" mode get only 5 tools
- **Context compaction**: Automatic summarization with chained summaries
- **Programmatic validators**: Catch hallucination patterns without LLM calls
- **`depends_on_previous` flag**: Avoids session ID tracking for simple sequential workflows
- **Component size monitoring**: Auto-triggers summarization when limits exceeded
- **Tool result trimming**: Automatic truncation of old tool results per round
