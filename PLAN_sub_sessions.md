# Implementation Plan: Agentic Sub-sessions

## Goal
Replace the "default thread" catch-all with isolated, ephemeral sub-sessions that run
background tasks without polluting any user-facing conversation thread. The main thread
(orchestrator) can spawn a worker, remain responsive, and receive a structured result
back as a system event once the worker is done.

---

## Step 1 — Core infrastructure: `sub_session.py` + `tools.py`

**New file: `ganglion/sub_session.py`**

- `SubSessionState` dataclass:
  ```
  session_id, objective, parent_thread_id, status, created_at,
  completed_at, result, error
  ```
- `SubSessionManager`:
  - Holds: `AsyncOpenAI` client, `enqueue_system_event` callback, active task/state dicts
  - `spawn(objective, context_blobs, parent_thread_id, system_prompt_mode, timeout)` → session_id
    - `system_prompt_mode`: `"full"` | `"base_only"` | `"none"` (default `"base_only"`)
    - Creates asyncio.Task running `_worker_loop()`
  - `cancel_for_thread(thread_id)` — cancels sub-sessions whose parent is `thread_id`
  - `list_active()` → list of SubSessionState dicts
  - `_worker_loop()` — inference loop identical to LLMThread._inference_loop but:
    - In-memory message list (no DB reads/writes)
    - Focused system prompt (not full assembled prompt unless mode="full")
    - Tool access identical except `spawn_sub_session` is blocked (depth limit)
    - On completion: calls `enqueue_system_event("[SUB-SESSION {id} RESULT]\n{summary}", parent_thread_id)`
    - On timeout/failure: reports back similarly
    - If `parent_thread_id` is None: logs only (fire-and-forget for heartbeats/system reminders)

**Changes to `ganglion/tools.py`**

- Add `spawn_sub_session` tool schema:
  ```json
  {
    "name": "spawn_sub_session",
    "description": "Spawn an isolated background worker for a complex task. Returns immediately
                    with a session_id. The result is delivered back to this thread when done.",
    "parameters": {
      "objective": "string — the full task description for the worker",
      "context_blobs": "array of strings — relevant context snippets to pass (memories excerpt,
                        file contents, etc.)",
      "system_prompt_mode": "enum: full | base_only | none (default base_only)"
    }
  }
  ```
- Add `register_sub_session_manager(fn)` + `_tool_spawn_sub_session(inputs, thread_id, in_sub_session)`
  - Returns `{"error": "..."}` if called from within a sub-session (`in_sub_session=True`)
- `execute_tool()` gains `in_sub_session: bool = False` parameter passed through to the handler

---

## Step 2 — `llm_thread.py` integration

**Changes to `LLMThread`:**

- Constructor accepts optional `sub_session_manager: SubSessionManager`
- `execute_tool` calls in `_inference_loop` pass `in_sub_session` flag (False for main threads)
- `reset_session()` calls `sub_session_manager.cancel_for_thread(thread_id)` if manager is set
- Remove the `thread_id or "default"` fallback from system event routing; system events with
  no thread_id now raise a clear error (they should all be converted to sub-sessions by Step 3)

**No change to the queue model** — the queue remains sequential per-thread; sub-sessions are
separate Tasks and do not go through the queue.

---

## Step 3 — Fix the "default thread" muddle: heartbeat + scheduler + main.py

**`ganglion/heartbeat.py`:**
- `_review_global()`: replace `_llm_enqueue(HEARTBEAT_REVIEW_PROMPT, "default")` with
  `sub_session_manager.spawn(objective=HEARTBEAT_REVIEW_PROMPT, parent_thread_id=None,
                              system_prompt_mode="full")`
  (Global heartbeat needs full context — memories + heartbeats — to be useful.)
- `_review_per_thread()`: unchanged — already correctly uses per-thread enqueue with delivery

**`ganglion/scheduler_thread.py`:**
- `_fire_reminder()` system path (ai_prompt + no thread_id): replace
  `_llm_enqueue(ai_prompt, "default")` with
  `sub_session_manager.spawn(objective=ai_prompt, parent_thread_id=None,
                              system_prompt_mode="base_only")`

**`ganglion/main.py`:**
- Instantiate `SubSessionManager` after `LLMThread` is built (shares the same `AsyncOpenAI` client)
- Inject into: `LLMThread`, `HeartbeatLoop`, `ReminderScheduler`
- Register spawn function into tools module via `register_sub_session_manager`

**"default" thread status after Step 3:**
- Nothing writes to it anymore
- DB schema untouched (backwards compat)
- `get_active_thread_ids()` will not return it for new installs
- No code removal needed — it simply becomes unused
