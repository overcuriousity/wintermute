# Sub-Session Outcome Tracking & Historical Feedback

## Overview

Sub-session outcomes are tracked in the database so Wintermute can learn from past task execution. Before spawning a new sub-session, the system queries for similar past objectives and injects historical feedback (durations, success rates, tool call counts) as context — helping the worker avoid repeating mistakes like using too-short timeouts for research tasks.

## Database Schema

The `sub_session_outcomes` table in `data/conversation.db`:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT | Sub-session identifier |
| `workflow_id` | TEXT | Parent workflow ID (nullable) |
| `timestamp` | REAL | Unix timestamp of outcome |
| `objective` | TEXT | Task objective text |
| `system_prompt_mode` | TEXT | Prompt mode used (minimal/full/base_only/none) |
| `tools_available` | TEXT | JSON array of available tool names |
| `tools_used` | TEXT | JSON array of tools actually called |
| `tool_call_count` | INTEGER | Total number of tool calls made |
| `duration_seconds` | REAL | Wall-clock duration |
| `timeout_value` | INTEGER | Configured timeout in seconds |
| `turing_verdict` | TEXT | pass / fail / skipped |
| `status` | TEXT | completed / timeout / failed |
| `result_length` | INTEGER | Character length of result |
| `nesting_depth` | INTEGER | 1 = direct child, 2 = grandchild |
| `continuation_count` | INTEGER | Number of timeout continuations |
| `backend_used` | TEXT | LLM backend that served the session |
| `objective_embedding` | BLOB | Vector embedding for similarity search |

## What Gets Tracked

Outcomes are persisted at three exit points in the sub-session lifecycle:

1. **Completed** — Worker produced a final response and passed TP validation
2. **Timeout** — Worker exceeded its timeout budget (before continuation spawning)
3. **Failed** — Worker raised an unhandled exception

The Turing Protocol verdict (`pass`/`fail`/`skipped`) is captured from the `post_inference` phase evaluation.

## Historical Feedback

When `spawn_sub_session` is called, the system queries `sub_session_outcomes` for similar past objectives before creating the worker. If matches are found, a context blob is prepended:

```
[Historical Feedback] Similar past sub-sessions:
- "research X" (300s timeout): completed in 245s, 12 tool calls
- "research Y" (300s timeout): timeout in 300s, 8 tool calls, continued 1x
Average duration: 272s | Success rate: 60%
```

This blob is included in the worker's system prompt context, allowing the LLM to make informed decisions about pacing and tool usage.

### Similarity Search

Two search strategies are used:

1. **Vector similarity** (preferred): If a vector embedding backend is configured, objectives are embedded using the same endpoint as the memory store. Cosine similarity > 0.5 threshold.
2. **Keyword matching** (fallback): Splits the objective into significant words (>3 chars, excluding stopwords), matches with `LIKE` queries, ordered by recency.

## Configuration

No additional configuration is required. Outcome tracking is automatic once the system is running. Vector similarity search requires an active embedding backend (configured under `memory.embeddings` in `config.yaml`); otherwise keyword matching is used.
