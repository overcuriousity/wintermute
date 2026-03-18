# Lite Mode

Lite mode disables `worker_delegation` (background sub-session workers) globally to reduce LLM API calls. This is useful for local or small models where spawned workflows are the biggest source of inference cost.

## What it does

When `sub_sessions_enabled` is set to `false`:

- The `worker_delegation` tool schema is removed from the tool list sent to the LLM
- System prompt sections that require `worker_delegation` are excluded
- The LLM cannot spawn background workers or workflows

## How to enable

Set the top-level key in `config.yaml`:

```yaml
sub_sessions_enabled: false
```

On startup, Wintermute logs:

```
sub_sessions_enabled=false (lite mode) — worker_delegation disabled globally
```

## Per-thread override

Individual threads can re-enable sub-sessions even when the global default is `false`:

```
/config sub_sessions_enabled true
```

This uses the three-layer resolution chain: per-thread override > global config default > hardcoded default (`true`).

## What's lost

- No background workers or multi-step workflows
- No DAG-based task execution via `worker_delegation`
- The LLM must handle all work inline in the main conversation thread

All other features (memory harvest, dreaming, reflection, scheduled tasks) continue to work normally.

## Inline tool limit in lite mode

The `inline_tool_limit` Convergence Protocol hook remains active in lite mode. When the model exceeds `max_inline_tool_rounds` tool calls in a turn, it is instructed to present its findings as-is and transparently suggest the user split the remaining work into smaller subtasks — rather than attempting to delegate to an unavailable worker.

Since there is no worker delegation to offload to, consider setting `max_inline_tool_rounds` higher (e.g. 8–12) in lite mode to give the model more room per turn:

```yaml
tuning:
  max_inline_tool_rounds: 10
```

Wintermute lite mode is best suited for focused, single-step tasks. Complex multi-step work (large refactors, continuous coding across many files) should be broken into smaller requests by the user.
