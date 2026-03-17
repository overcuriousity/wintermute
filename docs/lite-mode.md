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
