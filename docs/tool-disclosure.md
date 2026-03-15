# Tool Disclosure (Progressive Tool Exposure)

By default, all 12 tool schemas are sent with every LLM turn (~1000-1500 tokens). This can overwhelm small models (< 14B parameters) and waste context budget. Tool disclosure classifies user intent each turn and only exposes relevant tool tiers.

## How It Works

1. **Startup**: Tier label embeddings are pre-computed using the existing `memory.embeddings` endpoint.
2. **Per-turn**: The user message is embedded and compared (cosine similarity) against tier labels.
3. **Selection**: Tiers whose similarity exceeds the threshold are included. Tier 0 is always included.
4. **Fallback**: If the embeddings endpoint is unavailable or any error occurs, all tools are exposed (generous default).

The classifier is language-agnostic — it uses semantic similarity, not keyword matching.

## Tiers

| Tier | Tools | When Included |
|------|-------|---------------|
| 0 (always) | `append_memory`, `read_file`, `write_file` | Every turn |
| 1 (research/execution) | `search_web`, `fetch_url`, `execute_shell`, `send_file` | User intent suggests research or action |
| 2 (orchestration) | `worker_delegation`, `task`, `skill`, `query_telemetry`, `restart_self` | Complex, multi-step, or delegation requests |

## Configuration

```yaml
tool_disclosure:
  enabled: false              # Default off for backward compatibility
  similarity_threshold: 0.3   # Cosine similarity threshold (lower = more generous)
  always_include_delegation: true  # Always include worker_delegation (for CP inline_tool_limit)
```

No additional backends are needed — tool disclosure reuses the `memory.embeddings` configuration.

## Convergence Protocol Interaction

The `inline_tool_limit` hook (pre_execution, main scope) injects a correction telling the LLM to use `worker_delegation`. When `always_include_delegation: true` (default), `worker_delegation` is always in the disclosed set regardless of classifier output, ensuring the CP correction is actionable.

## Tuning the Threshold

- **0.2** — Very generous: most tiers included most of the time. Minimal token savings but safest.
- **0.3** — Default: good balance between token savings and tool availability.
- **0.5** — Aggressive: significant token savings but may miss relevant tools for ambiguous messages.

Start with the default and lower it if tools are missing when needed. The debug panel (`/debug`) shows which tools were disclosed each turn.
