# Best Practices

Operational notes for running Wintermute reliably and safely.

---

## Isolation: Run in an LXC Container or VM

Wintermute runs with the full permissions of its user, has unrestricted shell access, and stores credentials in plaintext. **Do not run it on a machine that holds data you care about.**

The recommended deployment is a dedicated LXC unprivileged container or a lightweight VM:

- Reset or rebuild without regret if something goes wrong
- Prevents lateral movement if the LLM is manipulated into executing destructive commands
- Keeps credentials (Matrix token, API keys) isolated from your workstation
- Easy to snapshot before major config changes

If you use LXC, an unprivileged container with a dedicated UID mapping is sufficient. Allocate enough disk for logs, the SQLite database, and any files the model might write during tool use.

---

## Model Selection

Wintermute is designed to work with small models. A practical multi-model setup:

| Role | Recommended size | Notes |
|------|-----------------|-------|
| `base` (main conversation) | 7B–14B | Primary model; drives most behaviour |
| `sub_sessions` | 7B–14B | Can match base or be smaller for mechanical tasks |
| `compaction` | 3B–7B | Summarisation only; does not need to reason |
| `dreaming` | 3B–7B | Direct LLM call, no tool loop |
| `turing_protocol` | 3B–7B | Fast detection; small model preferred |

Route the Turing Protocol and compaction to your smallest/fastest model. They don't need broad knowledge — just reliable instruction-following.

---

## Turing Protocol: Enable It

Enable the Turing Protocol from the start, especially with smaller models. The three failure modes it catches (hallucinated actions, phantom tool results, empty promises) become more frequent as model size decreases, and they silently corrupt the conversation state if uncorrected.

Recommended minimal config:

```yaml
turing_protocol:
  backends: ["your_small_model"]
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
    objective_completion:
      enabled: true
      scope: "sub_session"
```

`tool_schema_validation` and `agenda_complete` are always-on and require no configuration.

---

## Web Search: Run a Local SearXNG

The `search_web` tool queries a local [SearXNG](https://docs.searxng.org/) instance by default and falls back to DuckDuckGo via `curl` when unavailable. Running your own SearXNG:

- Eliminates rate limits that break the DuckDuckGo fallback
- Keeps search queries off third-party infrastructure
- Configurable result engines and filters

SearXNG can run in a separate container on the same host and is reachable via `http://localhost:8888` (or whichever port you configure).

---

## Memory Hygiene

The system auto-summarises memory components when they exceed their size limits, but deliberate maintenance keeps the model better oriented:

- **MEMORIES.txt**: Keep entries factual and specific. Avoid storing transient state here — use Agenda for that. The nightly dreaming pass deduplicates and merges, but only as well as the model can reason about your entries.
- **Agenda**: Complete items promptly when done. Stale agenda items consume prompt tokens on every turn and confuse the model about what is still active.
- **Skills**: Keep skill files focused on one procedure. Large or overly broad skill files are harder for small models to apply correctly.

---

## Systemd Service

The `setup.sh` script installs a systemd user service. Key operational notes:

- Logs: `journalctl --user -u wintermute -f`
- Restart on failure: `systemctl --user edit wintermute` and add `Restart=on-failure`
- The service runs under your user UID — no root required
- `data/` is relative to the working directory set in the service file; check `WorkingDirectory=` if paths break after moving the install

---

## Credentials

All credentials live in `config.yaml` (gitignored). Additional credential files:

| File | Contents |
|------|----------|
| `data/kimi_credentials.json` | Kimi-Code OAuth tokens |
| `data/matrix_crypto.db` | Matrix E2E encryption keys |
| `data/matrix_recovery.key` | Matrix cross-signing recovery key |

Back up `data/` regularly. Loss of `matrix_crypto.db` breaks E2E encryption for existing Matrix sessions and requires re-verification.
