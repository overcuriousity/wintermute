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

If you use LXC, an unprivileged container with a dedicated UID mapping is sufficient. Allocate enough disk for logs, the SQLite database, and any files the model might write during tool use. You do not need a full setup guide — the standard LXC or Proxmox workflow applies. The important principle is containment.

---

## Trust Model: Treat Wintermute Like an External Employee

Regardless of how capable Wintermute becomes, the underlying LLM is a probabilistic system with no verified identity and no accountability. Operate accordingly:

- **Never grant access to confidential data.** Wintermute should not be able to read your personal documents, financial records, password managers, private keys, or any data you would not hand to an untrusted contractor.
- **Never use your personal accounts on Wintermute's behalf.** Do not give it your personal email, your personal GitHub account, your personal Nextcloud login, or any account that carries your identity or has side effects you cannot trivially audit and reverse.
- **Assume prompt injection is possible.** Wintermute reads external content (web pages, files, Matrix messages from other users). Any of that content could attempt to manipulate its behaviour. The blast radius of a successful injection is bounded entirely by the permissions you grant.

This is not a statement about the quality of the model. It is a structural property of all LLM-based agents.

---

## Dedicated Accounts and Least-Privilege Access

When Wintermute needs to interact with a service — Nextcloud, Gitea, a home automation API, a mail relay — create a **fresh, dedicated account** for that purpose:

- Use a username that clearly identifies the account as belonging to Wintermute (e.g. `wintermute-bot`, `wm-gitea`)
- Set permissions to the minimum required for the task. Read-only where possible. Scoped API tokens rather than full credentials.
- Do not reuse tokens or passwords across services
- Revoke and rotate credentials without disrupting anything else when you need to

This applies equally to local self-hosted infrastructure. A Nextcloud account limited to one shared folder, a Gitea account with access to specific repositories only, or a Home Assistant long-lived token scoped to a single domain are all better than administrator access.

The goal is that if Wintermute is manipulated, misconfigured, or compromised, the damage is contained to what you explicitly scoped — not to everything the account could theoretically reach.

---

## Model Selection

Wintermute's role-based backend routing (see [Configuration](configuration.md)) lets you assign different models to different functions. The following tiers reflect a practical cost/capability tradeoff with specific model recommendations. All cloud options assume an OpenAI-compatible endpoint such as a Venice.ai proxy; local options assume a 16 GiB VRAM budget and a 32k context requirement.

---

### Tier 1 — Reasoning Core (`base`, `sub_sessions`)

**Responsibilities:** Main conversation loop, Turing Protocol evaluation, autonomous objective generation, sub-session orchestration.

**Requirement:** Deep reasoning, reliable multi-step planning, strong instruction adherence, and consistent persona retention across long sessions.

| Option | Model | Approx. price (in/out per 1M tokens) |
|--------|-------|--------------------------------------|
| Premium cloud | Claude Sonnet 4.6 | $3.75 / $18.75 |
| Budget cloud | Llama 3.3 70B | $0.70 / $2.80 |
| Budget cloud (alt) | GLM-4 | $1.00 / $3.20 |
| Local (≤16 GiB) | Nanbeige-4.1-3B | — |
| Local (alt) | Ministral-3-8B Reasoning (4-bit) | — |

**Notes:**

- Claude Sonnet 4.6 is the reference implementation. Anthropic's models lead on stateful agentic planning and persona retention; the Turing Protocol runs most reliably against them.
- Llama 3.3 70B and GLM-4 offer strong reasoning at substantially lower cost and are well-suited to driving the main loop without Anthropic pricing.
- For fully local deployments: Nanbeige-4.1-3B is an outlier — dual-stage RL training gives it reasoning performance disproportionate to its parameter count. Ministral-3-8B Reasoning is the alternative if you prefer a Mistral-family architecture.

---

### Tier 2 — Tool Execution (`sub_sessions` for mechanical tasks, `turing_protocol`)

**Responsibilities:** Function-calling loops, JSON schema adherence, tool dispatch, Turing Protocol detection passes.

**Requirement:** Strict JSON output, reliable schema compliance, fast inference. Reasoning depth matters less than syntax precision.

| Option | Model | Approx. price (in/out per 1M tokens) |
|--------|-------|--------------------------------------|
| Premium cloud | GPT-4.1 | $2.00 / $8.00 |
| Budget cloud | MiniMax M2.5 | $0.40 / $1.60 |
| Local (≤16 GiB) | Qwen3-4B-Instruct (8-bit or 16-bit) | — |

**Notes:**

- OpenAI's GPT-4.1 remains the most battle-tested model for raw tool-calling syntax and API interaction. Use it when correctness of shell commands or infrastructure mutations is non-negotiable.
- MiniMax M2.5 consistently tops multi-turn tool-calling benchmarks (BFCL) and represents exceptional value for high-volume agentic loops.
- Qwen models train heavily on tool-calling syntax. The 4B variant at 8-bit quantization fits comfortably within a 16 GiB budget while retaining syntax accuracy and leaving significant headroom for the context window KV cache.

---

### Tier 3 — Background Processing (`compaction`, `dreaming`, `memory_harvest`)

**Responsibilities:** Nightly memory consolidation, conversation summarisation, memory harvest extraction, context compaction.

**Requirement:** Efficient processing of large text volumes, reliable extraction and deduplication, long context handling. These are offline or background tasks; latency is not a concern.

| Option | Model | Approx. price (in/out per 1M tokens) |
|--------|-------|--------------------------------------|
| Premium cloud | Gemini 2.5 Pro | $2.50 / $15.00 |
| Budget cloud | Gemma 3 27B Instruct | $0.12 / $0.20 |
| Budget cloud (alt) | GPT-4o Mini | $0.15 / $0.60 |
| Local (≤16 GiB) | Gemma 3 4B (4-bit) | — |
| Local (alt) | Ministral-3-8B Instruct | — |

**Notes:**
- Gemini 2.5 Pro's extended context window makes it well-suited to ingesting a full day of interaction logs, conversation history, and accumulated memories in a single pass for the dreaming and harvest jobs.
- Gemma 3 27B and GPT-4o Mini both offer strong summarisation quality at near-negligible cost — overnight batch jobs here amount to a few cents.
- For local use: Gemma 3 4B at 4-bit quantization (~3.5 GB) is the only Gemma 3 variant that safely reaches 32k context on a 16 GiB card without an out-of-memory error. Ministral-3-8B Instruct is an efficient alternative with good long-context retention, well suited to offline compaction and summarisation.

---

## Turing Protocol: Enable It

Enable the Turing Protocol from the start, especially with smaller models. The three failure modes it catches — hallucinated actions, phantom tool results, and empty promises — become more frequent as model size decreases, and they silently corrupt conversation state if uncorrected.

**Use a plain instruct model for the Turing Protocol backend, not a reasoning model.** Reasoning models (those that emit a chain-of-thought scratchpad before responding) intro reduce significant latency on every validation pass and can produce verbose, unpredictable output that interferes with the structured detection pipeline. A small, fast instruct model — the same one you might use for compaction — is the correct choice here. The Turing Protocol does not require deep reasoning; it requires reliable instruction-following and consistent output format.  

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

`tool_schema_validation` and `task_complete` are always-on and require no configuration.

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

- **MEMORIES.txt**: Keep entries factual and specific. Avoid storing transient state here — use Tasks for that. The nightly dreaming pass deduplicates and merges, but only as well as the model can reason about your entries.
- **Tasks**: Complete tasks promptly when done. Stale tasks consume prompt tokens on every turn and confuse the model about what is still active.
- **Skills**: Keep skill files focused on one procedure. The first line of each skill file is used as a summary in the system prompt's TOC — make it clear and descriptive so the model knows when to load the full skill via `read_file`.

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
