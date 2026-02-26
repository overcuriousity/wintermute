# Wintermute

![wintermute](static/Gemini_Generated_Image_50dxia50dxia50dx.png)

> *"Wintermute was hive mind, decision maker, effecting change in the world outside."*
> — William Gibson, *Neuromancer* (1984)

**Wintermute** is a self-hosted personal AI assistant with persistent memory, autonomous background workers, and multi-interface support. It connects to any OpenAI-compatible LLM endpoint or Kimi-Code, and reaches you via Matrix chat. All inference, tool calls, and validation decisions are logged with the goal of forensic-grade auditability.

---

## Concept

Wintermute accumulates knowledge about you over time, maintains active *tasks* (goals, reminders, scheduled actions), and learns reusable procedures as *skills*. A nightly *dreaming* pass consolidates memories autonomously.

For long-running or complex tasks, Wintermute spawns isolated background workers (*sub-sessions*) so the main conversation stays responsive. Multi-step requests are expressed as workflow DAGs: the orchestrator defines all stages upfront with `depends_on` dependencies, and downstream tasks auto-start when their prerequisites complete — no human nudging required. Workers can themselves spawn further workers up to a configurable nesting depth.

The design philosophy treats small LLMs and digital independence not as afterthoughts, but as first principles. No cloud services. No telemetry. It runs on your hardware, speaks to your endpoints, and answers to you.

Two architectural choices make this concrete:

- **No framework abstraction layer.** Tool calls use the OpenAI function-calling wire format directly — no LangChain, no LlamaIndex, no hidden prompt rewriting. What the model receives is exactly what you configure.
- **Turing Protocol.** A three-stage (detect → validate → correct) post-inference validation pipeline that automatically catches and self-corrects the hallucination patterns small models are most prone to: claiming to have done things they didn't, fabricating tool output, or making promises without acting. No human in the loop required.

---

## Features

- **Persistent memory** — `MEMORIES.txt` (long-term facts, append-based), tasks in SQLite (active goals / working memory with priorities and optional schedules), and `skills/*.md` (reusable procedures) survive restarts and are injected into every prompt
- **Sub-session workers** — long-running tasks are delegated to autonomous background agents that report back when done; the main agent stays responsive during execution; workers auto-resume after timeouts (nesting possible)
- **Workflow DAG** — multi-step tasks are expressed as dependency graphs via `depends_on`; downstream tasks auto-start when their dependencies complete, with results passed as context
- **Tool-filtered workers** — minimal workers receive only execution + research tools; `full`-mode workers get orchestration tools too. Named tool profiles (`researcher`, `file_worker`, etc.) and conditional system prompt sections keep context lean
- **Web search** — `search_web` queries a local SearXNG instance and falls back to DuckDuckGo when SearXNG is unavailable
- **Task scheduler** — one-time and recurring tasks with optional AI inference on trigger; per-timezone scheduling
- **Introspection** — the LLM can query its own operational telemetry (success rates, tool usage, skill stats, interaction logs, self-model summary) via the `query_telemetry` tool
- **Skill evolution** — skills track usage stats (read counts, session outcomes, failure rates); unused skills are auto-retired during nightly dreaming; the reflection cycle correlates skill usage with failures, recommends updates, and synthesizes new skills from recurring successful patterns
- **Nightly dreaming** — automatic overnight consolidation of MEMORIES.txt and tasks via a direct LLM call (no tool loop, no conversation side effects)
- **Task reviews** — periodic autonomous reviews of active tasks via an isolated sub-session (no conversation pollution)
- **Context compaction** — when conversation history approaches the model's context window, older messages are summarised and chained into a rolling summary that preserves context across compaction cycles
- **Turing Protocol** — three-stage validation pipeline (detect → validate → correct) that automatically corrects hallucinations and unfulfilled commitments
- **Audit trail** — every inference call, tool execution, and Turing Protocol decision is logged to SQLite. A web interface provides a live inspection panel for sessions, sub-sessions, jobs, tasks, and assembled system prompts
- **Any OpenAI-compatible backend** — llama-server, vLLM, LM Studio, OpenAI or any compatible endpoint. Working towards integrating subscription-based providers, kimi-code currently functional

---

## Requirements

- Linux (Fedora / RHEL or Debian / Ubuntu)
- `bash` and `curl` — everything else is installed automatically
- An OpenAI-compatible LLM endpoint
- *(Strongly recommended)* A local [SearXNG](https://docs.searxng.org/) instance for web search
- *(Recommended)* A dedicated Matrix account for the bot

---

## Quickstart

```bash
git clone https://github.com/overcuriousity/wintermute.git wintermute
cd wintermute
bash onboarding.sh
```

The onboarding script installs all system dependencies, then hands off to an **AI-powered configuration assistant** that uses your own LLM endpoint to walk you through every `config.yaml` option interactively. It tests your endpoints, validates Matrix credentials, runs OAuth flows, and installs the systemd service — all through a conversational interface.

> **Experimental:** The AI-driven onboarding (`onboarding.sh`) is new and requires a model with function-calling support. The previous programmatic setup script (`setup.sh`) is retained as a fallback.

See [docs/installation.md](docs/installation.md) for manual installation and the classic setup script

---

## Security Disclaimer

> *"The Turing Registry exists for a reason."*

Wintermute runs with the full permissions of the user that starts it. It has unrestricted shell access. It will read your files, execute commands, speak in your voice, and reach into the systems around it. That is the point — and the risk.

**Do not run this on your personal workstation, or any machine that holds data you care about.**

Credentials (API keys, Matrix tokens) are stored in plain text in `config.yaml`. Any model you connect to will see everything you tell Wintermute. The host machine should be treated as potentially compromised from the moment Wintermute is installed.

Run it in a dedicated LXC container or VM — something you can reset without regret.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](docs/installation.md) | Quickstart, manual setup, systemd service |
| [Configuration](docs/configuration.md) | Full `config.yaml` reference |
| [Matrix Setup](docs/matrix-setup.md) | Account creation, credentials, E2E encryption, troubleshooting |
| [Architecture](docs/architecture.md) | Component overview, diagrams, data flow, small-LLM design |
| [Turing Protocol](docs/turing-protocol.md) | Validation pipeline: hooks, phases, scopes, configuration |
| [System Prompts](docs/system-prompts.md) | Prompt assembly, components, size limits |
| [Tools](docs/tools.md) | All 13 tools with parameters and categories |
| [Commands](docs/commands.md) | Slash commands (`/new`, `/compact`, `/tasks`, etc.) |
| [Web Interface](docs/web-interface.md) | Debug panel, REST API |
| [Autonomy](docs/autonomy.md) | Dreaming, task reviews, sub-sessions, workflows |
| [Best Practices](docs/best-practices.md) | Deployment, model selection, security |

---

## License

GPL

---

## Contributions

All contributions welcome. Inclusive to AI agents.
