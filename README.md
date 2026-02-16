# Wintermute

![wintermute](static/Gemini_Generated_Image_50dxia50dxia50dx.png)

> *"Wintermute was hive mind, decision maker, effecting change in the world outside."*
> — William Gibson, *Neuromancer* (1984)

**Wintermute** is a self-hosted personal AI assistant with persistent memory, autonomous background workers, and multi-interface support. It connects to any OpenAI-compatible LLM endpoint and reaches you via Matrix chat or a built-in web UI.

---

## Concept

Wintermute accumulates knowledge about you over time, maintains an active working memory (*Pulse*), and learns reusable procedures as *skills*. Conversations across restarts are summarised and retained. A nightly *dreaming* pass consolidates memories autonomously while you sleep — no human required.

For long-running or complex tasks, Wintermute spawns isolated background workers (*sub-sessions*) so the main conversation stays responsive. Multi-step requests are expressed as workflow DAGs: the orchestrator defines all stages upfront with `depends_on` dependencies, and downstream tasks auto-start when their prerequisites complete — no human nudging required. Workers can themselves spawn further workers up to a configurable nesting depth.

The philosophy differs from similar projects by treating small LLMs and digital independence not as an afterthought, but as a first principle. No cloud services. No telemetry. It runs on your hardware, speaks to your endpoints, and answers to you.

---

## Features

- **Persistent memory** — `MEMORIES.txt` (long-term facts, append-based), pulse items in SQLite (active goals / working memory with priorities), and `skills/*.md` (reusable procedures) survive restarts and are injected into every prompt
- **Multi-interface** — Matrix chat (with E2E encryption) and a browser-based web UI run simultaneously; each room / tab has independent conversation history
- **Sub-session workers** — long-running tasks are delegated to autonomous background agents that report back when done; the main agent stays responsive during execution; workers auto-resume after timeouts (up to 3 hops)
- **Workflow DAG** — multi-step tasks are expressed as dependency graphs via `depends_on`; downstream tasks auto-start when their dependencies complete, with results passed as context — no LLM decision-making needed after the initial plan; tasks can include `not_before` time gates for scheduled execution ("research now, upload after 20:00")
- **Tool-filtered workers** — minimal workers receive only execution + research tools; `full`-mode workers get orchestration tools too, keeping context lean
- **Web search** — `search_web` queries a local SearXNG instance and falls back to DuckDuckGo via `curl` when SearXNG is unavailable; `fetch_url` fetches and strips any web page
- **Reminders & scheduler** — one-time and recurring reminders with optional AI inference on trigger; per-timezone scheduling
- **Nightly dreaming** — automatic overnight consolidation of MEMORIES.txt and pulse items via a direct LLM call (no tool loop, no conversation side effects)
- **Pulse reviews** — periodic autonomous reviews of active pulse items via an isolated sub-session (no conversation pollution)
- **Context compaction** — when conversation history approaches the model's context window, older messages are summarised and chained into a rolling summary that preserves context across compaction cycles
- **Debug panel** — `http://localhost:8080/debug` provides a live view of sessions, sub-sessions, scheduled jobs, reminders, and the current system prompt
- **Any OpenAI-compatible backend** — Ollama, vLLM, LM Studio, OpenAI, Kimi-Code, or any compatible endpoint

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
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
bash setup.sh
```

The setup script installs all dependencies, walks you through configuration (LLM endpoint, Matrix, timezone), installs a systemd user service, runs pre-flight checks, and offers to start the daemon immediately. One command, working service.

See [docs/installation.md](docs/installation.md) for manual installation, setup script options (`--dry-run`, `--no-matrix`, etc.), and more.

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
| [Architecture](docs/architecture.md) | Component overview, diagrams, data flow |
| [System Prompts](docs/system-prompts.md) | Prompt assembly, components, size limits |
| [Tools](docs/tools.md) | All 12 tools with parameters and categories |
| [Commands](docs/commands.md) | Slash commands (`/new`, `/compact`, `/pulse`, etc.) |
| [Web Interface](docs/web-interface.md) | Chat UI, debug panel, REST API |
| [Autonomy](docs/autonomy.md) | Dreaming, pulse reviews, sub-sessions, workflows |

---

## License

MIT
