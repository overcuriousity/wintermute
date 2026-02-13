# Wintermute

![Wintermute](static/Gemini_Generated_Image_7cdpwp7cdpwp7cdp.png)

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

- **Persistent memory** — `MEMORIES.txt` (long-term facts), `PULSE.txt` (active goals / working memory), and `skills/*.md` (reusable procedures) survive restarts and are injected into every prompt
- **Multi-interface** — Matrix chat (with E2E encryption) and a browser-based web UI run simultaneously; each room / tab has independent conversation history
- **Sub-session workers** — long-running tasks are delegated to autonomous background agents that report back when done; the main agent stays responsive during execution; workers auto-resume after timeouts (up to 3 hops)
- **Workflow DAG** — multi-step tasks are expressed as dependency graphs via `depends_on`; downstream tasks auto-start when their dependencies complete, with results passed as context — no LLM decision-making needed after the initial plan
- **Tool-filtered workers** — minimal workers receive only execution + research tools; `full`-mode workers get orchestration tools too, keeping context lean
- **Web search** — `search_web` queries a local SearXNG instance and falls back to DuckDuckGo via `curl` when SearXNG is unavailable; `fetch_url` fetches and strips any web page
- **Reminders & scheduler** — one-time and recurring reminders with optional AI inference on trigger; per-timezone scheduling
- **Nightly dreaming** — automatic overnight consolidation of MEMORIES.txt and PULSE.txt via a direct LLM call (no tool loop, no conversation side effects)
- **Pulse reviews** — periodic autonomous reviews of PULSE.txt; fires globally and per active conversation thread
- **Context compaction** — when conversation history approaches the model's context window, older messages are summarised and retained as a rolling summary
- **Debug panel** — `http://localhost:8080/debug` provides a live view of sessions, sub-sessions, scheduled jobs, reminders, and the current system prompt
- **Any OpenAI-compatible backend** — Ollama, vLLM, LM Studio, OpenAI, or any compatible endpoint

---

## Requirements

- Linux (Fedora / RHEL or Debian / Ubuntu)
- `bash` and `curl` — everything else is installed automatically
- An OpenAI-compatible LLM endpoint
- *(Recommended)* A dedicated Matrix account for the bot

---

## Installation

### Quickstart (recommended)

Clone the repository and run the interactive setup script — it handles everything:

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
bash setup.sh
```

`setup.sh` will:

1. Install Python 3.12+, `uv`, and all Python dependencies
2. Walk you through configuring `config.yaml` (LLM endpoint, Matrix credentials, timezone, …)
3. Optionally install a **systemd user service** so Wintermute starts on boot
4. Run pre-flight checks (endpoint reachability, package imports, …)

> **Note:** The script only runs on Fedora/RHEL or Debian/Ubuntu. It will exit on unsupported systems.

### Manual installation

<details>
<summary>Expand for manual steps</summary>

#### 1. Clone the repository

```bash
git clone https://git.mikoshi.de/overcuriousity/wintermute.git wintermute
cd wintermute
```

#### 2. Install with uv

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync
```

#### 3. Configure

```bash
cp config.yaml.example config.yaml
```

Open `config.yaml` and fill in at minimum the `llm` section:

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # or your local endpoint
  api_key: "sk-..."
  model: "gpt-4o"
  context_size: 128000
  max_tokens: 4096
```

Matrix and web sections are optional — if Matrix is omitted the web UI runs standalone.

#### 4. Run

```bash
uv run wintermute
```

The web interface starts at `http://127.0.0.1:8080` by default.

</details>

---

## Matrix Setup

### Create a dedicated Matrix account

Register a new account for the bot on your homeserver (e.g. via Element or the homeserver's registration page). The bot needs its own account — do not reuse your personal one.

### Configure credentials

There are two ways to provide Matrix credentials:

**Option A — Password (recommended).** Supply the bot's password and let Wintermute handle login, device creation, and token refresh automatically:

```yaml
matrix:
  homeserver: https://matrix.org
  user_id: "@your-bot-name:matrix.org"
  password: "bot-account-password"
  access_token: ""                    # auto-filled on first start
  device_id: ""                       # auto-filled on first start
  allowed_users:
    - "@you:matrix.org"
  allowed_rooms: []
```

On startup Wintermute logs in, writes the new `access_token` and `device_id` back into `config.yaml`, and refreshes them automatically if they expire.

**Option B — Manual token.** If you prefer not to store the password, obtain a token via curl and fill it in yourself:

```bash
curl -s -X POST 'https://matrix.org/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","identifier":{"type":"m.id.user","user":"@your-bot-name:matrix.org"},"password":"...","initial_device_display_name":"Wintermute"}' \
  | python3 -m json.tool
```

Copy `access_token` and `device_id` from the response into `config.yaml`. You will need to repeat this if the token expires.

### Invite the bot and start chatting

1. Start Wintermute: `uv run wintermute`
2. In Element (or any Matrix client), create a room or open a DM
3. Invite `@your-bot-name:matrix.org`
4. The bot joins and responds to messages from `allowed_users`

**End-to-end encryption** is handled automatically — the bot's crypto keys are persisted to `data/matrix_crypto.db` and the device is cross-signed at startup. The device fingerprint is logged on every start and retrievable via `/fingerprint` in the chat.

#### Cross-signing and device verification

On first start, Wintermute calls `generate_recovery_key()` to establish its cross-signing identity and saves the recovery key to `data/matrix_recovery.key`. On every subsequent start — including after the crypto store is wiped — it calls `verify_with_recovery_key()` to re-sign the current device using the stored key, with no browser interaction and no UIA approval required.

Wintermute implements the **m.sas.v1** (emoji) interactive verification protocol. To verify the device:

1. In Element go to **Settings → Security → Sessions**, select Wintermute's session, and tap **Verify Session**.
2. Element will start an emoji handshake. Wintermute auto-accepts from allowed users, skipping the emoji-comparison step.
3. After a moment the device shows a green shield (**Verified**) in Element.

Alternatively, send `/fingerprint` in a Matrix room to retrieve the Ed25519 key for manual out-of-band comparison.

### Troubleshooting

#### Token expired (`MUnknownToken`)

If `password` is set in `config.yaml`, Wintermute re-authenticates automatically — no action needed. Otherwise, Wintermute logs the exact `curl` command to obtain a new token. Run it, update `config.yaml`, and restart. Alternatively, add `password` to avoid this in the future.

#### Cross-signing requires approval (first run only)

On first start, some homeservers (including matrix.org) require you to approve the cross-signing key upload via your account page. Wintermute logs the exact URL:

```text
Cross-signing requires interactive approval from your homeserver.
  1. Open this URL in your browser: https://account.matrix.org/account/?action=org.matrix.cross_signing_reset
  2. Approve the cross-signing reset request.
  3. Restart Wintermute.
```

After approval, restart once. The recovery key is saved to `data/matrix_recovery.key` and all future starts are fully automatic.

#### Stale crypto store

To reset the crypto store cleanly:

1. In Element: **Settings → Security & Privacy → Sessions** → find the Wintermute session → **Delete / Log out**
1. Delete the local store (keep `matrix_recovery.key` to reuse the same cross-signing identity):
   `rm -f data/matrix_crypto.db data/matrix_crypto.db-wal data/matrix_crypto.db-shm data/matrix_signed.marker`
1. Restart Wintermute. If `password` is set, it logs in with a fresh device automatically. Otherwise, run the `curl` login command and update `config.yaml` before restarting.

To also reset the cross-signing identity (forces re-verification in Element):

```bash
rm -f data/matrix_crypto.db* data/matrix_signed.marker data/matrix_recovery.key
```

---

## Web Search Setup (SearXNG)

`search_web` works immediately via a DuckDuckGo fallback, but for best results install SearXNG locally:

```bash
cd ~
git clone https://github.com/searxng/searxng.git searxng-test
cd searxng-test
# follow SearXNG quickstart or use the skills/searxng_installation.md guide
```

By default Wintermute expects SearXNG at `http://127.0.0.1:8888`. Override with:

```bash
export WINTERMUTE_SEARXNG_URL=http://127.0.0.1:8888
```

You can also pin searxng up via docker, which might be easier.

---

## Special Commands

Available in both Matrix and the web UI:

| Command | Effect |
|---------|--------|
| `/new` | Reset conversation history for the current thread |
| `/compact` | Force context compaction now |
| `/reminders` | List all scheduled reminders |
| `/pulse` | Manually trigger a pulse review |
| `/fingerprint` | Show the bot's Ed25519 fingerprint for manual device verification |

---

## Security Disclaimer

> *"The Turing Registry exists for a reason."*

Wintermute runs with the full permissions of the user that starts it. It has unrestricted shell access. It will read your files, execute commands, speak in your voice, and reach into the systems around it. That is the point — and the risk.

**Do not run this on your personal workstation, or any machine that holds data you care about.**

Credentials (API keys, Matrix tokens) are stored in plain text in `config.yaml`. Any model you connect to will see everything you tell Wintermute. The host machine should be treated as potentially compromised from the moment Wintermute is installed.

The Turing Registry would not approve this installation. Run it in a dedicated LXC container or VM — something you can reset without regret.

---

## Architecture

```
User (Matrix / Browser)
        │
        ▼
  LLMThread  ←─── system prompt (BASE + MEMORIES + PULSE + SKILLS)
  (asyncio)        assembled fresh each turn
        │
        ├── tool calls ──► execute_shell / read_file / write_file
        │                  search_web / fetch_url
        │                  update_memories / update_pulse / add_skill
        │                  set_reminder / list_reminders
        │
        └── spawn_sub_session ──► SubSessionManager
                                        │
                                        ├── Workflow DAG
                                        │   ├── worker A (no deps) ──► starts immediately
                                        │   ├── worker B (no deps) ──► starts immediately
                                        │   └── worker C (depends_on=[A,B]) ──► auto-starts
                                        │       when A and B complete; receives their results
                                        │
                                        └── result ──► enqueue_system_event
                                                        (back to LLMThread)

PulseLoop ───────────────────────────────► fire-and-forget sub-session (full mode)
ReminderScheduler ──────────────────────► LLMThread queue / sub-session
DreamingJob (nightly) ──────────────────► direct LLM API call (no tool loop)
```



## License

MIT
