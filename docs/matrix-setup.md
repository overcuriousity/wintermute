# Matrix Setup

## Create a Dedicated Matrix Account

Register a new account for the bot on your homeserver (e.g. via Element or the homeserver's registration page). The bot needs its own account — do not reuse your personal one.

## Configure Credentials

There are two ways to provide Matrix credentials:

### Option A — Password (recommended)

Supply the bot's password and let Wintermute handle login, device creation, and token refresh automatically:

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

### Option B — Manual Token

If you prefer not to store the password, obtain a token via curl and fill it in yourself:

```bash
curl -s -X POST 'https://matrix.org/_matrix/client/v3/login' \
  -H 'Content-Type: application/json' \
  -d '{"type":"m.login.password","identifier":{"type":"m.id.user","user":"@your-bot-name:matrix.org"},"password":"...","initial_device_display_name":"Wintermute"}' \
  | python3 -m json.tool
```

Copy `access_token` and `device_id` from the response into `config.yaml`. You will need to repeat this if the token expires.

## Invite the Bot and Start Chatting

1. Start Wintermute: `uv run wintermute`
2. In Element (or any Matrix client), create a room or open a DM
3. Invite `@your-bot-name:matrix.org`
4. The bot joins and responds to messages from `allowed_users`

## End-to-End Encryption

E2EE is handled automatically. Crypto state is persisted in `data/matrix_crypto.db`; the recovery key that anchors the cross-signing identity is stored in `data/matrix_recovery.key`.

### What happens on a normal restart

Nothing. Wintermute reloads its crypto store, re-signs its device using the stored recovery key, and is ready within seconds. No user action required.

### What happens on a fresh install (first run, or `data/` wiped)

Matrix E2EE requires that every new device be verified by the account owner before other clients will trust it. **This is a fundamental Matrix protocol requirement — it cannot be eliminated.** However, it can be reduced to a single tap in your Matrix client:

1. Wintermute starts, generates a new device identity, and immediately sends a **verification request** to every user listed in `allowed_users`.
2. You will see a notification in Element (or your Matrix client): *"Wintermute is requesting verification"* — tap **Accept**.
3. Wintermute auto-completes the SAS handshake. No emoji comparison is required.
4. The device shows as **Verified** (green shield). E2EE now works for all your devices.

**No SSH needed. No recovery key copy-paste. One tap.**

The recovery key is saved to `data/matrix_recovery.key` — keep this file across reinstalls and all future starts will be fully automatic (no verification needed again).

> **First-run note for matrix.org:** Generating the cross-signing identity requires a one-time approval at `account.matrix.org`. Wintermute logs the exact URL if this is needed. Approve it in a browser, then restart once.

### What to back up

| File | What it is | If lost |
|---|---|---|
| `data/matrix_recovery.key` | Cross-signing anchor | Must re-verify after next install |
| `config.yaml` | Credentials + config | Must reconfigure |

### SAS Verification (manual alternative)

If for any reason the automatic verification request doesn't arrive, you can trigger it manually:

1. In Element go to **Settings → Security → Sessions**, find Wintermute's session, tap **Verify**.
2. Wintermute auto-completes the handshake immediately.

## Group Mode

Group mode lets Wintermute participate in multi-user Matrix rooms while minimizing data exposure to LLM providers.

### Privacy model

When `group_mode: true`, **only messages that @mention the bot (via Matrix pill)** are processed. All other room messages are silently ignored — they are never saved to the database and never sent to the LLM provider.

Each @mention is a **single-turn conversation**: the LLM receives only the current message plus the system prompt (with memories and skills), but no prior conversation history. This means the LLM provider never sees accumulated chat context from the room.

The user's message is saved to the local database before inference; the bot's response is saved after. This allows the memory harvester, dreaming cycle, and reflection systems to extract long-term knowledge from group interactions — but the raw conversation history is never forwarded to the LLM in bulk.

### Configuration

```yaml
matrix:
  group_mode: true
  allowed_rooms:
    - "!room-id:matrix.org"       # REQUIRED — prevents unintended listening
  allowed_users:
    - "@admin:matrix.org"         # Only these users can trigger responses via @mention
```

`allowed_rooms` must be set when `group_mode` is enabled. If omitted, group mode is automatically disabled at startup with a warning.

### How it works

1. Bot joins the configured rooms and listens for events
2. Messages without an @mention (pill or `m.mentions`) are dropped entirely
3. When an allowed user @mentions the bot, the mention text is extracted, the bot name is stripped, and inference runs with **only that message** as context
4. The response is sent back to the room
5. Both the user message and response are persisted locally for memory harvesting

### Debug panel

Group mode sessions show a `[GROUP]` badge in the web debug panel (`/debug`). The context size display reflects the accumulated DB history (used for memory harvesting), not what the LLM actually sees per turn.

## Troubleshooting

### Token Expired (`MUnknownToken`)

If `password` is set in `config.yaml`, Wintermute re-authenticates automatically — no action needed. Otherwise, Wintermute logs the exact `curl` command to obtain a new token. Run it, update `config.yaml`, and restart. Alternatively, add `password` to avoid this in the future.

### Cross-signing Requires Approval (first run on matrix.org only)

On first start, matrix.org requires you to approve the cross-signing key upload. Wintermute logs the exact URL:

```
Cross-signing requires interactive approval from your homeserver.
  1. Open this URL in your browser: https://account.matrix.org/account/?action=org.matrix.cross_signing_reset
  2. Approve the cross-signing reset request.
  3. Restart Wintermute.
```

After approval, restart once. All future starts (including after crypto-store wipes) are automatic, provided `data/matrix_recovery.key` is preserved.

### Messages fail to decrypt after reinstall

If you wiped `data/` including `matrix_recovery.key`, Wintermute has a new device identity. Accept the verification request that Wintermute sends to your Matrix client. Once verified, send a new message — it will decrypt correctly. Old messages sent before verification are unrecoverable (Matrix E2EE design).

### Stale Crypto Store (keep identity)

To reset only the crypto store while preserving the cross-signing identity (no re-verification needed):

1. In Element: **Settings → Security → Sessions** → find the Wintermute session → **Delete / Sign out**.
2. ```bash
   rm -f data/matrix_crypto.db data/matrix_crypto.db-wal data/matrix_crypto.db-shm data/matrix_signed.marker
   ```
3. Restart Wintermute.

### Full Reset (new identity, re-verification required)

```bash
rm -f data/matrix_crypto.db* data/matrix_signed.marker data/matrix_recovery.key
```

Restart Wintermute, then accept the verification request that appears in your Matrix client.
