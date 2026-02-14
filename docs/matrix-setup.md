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

E2E encryption is handled automatically — the bot's crypto keys are persisted to `data/matrix_crypto.db` and the device is cross-signed at startup. The device fingerprint is logged on every start.

### Cross-signing and Device Verification

On first start, Wintermute calls `generate_recovery_key()` to establish its cross-signing identity and saves the recovery key to `data/matrix_recovery.key`. On every subsequent start — including after the crypto store is wiped — it calls `verify_with_recovery_key()` to re-sign the current device using the stored key, with no browser interaction and no UIA approval required.

### SAS Verification (Emoji Handshake)

Wintermute implements the **m.sas.v1** (emoji) interactive verification protocol. To verify the device:

1. In Element go to **Settings > Security > Sessions**, select Wintermute's session, and tap **Verify Session**.
2. Element will start an emoji handshake. Wintermute auto-accepts from allowed users, skipping the emoji-comparison step.
3. After a moment the device shows a green shield (**Verified**) in Element.

## Troubleshooting

### Token Expired (`MUnknownToken`)

If `password` is set in `config.yaml`, Wintermute re-authenticates automatically — no action needed. Otherwise, Wintermute logs the exact `curl` command to obtain a new token. Run it, update `config.yaml`, and restart. Alternatively, add `password` to avoid this in the future.

### Cross-signing Requires Approval (first run only)

On first start, some homeservers (including matrix.org) require you to approve the cross-signing key upload via your account page. Wintermute logs the exact URL:

```text
Cross-signing requires interactive approval from your homeserver.
  1. Open this URL in your browser: https://account.matrix.org/account/?action=org.matrix.cross_signing_reset
  2. Approve the cross-signing reset request.
  3. Restart Wintermute.
```

After approval, restart once. The recovery key is saved to `data/matrix_recovery.key` and all future starts are fully automatic.

### Stale Crypto Store

To reset the crypto store cleanly:

1. In Element: **Settings > Security & Privacy > Sessions** > find the Wintermute session > **Delete / Log out**
2. Delete the local store (keep `matrix_recovery.key` to reuse the same cross-signing identity):
   ```bash
   rm -f data/matrix_crypto.db data/matrix_crypto.db-wal data/matrix_crypto.db-shm data/matrix_signed.marker
   ```
3. Restart Wintermute. If `password` is set, it logs in with a fresh device automatically. Otherwise, run the `curl` login command and update `config.yaml` before restarting.

To also reset the cross-signing identity (forces re-verification in Element):

```bash
rm -f data/matrix_crypto.db* data/matrix_signed.marker data/matrix_recovery.key
```
