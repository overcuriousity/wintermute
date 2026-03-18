# Signal Setup

Wintermute can connect to Signal messenger via [signal-cli](https://github.com/AsamK/signal-cli), a command-line interface for Signal that uses the Signal protocol directly. Wintermute spawns signal-cli in HTTP daemon mode (`daemon --http`) and communicates via HTTP JSON-RPC for sending and SSE (Server-Sent Events) for receiving messages.

## Prerequisites

- **JRE 25+** (signal-cli requires Java)
  - Fedora: `sudo dnf install java-25-openjdk-headless`
  - Debian/Ubuntu: `sudo apt install openjdk-25-jre-headless`
- **signal-cli** (download from [GitHub releases](https://github.com/AsamK/signal-cli/releases))
- *(Optional)* **qrencode** — for rendering QR codes in the terminal when linking as a secondary device
  - Fedora: `sudo dnf install qrencode`
  - Debian/Ubuntu: `sudo apt install qrencode`
- *(Optional)* **ffmpeg** — required for voice message transcription (same as Matrix)

## Installing signal-cli

Download the latest release and extract it:

```bash
# Check https://github.com/AsamK/signal-cli/releases for the latest version
VERSION=$(curl -Ls -o /dev/null -w %{url_effective} https://github.com/AsamK/signal-cli/releases/latest | sed -e 's/^.*\/v//')
curl -L -O https://github.com/AsamK/signal-cli/releases/download/v"${VERSION}"/signal-cli-"${VERSION}".tar.gz
sudo tar xf signal-cli-"${VERSION}".tar.gz -C /opt
sudo ln -sf /opt/signal-cli-"${VERSION}"/bin/signal-cli /usr/local/bin/
```

Verify installation:

```bash
signal-cli --version
```

## Registering a Phone Number

You need a phone number for the bot. There are two options:

### Option A — Standalone registration (recommended for headless)

Use a dedicated phone number (SIM card, VoIP number, or SIP number):

```bash
# Register via SMS
signal-cli -a +1234567890 register

# Or register via voice call (works with landlines/SIP numbers)
signal-cli -a +1234567890 register --voice

# Verify with the code you received
signal-cli -a +1234567890 verify 123-456
```

This is fully headless — no QR code or smartphone involved. The bot gets its own independent Signal identity.

### Option B — Link as secondary device (no separate number needed)

Link signal-cli to your existing Signal account as a secondary device:

```bash
# Generate a linking URI
signal-cli link -n "Wintermute"
```

This outputs a `sgnl://linkdevice?uuid=...&pub_key=...` URI. On a headless system, render it as a terminal QR code:

```bash
signal-cli link -n "Wintermute" | tee >(xargs -L 1 qrencode -t utf8)
```

The QR code renders as UTF-8 art visible in SSH terminals or `journalctl` output. Scan it with the Signal app on your primary phone to complete linking.

**Note:** When linked as a secondary device, the bot inherits your primary's groups and contacts but cannot register for new groups independently.

## Testing

Send a test message to verify everything works:

```bash
signal-cli -a +1234567890 send -m "Hello from Wintermute" +0987654321
```

## Wintermute Configuration

Add the following to your `config.yaml`:

```yaml
signal:
  enabled: true
  phone_number: "+1234567890"           # Bot's registered Signal number
  signal_cli_path: "signal-cli"         # Path to signal-cli binary
  allowed_users: ["+0987654321"]        # Phone numbers or UUIDs allowed to interact
  allowed_groups: []                    # Group IDs (empty = allow all)
  group_mode: false                     # Only respond when mentioned
  trust_new_keys: true                  # Auto-trust new identity keys
  http_port: 8190                       # Port for signal-cli HTTP daemon
```

### Allowed users

`allowed_users` accepts both phone numbers and Signal UUIDs. This is important because Signal allows users to hide their phone number — in that case, only their UUID is available in the message envelope, and phone-number-based allowlisting will silently reject the message.

If a user has no visible phone number, you **must** use their UUID in `allowed_users`.

```yaml
allowed_users:
  - "+491234567890"                     # Phone number
  - "a1b2c3d4-e5f6-7890-abcd-ef1234567890"  # UUID (required if phone is hidden)
```

If the list is empty, all users are allowed.

**Finding a user's UUID:**

Signal UUIDs aren't visible in the app. There are three ways to find them:

1. **From Wintermute's logs** (easiest) — temporarily remove the user from `allowed_users` or leave the list empty, then have them send a message. The log will show:
   ```
   [signal] User (none) (uuid=a1b2c3d4-e5f6-7890-abcd-ef1234567890) not in allowed_users, ignoring
   ```
   Copy the full UUID into your config.

2. **Via signal-cli** — list all known contacts with their UUIDs:
   ```bash
   signal-cli -a +1234567890 listContacts
   ```
   Or look up a specific number:
   ```bash
   signal-cli -a +1234567890 getUserStatus +0987654321
   ```

3. **signal-cli data directory** — stored in `~/.local/share/signal-cli/data/`, but the methods above are easier.

### Thread ID format

- 1:1 chats: `sig_+491234567890` (phone) or `sig_<uuid>` (UUID)
- Groups: `sig_group_<base64-group-id>`

### Allowed groups

`allowed_groups` uses signal-cli's internal base64-encoded group IDs. To find them:

1. **Via signal-cli** (easiest):
   ```bash
   signal-cli -a +1234567890 listGroups
   ```
   This prints all groups with their IDs, names, and members. Copy the `Id` value into your config.

2. **From Wintermute's logs** — if `allowed_groups` is empty (allow all), send a message in the group and look for the thread_id in the logs. The group ID is the base64 value after `sig_group_`.

```yaml
allowed_groups:
  - "mQ5xR7k3bT..."                    # Base64 group ID from listGroups
```

If the list is empty, all groups are allowed (1:1 mode default).

### Group mode

When `group_mode: true`, the bot only responds to messages that mention its phone number. Each mention is a single-turn conversation (no prior history sent to the LLM). Sender attribution is included as `[+491234567890]: message`.

`allowed_groups` must be set when group mode is enabled to prevent unintended data collection.

## Trust Model

By default, `trust_new_keys: true` tells signal-cli to automatically trust new identity keys (`--trust-new-identities always`). This is required for headless bot operation — without it, signal-cli would block on identity key changes requiring interactive confirmation.

If you want stricter trust (manual key verification), set `trust_new_keys: false` and manage identity trust via signal-cli commands directly.

## Troubleshooting

### Java version errors

signal-cli requires JRE 21+. Check your version:

```bash
java -version
```

If you have multiple Java versions, set `JAVA_HOME` or use the full path in `signal_cli_path`.

### libsignal native library errors

On some systems, signal-cli may fail to load the native libsignal library. Ensure you're using the correct architecture (x86_64 vs aarch64) release of signal-cli.

### Attachment paths

signal-cli stores received attachments in `~/.local/share/signal-cli/attachments/`. Wintermute reads them from the paths provided in the SSE event data. Ensure the Wintermute process has read access to this directory.

### Process management

Wintermute spawns signal-cli as a subprocess in HTTP daemon mode and manages its lifecycle. On startup, it polls `/api/v1/check` until the daemon is ready (up to 60s). If signal-cli crashes, Wintermute automatically restarts it with exponential backoff (1s, 2s, 4s, ... up to 60s). If only the SSE stream disconnects while the process is alive, Wintermute reconnects the stream without restarting signal-cli.

To check if signal-cli is running independently:

```bash
signal-cli -a +1234567890 daemon --http 127.0.0.1:8190 --receive-mode on-start
```

### Whisper transcription

Voice message transcription requires the same Whisper setup as Matrix. See the `whisper:` section in `config.yaml.example`. ffmpeg must be installed for audio format conversion.
