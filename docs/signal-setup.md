# Signal Setup

Wintermute can connect to Signal messenger via [signal-cli](https://github.com/AsamK/signal-cli), a command-line interface for Signal that uses the Signal protocol directly. Wintermute spawns signal-cli as a subprocess and communicates via JSON-RPC.

## Prerequisites

- **JRE 21+** (signal-cli requires Java)
  - Fedora: `sudo dnf install java-21-openjdk-headless`
  - Debian/Ubuntu: `sudo apt install openjdk-21-jre-headless`
- **signal-cli** (download from [GitHub releases](https://github.com/AsamK/signal-cli/releases))
- *(Optional)* **qrencode** — for rendering QR codes in the terminal when linking as a secondary device
  - Fedora: `sudo dnf install qrencode`
  - Debian/Ubuntu: `sudo apt install qrencode`
- *(Optional)* **ffmpeg** — required for voice message transcription (same as Matrix)

## Installing signal-cli

Download the latest release and extract it:

```bash
# Check https://github.com/AsamK/signal-cli/releases for the latest version
SIGNAL_CLI_VERSION=0.13.12
wget https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}-Linux.tar.gz
sudo tar xf signal-cli-${SIGNAL_CLI_VERSION}-Linux.tar.gz -C /opt
sudo ln -sf /opt/signal-cli-${SIGNAL_CLI_VERSION}/bin/signal-cli /usr/local/bin/signal-cli
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
  allowed_users: ["+0987654321"]        # Phone numbers allowed to interact
  allowed_groups: []                    # Group IDs (empty = allow all)
  group_mode: false                     # Only respond when mentioned
  trust_new_keys: true                  # Auto-trust new identity keys
```

### Thread ID format

- 1:1 chats: `sig_+491234567890`
- Groups: `sig_group_<base64-group-id>`

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

signal-cli stores received attachments in `~/.local/share/signal-cli/attachments/`. Wintermute reads them from the paths provided in the JSON-RPC response. Ensure the Wintermute process has read access to this directory.

### Process management

Wintermute spawns signal-cli as a subprocess and manages its lifecycle. If signal-cli crashes, Wintermute automatically restarts it with exponential backoff (1s, 2s, 4s, ... up to 60s).

To check if signal-cli is running independently:

```bash
signal-cli -a +1234567890 daemon --json
```

### Whisper transcription

Voice message transcription requires the same Whisper setup as Matrix. See the `whisper:` section in `config.yaml.example`. ffmpeg must be installed for audio format conversion.
