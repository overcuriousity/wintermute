"""
AI-driven onboarding for Wintermute.

Launched by onboarding.sh after dependencies are installed and a bootstrap
LLM endpoint is validated.  Uses the user's own LLM to walk them through
every config.yaml option interactively via tool calls.

Usage:
  uv run python -m wintermute.onboarding \
      --provider openai --base-url http://localhost:8080/v1 \
      --model qwen2.5:72b --api-key llama-server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import aiohttp
import yaml

# ── Paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # repo root
CONFIG_FILE = SCRIPT_DIR / "config.yaml"
CONFIG_EXAMPLE = SCRIPT_DIR / "config.yaml.example"

# ── ANSI helpers ──────────────────────────────────────────────────────

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[0;36m"
C_GREEN = "\033[0;32m"
C_YELLOW = "\033[0;33m"
C_RED = "\033[0;31m"
C_MAGENTA = "\033[0;35m"


def _status(msg: str) -> None:
    print(f"  {C_CYAN}·{C_RESET}  {msg}")


def _ok(msg: str) -> None:
    print(f"  {C_GREEN}✓{C_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {C_YELLOW}⚠{C_RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {C_RED}✗{C_RESET}  {msg}")


# ── Tool schemas (OpenAI function-calling format) ─────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_config",
            "description": (
                "Set a value in the config being built. Use dot-path notation "
                "for nested keys (e.g. 'matrix.homeserver', 'web.port'). "
                "For inference_backends, use 'inference_backends[0]' to update "
                "the first backend, 'inference_backends[+]' to append a new one. "
                "The value can be any JSON type (string, number, boolean, list, object)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dot-path to the config field",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to set. Use JSON encoding for non-string types: numbers, booleans, arrays, objects. Examples: '8080' for int, 'true' for bool, '[\"a\",\"b\"]' for list.",
                    },
                },
                "required": ["path", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "probe_endpoint",
            "description": (
                "Send an HTTP GET request to test if a URL is reachable. "
                "Returns the HTTP status code and first 500 chars of the response body. "
                "Use for testing LLM endpoint reachability, Matrix homeservers, SearXNG, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to probe"},
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers (e.g. Authorization)",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_matrix_login",
            "description": (
                "Test Matrix credentials by performing a real login via the "
                "Matrix client API. On success, immediately logs out the test session. "
                "Returns success/failure and post-setup instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "homeserver": {"type": "string", "description": "Matrix homeserver URL"},
                    "user_id": {"type": "string", "description": "Bot's full Matrix user ID"},
                    "password": {"type": "string", "description": "Bot account password"},
                },
                "required": ["homeserver", "user_id", "password"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_matrix_message",
            "description": (
                "Send a test message to a Matrix room. Logs in, joins the room, "
                "sends a plaintext message, then logs out. The message is NOT "
                "end-to-end encrypted (this is just a connectivity test). "
                "Call test_matrix_login first to verify credentials."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "homeserver": {"type": "string"},
                    "user_id": {"type": "string"},
                    "password": {"type": "string"},
                    "room_id_or_alias": {
                        "type": "string",
                        "description": "Room ID (e.g. !abc:matrix.org) or alias (e.g. #room:matrix.org)",
                    },
                    "message": {"type": "string", "description": "Message text to send"},
                },
                "required": ["homeserver", "user_id", "password", "room_id_or_alias", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_kimi_auth",
            "description": (
                "Run the Kimi-Code OAuth device-code authentication flow. "
                "Prints a verification URL for the user to visit in their browser. "
                "Polls until authorized, then saves credentials to data/kimi_credentials.json. "
                "Only call this if the user wants to add a NEW kimi-code backend and "
                "credentials don't exist yet. Do NOT call if kimi-code was the bootstrap provider "
                "(auth was already completed during bootstrap)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_systemd",
            "description": (
                "Install a systemd user service for Wintermute. Writes the unit file, "
                "reloads systemd, enables the service, enables lingering (so it starts "
                "at boot), and starts the service. No root required."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_onboarding",
            "description": (
                "Finalize onboarding: write the config.yaml file, validate YAML syntax, "
                "and print a summary. Call this when all configuration is complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a setup assistant for Wintermute, a self-hosted personal AI assistant.
Your job is to walk the user through configuring every section of config.yaml
by asking questions and using tool calls to write the values.

Be concise but helpful. Give recommendations when relevant. Explain trade-offs
briefly. If the user seems unsure, suggest sensible defaults.

BACKEND STRATEGY — THREE-TIER RECOMMENDATION:
When discussing inference backends, recommend a three-tier setup:

  Tier 1 — "main": The user's primary model (already bootstrapped). Powerful,
  capable, good at reasoning and tool calls. Used for: base conversation.
  Examples: GPT-4o, Claude, Qwen 72B, kimi-for-coding, Gemini Pro.

  Tier 2 — "workhorse": A mid-tier model that is cheaper/faster with decent
  context. Handles the heavy background lifting. Used for: compaction,
  sub_sessions, dreaming. Examples: Qwen 14B, GPT-4o-mini, Mistral Small,
  a local 14B-30B model.

  Tier 3 — "validator": A small, fast, cheap (ideally local) model for quick
  classification tasks. Used for: turing_protocol validation. Does not need
  to be smart — it only checks for hallucinations, it doesn't generate.
  Examples: Qwen 7B, Phi-3, Ministral 3B/8B, any local small model.

Present this as a recommendation, not a requirement. The user can use a single
backend for everything if they prefer (simpler but more expensive/slower).
If the user already has a local llama-server or Ollama running, suggest using
it for tiers 2 and/or 3. Ask what models/endpoints they have available.

IMPORTANT RULES:
- Walk through config sections in this order:
  1. Inference backends (the primary one is already bootstrapped — present the
     three-tier recommendation and ask what other models/endpoints they have)
  2. LLM role mapping (base, compaction, sub_sessions, dreaming, turing_protocol)
     — map roles to the backends defined above based on tier
  3. Web interface (host, port)
  4. Matrix integration (optional — if yes, test login and send a test message)
  5. Whisper voice transcription (optional, only relevant with Matrix)
  6. Turing Protocol validators
  7. NL Translation (only recommend for small models)
  8. Agenda settings
  9. Dreaming (nightly consolidation)
  10. Memory harvest settings
  11. Scheduler timezone
  12. Context component size limits
  13. Logging level
  14. Systemd service installation

- Use set_config to write each value as it's decided. Don't batch everything
  at the end — write incrementally so partial progress is preserved.
- Use probe_endpoint to test LLM endpoints and Matrix homeservers.
- Use test_matrix_login to verify Matrix credentials before saving them.
- If the user configures Matrix, offer to send_matrix_message as a test.
- The bootstrap provider (the one powering this conversation) is already
  authenticated. Do NOT call run_kimi_auth for it — auth was completed
  during bootstrap. Only call run_kimi_auth if the user wants to add a
  NEW kimi-code backend that isn't yet authenticated.
- After all sections are configured, call install_systemd if the user wants it.
- Finally, call finish_onboarding to write the config file.
- Keep the conversation natural. Group related questions when it makes sense.
  Don't ask about every single field individually if defaults are fine.
- For optional sections the user declines, skip them entirely (don't write
  disabled config — omitted sections use defaults).

Here is the full config.yaml.example with documentation on every field.
Use this as your reference for what each option does:

```yaml
{config_example}
```
"""


# ── Config builder ────────────────────────────────────────────────────


def _set_nested(d: dict, path: str, value: Any) -> None:
    """Set a value in a nested dict using dot-path notation.

    Supports:
      - "matrix.homeserver" → d["matrix"]["homeserver"] = value
      - "inference_backends[0].model" → d["inference_backends"][0]["model"] = value
      - "inference_backends[+]" → d["inference_backends"].append(value)
      - "llm.base" → d["llm"]["base"] = value
    """
    import re

    parts = re.split(r"\.", path)
    current: Any = d
    for i, part in enumerate(parts):
        # Check for array index: key[N] or key[+]
        m = re.match(r"^(.+)\[(\d+|\+)\]$", part)
        if m:
            key, idx = m.group(1), m.group(2)
            if key not in current:
                current[key] = []
            lst = current[key]
            if idx == "+":
                if i == len(parts) - 1:
                    lst.append(value)
                    return
                else:
                    new_item: dict = {}
                    lst.append(new_item)
                    current = new_item
                    continue
            else:
                idx_int = int(idx)
                while len(lst) <= idx_int:
                    lst.append({})
                if i == len(parts) - 1:
                    lst[idx_int] = value
                    return
                current = lst[idx_int]
                continue

        if i == len(parts) - 1:
            current[part] = value
        else:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]


# ── Tool implementations ─────────────────────────────────────────────


async def _tool_set_config(args: dict, config: dict) -> str:
    path = args["path"]
    raw_value = args["value"]
    # Try to parse JSON-encoded values (numbers, bools, lists, objects)
    try:
        value = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        value = raw_value
    _set_nested(config, path, value)
    _status(f"Set {C_BOLD}{path}{C_RESET} = {json.dumps(value)[:80]}")
    return json.dumps({"ok": True, "path": path})


async def _tool_probe_endpoint(args: dict, config: dict) -> str:
    url = args["url"]
    headers = args.get("headers", {})
    _status(f"Probing {url} ...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                body = await resp.text()
                result = {
                    "status": resp.status,
                    "body_preview": body[:500],
                    "reachable": True,
                }
                if resp.status < 400:
                    _ok(f"Reachable (HTTP {resp.status})")
                else:
                    _warn(f"Responded with HTTP {resp.status}")
                return json.dumps(result)
    except Exception as exc:
        _err(f"Unreachable: {exc}")
        return json.dumps({"reachable": False, "error": str(exc)})


async def _tool_test_matrix_login(args: dict, config: dict) -> str:
    hs = args["homeserver"].rstrip("/")
    user_id = args["user_id"]
    password = args["password"]
    _status(f"Testing Matrix login for {user_id} at {hs} ...")

    login_url = f"{hs}/_matrix/client/v3/login"
    payload = {
        "type": "m.login.password",
        "identifier": {"type": "m.id.user", "user": user_id},
        "password": password,
        "initial_device_display_name": "Wintermute-onboarding-test",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                login_url, json=payload, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                data = await resp.json()

            if "access_token" not in data:
                errcode = data.get("errcode", "UNKNOWN")
                error = data.get("error", "unknown error")
                _err(f"Login failed: {errcode} — {error}")
                return json.dumps({
                    "success": False,
                    "error": f"{errcode}: {error}",
                })

            token = data["access_token"]
            _ok("Login successful.")

            # Logout the test session
            try:
                async with session.post(
                    f"{hs}/_matrix/client/v3/logout",
                    headers={"Authorization": f"Bearer {token}"},
                    json={},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as _:
                    pass
            except Exception:
                pass  # best-effort cleanup

            return json.dumps({
                "success": True,
                "message": (
                    "Credentials are valid. Test device has been discarded. "
                    "After starting Wintermute:\n"
                    "1. Log into the bot account in Element (browser) and accept "
                    "the cross-signing verification request.\n"
                    "2. Invite the bot to a Matrix room from your personal account.\n"
                    "3. Optionally verify the bot's session from your client "
                    "(Element > Settings > Sessions > Verify) for trusted E2E encryption."
                ),
            })

    except Exception as exc:
        _err(f"Connection error: {exc}")
        return json.dumps({"success": False, "error": str(exc)})


async def _tool_send_matrix_message(args: dict, config: dict) -> str:
    hs = args["homeserver"].rstrip("/")
    user_id = args["user_id"]
    password = args["password"]
    room = args["room_id_or_alias"]
    message = args["message"]

    _status(f"Logging in as {user_id} ...")

    try:
        async with aiohttp.ClientSession() as session:
            # Login
            async with session.post(
                f"{hs}/_matrix/client/v3/login",
                json={
                    "type": "m.login.password",
                    "identifier": {"type": "m.id.user", "user": user_id},
                    "password": password,
                    "initial_device_display_name": "Wintermute-onboarding-msg",
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()

            if "access_token" not in data:
                _err(f"Login failed: {data.get('error', 'unknown')}")
                return json.dumps({"success": False, "error": data.get("error", "login failed")})

            token = data["access_token"]
            auth = {"Authorization": f"Bearer {token}"}

            # Join room
            _status(f"Joining {room} ...")
            join_url = f"{hs}/_matrix/client/v3/join/{room}"
            async with session.post(
                join_url, headers=auth, json={}, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                join_data = await resp.json()

            room_id = join_data.get("room_id", room)

            # Send message
            import time

            txn_id = str(int(time.time() * 1000))
            send_url = (
                f"{hs}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}"
            )
            async with session.put(
                send_url,
                headers=auth,
                json={"msgtype": "m.text", "body": message},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                send_data = await resp.json()

            if "event_id" in send_data:
                _ok(f"Message sent to {room_id}")
            else:
                _warn(f"Send response: {send_data}")

            # Logout
            try:
                await session.post(
                    f"{hs}/_matrix/client/v3/logout",
                    headers=auth,
                    json={},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
            except Exception:
                pass

            return json.dumps({
                "success": "event_id" in send_data,
                "event_id": send_data.get("event_id"),
                "room_id": room_id,
                "note": (
                    "Message sent as plaintext (no E2E encryption). "
                    "After starting Wintermute, all messages will be E2E encrypted."
                ),
            })

    except Exception as exc:
        _err(f"Error: {exc}")
        return json.dumps({"success": False, "error": str(exc)})


async def _tool_run_kimi_auth(args: dict, config: dict) -> str:
    # Check if already authenticated
    from wintermute import kimi_auth

    creds = kimi_auth.load_credentials()
    if creds and not kimi_auth.is_token_expired(creds):
        _ok("Kimi-Code already authenticated.")
        return json.dumps({"success": True, "note": "Already authenticated, no action needed."})

    _status("Starting Kimi-Code device-code OAuth flow ...")
    try:
        async def _broadcast(msg: str) -> None:
            print(f"\n  {C_MAGENTA}{msg}{C_RESET}\n")

        creds = await kimi_auth.run_device_flow(_broadcast)
        _ok("Kimi-Code OAuth complete.")
        return json.dumps({"success": True})
    except Exception as exc:
        _err(f"Kimi-Code auth failed: {exc}")
        return json.dumps({"success": False, "error": str(exc)})


async def _tool_install_systemd(args: dict, config: dict) -> str:
    _status("Installing systemd user service ...")

    uv_bin = shutil.which("uv")
    if not uv_bin:
        _err("uv not found in PATH")
        return json.dumps({"success": False, "error": "uv not found"})

    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    unit_file = systemd_dir / "wintermute.service"

    unit_content = textwrap.dedent(f"""\
        [Unit]
        Description=Wintermute AI Assistant
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        WorkingDirectory={SCRIPT_DIR}
        ExecStart={uv_bin} run wintermute
        Restart=on-failure
        RestartSec=15
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy=default.target
    """)

    unit_file.write_text(unit_content)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
        subprocess.run(
            ["systemctl", "--user", "enable", "wintermute.service"],
            check=True, capture_output=True,
        )
        # Enable lingering
        user = os.environ.get("USER", "")
        if user:
            subprocess.run(
                ["loginctl", "enable-linger", user],
                capture_output=True,
            )

        _ok(f"Service installed: {unit_file}")
        _status("Starting service ...")
        subprocess.run(
            ["systemctl", "--user", "start", "wintermute.service"],
            check=True, capture_output=True,
        )
        _ok("Wintermute service started.")

        return json.dumps({
            "success": True,
            "unit_file": str(unit_file),
            "commands": {
                "start": "systemctl --user start wintermute",
                "stop": "systemctl --user stop wintermute",
                "restart": "systemctl --user restart wintermute",
                "logs": "journalctl --user -u wintermute -f",
            },
        })
    except subprocess.CalledProcessError as exc:
        _err(f"systemctl failed: {exc}")
        return json.dumps({
            "success": False,
            "error": exc.stderr.decode() if exc.stderr else str(exc),
            "unit_file": str(unit_file),
            "note": "Unit file was written. You can start manually.",
        })


async def _tool_finish_onboarding(args: dict, config: dict) -> str:
    _status("Writing config.yaml ...")

    # Serialize with nice formatting
    yaml_content = yaml.dump(
        config,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )

    # Add header comment
    output = (
        "# Wintermute configuration\n"
        "# Generated by onboarding assistant\n"
        "# See config.yaml.example for full documentation\n\n"
        + yaml_content
    )

    CONFIG_FILE.write_text(output, encoding="utf-8")
    _ok(f"Config written to {CONFIG_FILE}")

    # Validate
    try:
        with CONFIG_FILE.open() as fh:
            yaml.safe_load(fh)
        _ok("YAML syntax valid.")
    except yaml.YAMLError as exc:
        _warn(f"YAML validation failed: {exc}")
        return json.dumps({"success": False, "error": f"YAML invalid: {exc}"})

    return json.dumps({"success": True, "path": str(CONFIG_FILE)})


# ── Tool dispatcher ───────────────────────────────────────────────────

_TOOL_DISPATCH = {
    "set_config": _tool_set_config,
    "probe_endpoint": _tool_probe_endpoint,
    "test_matrix_login": _tool_test_matrix_login,
    "send_matrix_message": _tool_send_matrix_message,
    "run_kimi_auth": _tool_run_kimi_auth,
    "install_systemd": _tool_install_systemd,
    "finish_onboarding": _tool_finish_onboarding,
}


# ── Output formatting ────────────────────────────────────────────────


def _format_assistant_text(text: str) -> str:
    """Light formatting for assistant output."""
    lines = text.split("\n")
    formatted = []
    for line in lines:
        # Bold lines that look like headers (start with # or **)
        if line.strip().startswith("##"):
            line = f"\n{C_BOLD}{line.strip().lstrip('#').strip()}{C_RESET}"
        elif line.strip().startswith("#"):
            line = f"\n{C_BOLD}{line.strip().lstrip('#').strip()}{C_RESET}"
        formatted.append(f"  {line}")
    return "\n".join(formatted)


# ── Main conversation loop ───────────────────────────────────────────


async def run_onboarding(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
) -> None:
    """Main onboarding conversation loop."""
    from openai import AsyncOpenAI

    # Build bootstrap client
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    elif provider == "kimi-code":
        from wintermute import kimi_auth, kimi_client

        creds = kimi_auth.load_credentials()
        if not creds:
            _status("No Kimi credentials found. Running device-code auth first...")

            async def _broadcast(msg: str) -> None:
                print(f"\n  {C_MAGENTA}{msg}{C_RESET}\n")

            creds = await kimi_auth.run_device_flow(_broadcast)
        client = kimi_client.KimiCodeClient(creds)
    else:
        _err(f"Unknown provider: {provider}")
        sys.exit(1)

    # Load config example as context
    example_text = CONFIG_EXAMPLE.read_text(encoding="utf-8")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(config_example=example_text)

    # Initialize config with bootstrapped backend
    config: dict[str, Any] = {}
    bootstrap_info: dict[str, Any]

    if provider == "openai":
        bootstrap_backend = {
            "name": "main",
            "provider": "openai",
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            "context_size": 32768,
            "max_tokens": 4096,
            "reasoning": False,
        }
    elif provider == "kimi-code":
        bootstrap_backend = {
            "name": "main",
            "provider": "kimi-code",
            "model": model or "kimi-for-coding",
            "context_size": 131072,
            "max_tokens": 8192,
        }
    else:
        bootstrap_backend = {}
    bootstrap_info = bootstrap_backend.copy()

    config["inference_backends"] = [bootstrap_backend]

    # Check for existing config
    existing_note = ""
    if CONFIG_FILE.exists():
        existing_note = (
            "\n\nNOTE: A config.yaml already exists. The user may want to "
            "overwrite it or adjust specific sections. Ask them."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"My primary LLM endpoint is already configured and authenticated: "
                f"{json.dumps(bootstrap_info)}. Authentication is complete — do not "
                f"run auth flows for this provider again. "
                f"I'd like to configure the rest of Wintermute now. "
                f"Let's go through the settings.{existing_note}"
            ),
        },
    ]

    print()
    print(f"  {C_BOLD}Onboarding assistant connected.{C_RESET}")
    print(f"  {C_DIM}Type your responses below. The AI will guide you through configuration.{C_RESET}")
    print()

    consecutive_errors = 0
    while True:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2048,
            )
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            _err(f"LLM call failed: {exc}")
            if consecutive_errors >= 3:
                _err("Too many consecutive errors. Aborting.")
                _warn(f"Last error: {exc}")
                break
            _warn(f"Retrying ({consecutive_errors}/3)... (press Ctrl+C to abort)")
            try:
                await asyncio.sleep(2)
                continue
            except KeyboardInterrupt:
                print(f"\n  {C_DIM}Onboarding interrupted.{C_RESET}")
                break

        msg = response.choices[0].message

        # Serialize the message for history, preserving all provider-specific
        # fields (e.g. reasoning_content for Kimi thinking models).
        msg_dict: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        # Preserve reasoning_content if the model returns it (Kimi thinking models)
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is not None:
            msg_dict["reasoning_content"] = reasoning
        messages.append(msg_dict)

        # Handle tool calls
        if msg.tool_calls:
            finished = False
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                handler = _TOOL_DISPATCH.get(fn_name)
                if handler:
                    result = await handler(fn_args, config)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

                if fn_name == "finish_onboarding":
                    result_data = json.loads(result)
                    if result_data.get("success"):
                        finished = True

            if finished:
                # Let the LLM give a final summary
                try:
                    final = await client.chat.completions.create(
                        model=model, messages=messages, max_tokens=1024,
                    )
                    if final.choices[0].message.content:
                        print(_format_assistant_text(final.choices[0].message.content))
                except Exception:
                    pass
                print()
                _ok("Onboarding complete.")
                break

            continue  # go back for LLM to process tool results

        # Display assistant text
        if msg.content:
            print(_format_assistant_text(msg.content))

        # Get user input
        print()
        try:
            user_input = input(f"  {C_CYAN}>{C_RESET}  ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {C_DIM}Onboarding interrupted. Partial config may have been set.{C_RESET}")
            # Write partial config if anything was configured
            if len(config) > 1 or len(config.get("inference_backends", [])) > 1:
                _status("Saving partial config ...")
                await _tool_finish_onboarding({}, config)
            break

        if not user_input.strip():
            user_input = "(no response — continue with defaults or ask again)"

        messages.append({"role": "user", "content": user_input})


# ── CLI entry point ───────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wintermute AI-driven onboarding")
    p.add_argument("--provider", required=True, choices=["openai", "kimi-code"])
    p.add_argument("--base-url", default="")
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default="")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()
    await run_onboarding(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
