"""
Kimi-Code OAuth Device Flow

Implements the OAuth device-code flow for Kimi-Code authentication.
Credentials are persisted to data/kimi_credentials.json for reuse across
restarts.  The device-code flow is designed to work via chat interfaces
(Matrix / web) — the user copy-pastes a URL into their browser.

Protocol details are based on Kimi's public OAuth endpoints.
"""

import asyncio
import json
import logging
import os
import platform
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Awaitable

import aiohttp

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path("data") / "kimi_credentials.json"

# OAuth constants
KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_OAUTH_HOST = "https://auth.kimi.com"

# Buffer before expiry to trigger proactive refresh.
_EXPIRY_BUFFER_SECONDS = 300  # 5 minutes

# Device ID persistence (one per installation)
_DEVICE_ID_FILE = Path("data") / ".kimi_device_id"


# ---------------------------------------------------------------------------
# Device identification headers (required by Kimi's OAuth endpoints)
# ---------------------------------------------------------------------------

def _get_device_id() -> str:
    """Get or create a persistent device ID."""
    _DEVICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _DEVICE_ID_FILE.exists():
        return _DEVICE_ID_FILE.read_text(encoding="utf-8").strip()
    device_id = uuid.uuid4().hex
    _DEVICE_ID_FILE.write_text(device_id, encoding="utf-8")
    try:
        os.chmod(_DEVICE_ID_FILE, 0o600)
    except OSError:
        pass
    return device_id


def _common_headers() -> dict[str, str]:
    """Build X-Msh-* headers expected by Kimi's OAuth endpoints."""
    return {
        "X-Msh-Platform": "wintermute",
        "X-Msh-Version": "1.0.0",
        "X-Msh-Device-Name": platform.node() or socket.gethostname(),
        "X-Msh-Device-Model": f"{platform.system()} {platform.release()} {platform.machine()}".strip(),
        "X-Msh-Os-Version": platform.version(),
        "X-Msh-Device-Id": _get_device_id(),
    }


# ---------------------------------------------------------------------------
# Credential persistence
# ---------------------------------------------------------------------------

def load_credentials() -> dict | None:
    """Load credentials from data/kimi_credentials.json."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        required = ["access_token", "refresh_token", "expires_at"]
        if all(k in data for k in required):
            return data
        logger.warning("Kimi credentials file is missing required fields")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load kimi credentials: %s", exc)
        return None


def save_credentials(creds: dict) -> None:
    """Persist credentials to data/kimi_credentials.json."""
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except OSError:
        pass
    logger.info("Kimi credentials saved to %s", CREDENTIALS_FILE)


def is_token_expired(creds: dict) -> bool:
    """Check whether the access token is expired (with 5-min buffer)."""
    expires_at = creds.get("expires_at", 0)
    return time.time() >= (expires_at - _EXPIRY_BUFFER_SECONDS)


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------

async def refresh_access_token(creds: dict) -> dict:
    """Refresh the access token using the stored refresh token.

    Returns updated credentials dict (and saves to disk).
    """
    oauth_host = os.getenv("KIMI_CODE_OAUTH_HOST") or KIMI_OAUTH_HOST
    url = f"{oauth_host.rstrip('/')}/api/oauth/token"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={
            "client_id": KIMI_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": creds["refresh_token"],
        }, headers=_common_headers()) as resp:
            data = await resp.json(content_type=None)
            status = resp.status

    if status in (401, 403):
        raise RuntimeError(
            f"Kimi token refresh unauthorized: {data.get('error_description', 'credentials rejected')}"
        )
    if status != 200:
        raise RuntimeError(
            f"Kimi token refresh failed ({status}): {data.get('error_description', data)}"
        )

    updated = dict(creds)
    updated["access_token"] = data["access_token"]
    updated["refresh_token"] = data["refresh_token"]
    updated["expires_at"] = time.time() + float(data["expires_in"])
    save_credentials(updated)
    logger.info("Kimi access token refreshed (expires_at=%.0f)", updated["expires_at"])
    return updated


# ---------------------------------------------------------------------------
# Device-code authorization flow
# ---------------------------------------------------------------------------

async def _request_device_authorization() -> dict[str, Any]:
    """POST to Kimi's device_authorization endpoint."""
    oauth_host = os.getenv("KIMI_CODE_OAUTH_HOST") or KIMI_OAUTH_HOST
    url = f"{oauth_host.rstrip('/')}/api/oauth/device_authorization"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={
            "client_id": KIMI_CLIENT_ID,
        }, headers=_common_headers()) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"Device authorization failed: {data}")
    return data


async def _poll_device_token(device_code: str) -> tuple[int, dict[str, Any]]:
    """Poll the token endpoint once for a device code grant."""
    oauth_host = os.getenv("KIMI_CODE_OAUTH_HOST") or KIMI_OAUTH_HOST
    url = f"{oauth_host.rstrip('/')}/api/oauth/token"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data={
            "client_id": KIMI_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }, headers=_common_headers()) as resp:
            data = await resp.json(content_type=None)
            return resp.status, data


async def run_device_flow(
    broadcast_fn: Callable[[str], Awaitable[None]],
) -> dict:
    """Run the OAuth device-code flow.

    *broadcast_fn* is called with user-facing status messages (the
    verification URL, success/failure notices).  It should accept a single
    string argument and be awaitable.

    Returns the credentials dict on success.
    """
    auth_data = await _request_device_authorization()

    verification_url = auth_data.get("verification_uri_complete", "")
    user_code = auth_data.get("user_code", "")
    device_code = auth_data["device_code"]
    poll_interval = int(auth_data.get("interval", 5))
    expires_in = int(auth_data.get("expires_in", 0)) or 600

    msg = (
        f"**Kimi-Code authentication required.**\n\n"
        f"Visit: {verification_url}\n\n"
        f"User code: `{user_code}`"
    )
    await broadcast_fn(msg)
    logger.info("Kimi device flow started — user code: %s", user_code)

    deadline = time.time() + expires_in

    while time.time() < deadline:
        await asyncio.sleep(poll_interval)
        try:
            status_code, payload = await _poll_device_token(device_code)
        except Exception as exc:
            logger.debug("Kimi device token poll error: %s", exc)
            continue

        if status_code == 200 and "access_token" in payload:
            creds = {
                "access_token": payload["access_token"],
                "refresh_token": payload["refresh_token"],
                "expires_at": time.time() + float(payload["expires_in"]),
            }
            save_credentials(creds)
            await broadcast_fn("Kimi-Code authentication successful!")
            logger.info("Kimi device flow completed successfully")
            return creds

        error = payload.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            poll_interval = min(poll_interval + 2, 15)
            continue
        if error == "expired_token":
            raise TimeoutError("Device code expired before user completed authorization")

        logger.warning("Kimi device token poll: %s", payload)

    raise TimeoutError("Kimi device-code flow timed out waiting for user authorization")


# ---------------------------------------------------------------------------
# CLI entry point (for setup.sh)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    async def _cli_broadcast(msg: str) -> None:
        print(msg)

    async def _main() -> None:
        creds = load_credentials()
        if creds and not is_token_expired(creds):
            print("Kimi-Code credentials already present and valid.")
            return
        if creds:
            try:
                creds = await refresh_access_token(creds)
                print("Kimi-Code token refreshed successfully.")
                return
            except Exception:
                print("Token refresh failed — starting device flow.")

        await run_device_flow(_cli_broadcast)

    asyncio.run(_main())
