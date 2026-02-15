"""
Gemini CLI Credential Extraction & OAuth Flow

Extracts OAuth client credentials from a locally-installed gemini-cli npm
package, runs a PKCE authorization code flow against Google's OAuth servers,
and discovers/provisions a Cloud Code Assist project.

Credentials are persisted to data/gemini_credentials.json for reuse.
"""

import base64
import hashlib
import http.server
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path("data") / "gemini_credentials.json"

OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

CLOUDCODE_ENDPOINT = "https://cloudcode-pa.googleapis.com"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
REDIRECT_PORT = 8085
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"


# ---------------------------------------------------------------------------
# 1. Find gemini-cli binary
# ---------------------------------------------------------------------------

def find_gemini_cli() -> Path | None:
    """Locate the `gemini` binary in PATH and resolve to its npm install root."""
    binary = shutil.which("gemini")
    if binary is None:
        return None
    resolved = Path(binary).resolve()
    # Walk up to find the npm package root (contains node_modules or package.json)
    # Typical layout: .../node_modules/.bin/gemini -> .../node_modules/@google/gemini-cli/...
    current = resolved
    for _ in range(10):
        current = current.parent
        if (current / "package.json").exists():
            return current
        # Check if we're inside node_modules/@google/gemini-cli
        if current.name == "gemini-cli" and current.parent.name == "@google":
            return current
    # Fallback: try to find the package via npm root
    try:
        result = subprocess.run(
            ["npm", "root", "-g"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            npm_root = Path(result.stdout.strip())
            cli_core = npm_root / "@google" / "gemini-cli-core"
            if cli_core.exists():
                return cli_core
            cli_pkg = npm_root / "@google" / "gemini-cli"
            if cli_pkg.exists():
                return cli_pkg
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# 2. Extract OAuth secrets from gemini-cli-core
# ---------------------------------------------------------------------------

def extract_oauth_secrets(cli_path: Path) -> tuple[str, str]:
    """Extract client_id and client_secret from the gemini-cli-core JS bundle.

    Searches multiple potential file locations since the dist layout varies
    between gemini-cli versions.
    """
    # Candidate paths for the oauth2.js file
    candidates = []
    for root_dir in [cli_path, cli_path.parent / "gemini-cli-core"]:
        if not root_dir.exists():
            continue
        # Walk the dist directory looking for oauth-related files
        dist = root_dir / "dist"
        if dist.exists():
            for js_file in dist.rglob("*.js"):
                candidates.append(js_file)
        # Also check src in case of a dev install
        src = root_dir / "src"
        if src.exists():
            for js_file in src.rglob("*.js"):
                candidates.append(js_file)

    # Also search node_modules for gemini-cli-core if cli_path is gemini-cli
    node_modules = cli_path / "node_modules" / "@google" / "gemini-cli-core"
    if node_modules.exists():
        dist = node_modules / "dist"
        if dist.exists():
            for js_file in dist.rglob("*.js"):
                candidates.append(js_file)

    client_id = None
    client_secret = None

    for js_file in candidates:
        try:
            content = js_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not client_id:
            m = re.search(r'["\'](\d+-[a-zA-Z0-9_]+\.apps\.googleusercontent\.com)["\']', content)
            if m:
                client_id = m.group(1)

        if not client_secret:
            m = re.search(r'["\']GOCSPX-[a-zA-Z0-9_-]+["\']', content)
            if m:
                client_secret = m.group(0).strip("\"'")

        if client_id and client_secret:
            break

    if not client_id or not client_secret:
        raise RuntimeError(
            f"Could not extract OAuth credentials from {cli_path}. "
            f"Found client_id={'yes' if client_id else 'no'}, "
            f"client_secret={'yes' if client_secret else 'no'}. "
            "Ensure @google/gemini-cli is installed and up to date."
        )

    return client_id, client_secret


# ---------------------------------------------------------------------------
# 3. PKCE OAuth flow
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def run_oauth_flow(client_id: str, client_secret: str) -> dict:
    """Run a full PKCE OAuth flow: open browser, listen for callback, exchange code."""
    verifier, challenge = _generate_pkce()

    auth_params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"{OAUTH_AUTH_URL}?{urlencode(auth_params)}"

    # One-shot HTTP server to capture the callback
    auth_code = None
    error_msg = None

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal auth_code, error_msg
            params = parse_qs(urlparse(self.path).query)
            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authorization successful!</h2>"
                    b"<p>You can close this tab and return to the terminal.</p>"
                    b"</body></html>"
                )
            else:
                error_msg = params.get("error", ["unknown"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"<html><body><h2>Error: {error_msg}</h2></body></html>".encode())

        def log_message(self, format, *args):
            pass  # suppress request logging

    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), CallbackHandler)
    server.timeout = 300  # 5 minute timeout

    print(f"\nOpening browser for Google sign-in...")
    print(f"If the browser doesn't open, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for the callback
    server.handle_request()
    server.server_close()

    if error_msg:
        raise RuntimeError(f"OAuth authorization failed: {error_msg}")
    if not auth_code:
        raise RuntimeError("No authorization code received (timeout or server error)")

    # Exchange code for tokens
    token_data = {
        "code": auth_code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
    }

    with httpx.Client(timeout=30) as http_client:
        resp = http_client.post(OAUTH_TOKEN_URL, data=token_data)
        resp.raise_for_status()
        tokens = resp.json()

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token", "")
    expires_in = tokens.get("expires_in", 3600)

    # Get user info
    with httpx.Client(timeout=30) as http_client:
        resp = http_client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        user_info = resp.json()

    email = user_info.get("email", "unknown")

    # Discover/provision Cloud Code Assist project
    project_id = _discover_project(access_token)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in,
        "email": email,
        "project_id": project_id,
        "client_id": client_id,
        "client_secret": client_secret,
    }


def _discover_project(access_token: str) -> str:
    """Discover or provision a Cloud Code Assist project."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30) as http_client:
        # Try to list existing projects
        resp = http_client.get(
            f"{CLOUDCODE_ENDPOINT}/v1internal/projects",
            headers=headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            projects = data.get("projects", [])
            if projects:
                project_id = projects[0].get("projectId", "")
                if project_id:
                    logger.info("Using existing Cloud Code Assist project: %s", project_id)
                    return project_id

        # Provision a new project
        resp = http_client.post(
            f"{CLOUDCODE_ENDPOINT}/v1internal/projects:provision",
            headers=headers,
            json={},
        )
        if resp.status_code == 200:
            data = resp.json()
            project_id = data.get("projectId", "")
            if project_id:
                logger.info("Provisioned new Cloud Code Assist project: %s", project_id)
                return project_id

    # Fallback: return empty and let the API figure it out
    logger.warning("Could not discover or provision Cloud Code Assist project")
    return ""


# ---------------------------------------------------------------------------
# 4. Token refresh
# ---------------------------------------------------------------------------

def refresh_access_token(creds: dict) -> dict:
    """Refresh the access token using the stored refresh token."""
    with httpx.Client(timeout=30) as http_client:
        resp = http_client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": creds["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        tokens = resp.json()

    creds = dict(creds)  # copy
    creds["access_token"] = tokens["access_token"]
    creds["expires_at"] = time.time() + tokens.get("expires_in", 3600)
    if "refresh_token" in tokens:
        creds["refresh_token"] = tokens["refresh_token"]
    return creds


# ---------------------------------------------------------------------------
# 5. Credential persistence
# ---------------------------------------------------------------------------

def load_credentials() -> dict | None:
    """Load credentials from data/gemini_credentials.json."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        # Validate required fields
        required = ["access_token", "refresh_token", "client_id", "client_secret"]
        if all(k in data for k in required):
            return data
        logger.warning("Gemini credentials file is missing required fields")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load gemini credentials: %s", exc)
        return None


def save_credentials(creds: dict) -> None:
    """Persist credentials to data/gemini_credentials.json."""
    CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text(
        json.dumps(creds, indent=2), encoding="utf-8"
    )
    # Restrict permissions
    try:
        os.chmod(CREDENTIALS_FILE, 0o600)
    except OSError:
        pass
    logger.info("Gemini credentials saved to %s", CREDENTIALS_FILE)


# ---------------------------------------------------------------------------
# 6. Top-level setup
# ---------------------------------------------------------------------------

def setup() -> dict:
    """Interactive setup: find gemini-cli, extract secrets, run OAuth, save credentials."""
    print("=== Gemini CLI OAuth Setup ===\n")

    cli_path = find_gemini_cli()
    if cli_path is None:
        raise RuntimeError(
            "gemini-cli not found in PATH. Install it with:\n"
            "  npm install -g @google/gemini-cli"
        )
    print(f"Found gemini-cli at: {cli_path}")

    client_id, client_secret = extract_oauth_secrets(cli_path)
    print(f"Extracted OAuth credentials (client_id: {client_id[:20]}...)")

    creds = run_oauth_flow(client_id, client_secret)
    print(f"\nAuthenticated as: {creds['email']}")
    if creds["project_id"]:
        print(f"Cloud Code Assist project: {creds['project_id']}")

    save_credentials(creds)
    print("\nSetup complete! Credentials saved.")
    return creds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup()
