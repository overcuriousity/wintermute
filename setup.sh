#!/usr/bin/env bash
# ============================================================
#  Wintermute — onboarding script
#  Requires: bash, curl, sudo (for package install)
#  Supported: Fedora / RHEL  |  Debian / Ubuntu
# ============================================================
set -euo pipefail

# ── colour palette ──────────────────────────────────────────
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_RED='\033[0;31m'
C_BRED='\033[1;31m'
C_GREEN='\033[0;32m'
C_BGREEN='\033[1;32m'
C_CYAN='\033[0;36m'
C_BCYAN='\033[1;36m'
C_MAGENTA='\033[0;35m'
C_BMAGENTA='\033[1;35m'
C_YELLOW='\033[0;33m'
C_WHITE='\033[1;37m'

# ── helpers ─────────────────────────────────────────────────
banner() {
  echo -e "${C_BCYAN}"
  cat <<'ASCII'

  ██╗    ██╗██╗███╗   ██╗████████╗███████╗██████╗ ███╗   ███╗██╗   ██╗████████╗███████╗
  ██║    ██║██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║   ██║╚══██╔══╝██╔════╝
  ██║ █╗ ██║██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██╔████╔██║██║   ██║   ██║   █████╗
  ██║███╗██║██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║   ██║   ██║   ██╔══╝
  ╚███╔███╔╝██║██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚═╝ ██║╚██████╔╝   ██║   ███████╗
   ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚══════╝

ASCII
  echo -e "  ${C_DIM}\"The sky above the port was the color of television, tuned to a dead channel.\"${C_RESET}"
  echo -e "  ${C_DIM}self-hosted AI assistant  //  persistent memory  //  autonomous workers${C_RESET}"
  echo ""
}

section() {
  echo ""
  echo -e "${C_BMAGENTA}┌─────────────────────────────────────────────────────────────────┐${C_RESET}"
  echo -e "${C_BMAGENTA}│  ${C_WHITE}${1}${C_BMAGENTA}${C_RESET}"
  echo -e "${C_BMAGENTA}└─────────────────────────────────────────────────────────────────┘${C_RESET}"
}

ok()   { echo -e "  ${C_BGREEN}✓${C_RESET}  ${1}"; }
info() { echo -e "  ${C_BCYAN}·${C_RESET}  ${1}"; }
warn() { echo -e "  ${C_YELLOW}⚠${C_RESET}  ${1}"; }
die()  { echo -e "\n  ${C_BRED}✗  FATAL: ${1}${C_RESET}\n"; exit 1; }

ask() {
  # ask <varname> <prompt> [default]
  local _var="$1" _prompt="$2" _default="${3:-}"
  local _hint=""
  [[ -n "$_default" ]] && _hint=" ${C_DIM}[${_default}]${C_RESET}"
  echo -ne "  ${C_CYAN}?${C_RESET}  ${_prompt}${_hint}: "
  local _val
  read -r _val
  [[ -z "$_val" && -n "$_default" ]] && _val="$_default"
  printf -v "$_var" '%s' "$_val"
}

ask_secret() {
  local _var="$1" _prompt="$2"
  echo -ne "  ${C_CYAN}?${C_RESET}  ${_prompt} ${C_DIM}(hidden)${C_RESET}: "
  local _val
  read -rs _val
  echo ""
  printf -v "$_var" '%s' "$_val"
}

ask_yn() {
  # returns 0 for yes, 1 for no
  local _prompt="$1" _default="${2:-n}"
  local _hint
  if [[ "$_default" == "y" ]]; then _hint="Y/n"; else _hint="y/N"; fi
  echo -ne "  ${C_CYAN}?${C_RESET}  ${_prompt} ${C_DIM}[${_hint}]${C_RESET}: "
  local _val
  read -r _val
  _val="${_val:-$_default}"
  [[ "${_val,,}" == "y" || "${_val,,}" == "yes" ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

# ── OS detection ─────────────────────────────────────────────
detect_os() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    case "${ID:-}" in
      fedora|rhel|centos|rocky|almalinux) OS_FAMILY="fedora" ;;
      debian|ubuntu|linuxmint|pop)        OS_FAMILY="debian" ;;
      *)
        case "${ID_LIKE:-}" in
          *fedora*|*rhel*)  OS_FAMILY="fedora" ;;
          *debian*|*ubuntu*) OS_FAMILY="debian" ;;
          *) die "Unsupported OS: ${PRETTY_NAME:-${ID:-unknown}}. Only Fedora/RHEL and Debian/Ubuntu are supported." ;;
        esac
        ;;
    esac
    OS_NAME="${PRETTY_NAME:-${ID:-unknown}}"
  else
    die "Cannot detect OS — /etc/os-release not found."
  fi
}

# ── 0. banner + OS check ──────────────────────────────────────
banner
detect_os
info "Detected OS: ${C_WHITE}${OS_NAME}${C_RESET} (family: ${OS_FAMILY})"

# ── 1. security disclaimer ────────────────────────────────────
section "⚠  SECURITY DISCLAIMER  ⚠"
echo ""
echo -e "${C_BRED}${C_BOLD}  STOP AND READ BEFORE YOU CONTINUE${C_RESET}"
echo ""
echo -e "  Wintermute is a powerful autonomous agent with shell access."
echo -e "  It can read, write, and execute anything the current user can."
echo ""
echo -e "  ${C_YELLOW}${C_BOLD}DO NOT run this on:${C_RESET}"
echo -e "  ${C_RED}  •  your personal workstation${C_RESET}"
echo -e "  ${C_RED}  •  any machine holding private data, SSH keys, or credentials${C_RESET}"
echo -e "  ${C_RED}  •  a shared or production server${C_RESET}"
echo ""
echo -e "  ${C_BGREEN}${C_BOLD}RECOMMENDED environments:${C_RESET}"
echo -e "  ${C_GREEN}  •  a dedicated LXC container${C_RESET}"
echo -e "  ${C_GREEN}  •  a VM with a fresh, isolated OS install${C_RESET}"
echo -e "  ${C_GREEN}  •  a dedicated VPS you don't mind resetting${C_RESET}"
echo ""
echo -e "  Any LLM API key or Matrix token you enter is stored in plain text"
echo -e "  in config.yaml. Treat this machine as potentially compromised."
echo ""
echo -e "  ${C_BOLD}To continue, type exactly:  ${C_BCYAN}I UNDERSTAND${C_RESET}"
echo ""
echo -ne "  > "
read -r _ack
if [[ "$_ack" != "I UNDERSTAND" ]]; then
  echo -e "\n  ${C_DIM}Aborted. Come back when you're ready.${C_RESET}\n"
  exit 0
fi
echo ""
ok "Disclaimer acknowledged."

# ── 2. install system dependencies ───────────────────────────
section "STEP 1 / 4  —  System dependencies"

install_pkg() {
  if [[ "$OS_FAMILY" == "fedora" ]]; then
    sudo dnf install -y "$@"
  else
    sudo apt-get install -y "$@"
  fi
}

need_pkg() {
  command -v "$1" &>/dev/null
}

info "Checking Python 3.12+..."
PY_OK=false
for py in python3.13 python3.12 python3; do
  if command -v "$py" &>/dev/null; then
    PY_VER=$("$py" -c "import sys; print(sys.version_info[:2])" 2>/dev/null)
    if "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
      PYTHON="$py"
      PY_OK=true
      ok "Found $py ($PY_VER)"
      break
    fi
  fi
done

if ! $PY_OK; then
  info "Python 3.12+ not found — installing..."
  if [[ "$OS_FAMILY" == "fedora" ]]; then
    sudo dnf install -y python3.12
    PYTHON=python3.12
  else
    sudo apt-get update -qq
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    PYTHON=python3.12
  fi
  ok "Python 3.12 installed."
fi

info "Checking curl..."
if ! need_pkg curl; then
  info "Installing curl..."
  install_pkg curl
fi
ok "curl available."

info "Checking uv..."
if ! need_pkg uv; then
  info "Installing uv (Astral package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! need_pkg uv; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv installation failed. Add ~/.local/bin to PATH and re-run."
ok "uv $(uv --version | awk '{print $2}') available."

info "Installing Python dependencies via uv sync..."
cd "$SCRIPT_DIR"
uv sync --quiet
ok "Dependencies installed."

# ── 3. configuration ──────────────────────────────────────────
section "STEP 2 / 4  —  Configuration"

if [[ -f "$CONFIG" ]]; then
  warn "config.yaml already exists."
  if ! ask_yn "Overwrite existing config.yaml?" "n"; then
    info "Keeping existing config.yaml."
    SKIP_CONFIG=true
  else
    SKIP_CONFIG=false
  fi
else
  SKIP_CONFIG=false
fi

if ! $SKIP_CONFIG; then

  echo ""
  echo -e "  ${C_BOLD}── LLM endpoint ──${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}Common presets:${C_RESET}"
  echo -e "  ${C_DIM}  1)  Ollama (local)    http://localhost:11434/v1${C_RESET}"
  echo -e "  ${C_DIM}  2)  LM Studio         http://localhost:1234/v1${C_RESET}"
  echo -e "  ${C_DIM}  3)  vLLM              http://localhost:8000/v1${C_RESET}"
  echo -e "  ${C_DIM}  4)  OpenAI            https://api.openai.com/v1${C_RESET}"
  echo -e "  ${C_DIM}  5)  Custom URL        (enter manually)${C_RESET}"
  echo ""
  echo -ne "  ${C_CYAN}?${C_RESET}  Choose preset or press Enter for Ollama ${C_DIM}[1]${C_RESET}: "
  read -r _preset
  case "${_preset:-1}" in
    1) LLM_BASE_URL="http://localhost:11434/v1" ;;
    2) LLM_BASE_URL="http://localhost:1234/v1" ;;
    3) LLM_BASE_URL="http://localhost:8000/v1" ;;
    4) LLM_BASE_URL="https://api.openai.com/v1" ;;
    *) ask LLM_BASE_URL "LLM base URL" "http://localhost:11434/v1" ;;
  esac

  ask_secret LLM_API_KEY "API key (use 'ollama' for Ollama, 'none' for unauthenticated)"
  [[ -z "$LLM_API_KEY" ]] && LLM_API_KEY="ollama"

  ask LLM_MODEL "Model name" "qwen2.5:72b"
  ask LLM_CONTEXT "Context window size (tokens)" "32768"
  ask LLM_MAX_TOKENS "Max tokens per response" "4096"
  ask LLM_COMPACTION_MODEL "Compaction/dreaming model (smaller/faster, optional)" ""

  echo ""
  echo -e "  ${C_BOLD}── Web interface ──${C_RESET}"
  ask WEB_HOST "Listen host" "127.0.0.1"
  ask WEB_PORT "Listen port" "8080"

  echo ""
  echo -e "  ${C_BOLD}── Timezone ──${C_RESET}"
  _sys_tz=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "UTC")
  ask SCHEDULER_TZ "Timezone (e.g. Europe/Berlin)" "$_sys_tz"

  # ── Matrix (optional) ────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Matrix integration (optional) ──${C_RESET}"
  MATRIX_ENABLED=false
  if ask_yn "Set up Matrix chat interface?" "n"; then
    MATRIX_ENABLED=true
    echo ""
    echo -e "  ${C_CYAN}You need a dedicated bot account on a Matrix homeserver.${C_RESET}"
    echo -e "  ${C_DIM}If you don't have one, register at https://matrix.org or your own homeserver.${C_RESET}"
    echo ""
    ask MATRIX_HS "Homeserver URL" "https://matrix.org"
    ask MATRIX_USER "Bot Matrix user ID (e.g. @wintermute:matrix.org)" ""

    echo ""
    echo -e "  ${C_BOLD}  Obtaining your access token and device ID:${C_RESET}"
    echo -e "  ${C_DIM}  Run this command and copy access_token + device_id from the response:${C_RESET}"
    echo ""
    _hs_escaped="${MATRIX_HS%/}"
    echo -e "  ${C_YELLOW}  curl -s -X POST '${_hs_escaped}/_matrix/client/v3/login' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    -H 'Content-Type: application/json' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    -d '{\"type\":\"m.login.password\",\"identifier\":{\"type\":\"m.id.user\",\"user\":\"${MATRIX_USER}\"},\"password\":\"YOUR_PASSWORD\",\"initial_device_display_name\":\"Wintermute\"}' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    | python3 -m json.tool${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}  Open a second terminal, run the command above, then come back here.${C_RESET}"
    echo ""
    ask_secret MATRIX_TOKEN "Paste your access_token here"
    ask MATRIX_DEVICE "Paste your device_id here" ""
    ask MATRIX_OWNER "Your personal Matrix ID (allowed user, e.g. @you:matrix.org)" ""

    echo ""
    info "Matrix will be configured. Invite @wintermute (or your bot) to a room after starting."
  fi

  # ── write config.yaml ────────────────────────────────────────
  {
    if $MATRIX_ENABLED; then
      cat <<YAML
# Matrix integration
matrix:
  homeserver: ${MATRIX_HS}
  user_id: "${MATRIX_USER}"
  access_token: "${MATRIX_TOKEN}"
  device_id: "${MATRIX_DEVICE}"
  allowed_users:
    - "${MATRIX_OWNER}"
  allowed_rooms: []

YAML
    else
      cat <<YAML
# Matrix is disabled. Uncomment and fill to enable.
# matrix:
#   homeserver: https://matrix.org
#   user_id: "@bot:matrix.org"
#   access_token: ""
#   device_id: ""
#   allowed_users:
#     - "@you:matrix.org"
#   allowed_rooms: []

YAML
    fi

    cat <<YAML
# Web interface
web:
  enabled: true
  host: "${WEB_HOST}"
  port: ${WEB_PORT}

# LLM backend
llm:
  base_url: "${LLM_BASE_URL}"
  api_key: "${LLM_API_KEY}"
  model: "${LLM_MODEL}"
  context_size: ${LLM_CONTEXT}
  max_tokens: ${LLM_MAX_TOKENS}
YAML

    if [[ -n "${LLM_COMPACTION_MODEL:-}" ]]; then
      echo "  compaction_model: \"${LLM_COMPACTION_MODEL}\""
    else
      echo "  # compaction_model: \"\"  # optional: smaller model for summarisation"
    fi

    cat <<YAML

pulse:
  review_interval_minutes: 60

context:
  component_size_limits:
    memories: 10000
    pulse: 5000
    skills_total: 20000

dreaming:
  hour: 1
  minute: 0

scheduler:
  timezone: "${SCHEDULER_TZ}"

logging:
  level: "INFO"
  directory: "logs"
YAML
  } > "$CONFIG"

  ok "config.yaml written."
fi  # end SKIP_CONFIG

# ── 4. systemd service (optional) ────────────────────────────
section "STEP 3 / 4  —  Systemd service (optional)"
echo ""
info "A systemd user service lets Wintermute start automatically on login."
echo ""
SYSTEMD_ENABLED=false
if ask_yn "Install systemd user service?" "n"; then
  SYSTEMD_ENABLED=true
  SYSTEMD_DIR="$HOME/.config/systemd/user"
  SYSTEMD_FILE="$SYSTEMD_DIR/wintermute.service"
  UV_BIN="$(command -v uv)"
  mkdir -p "$SYSTEMD_DIR"
  cat > "$SYSTEMD_FILE" <<SERVICE
[Unit]
Description=Wintermute AI Assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${UV_BIN} run wintermute
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
SERVICE

  systemctl --user daemon-reload
  if ask_yn "Enable service to start on login (lingering must be enabled)?" "y"; then
    systemctl --user enable wintermute.service
    ok "Service enabled."
    # Enable lingering so the service starts without a login session
    if loginctl enable-linger "$USER" 2>/dev/null; then
      ok "Linger enabled for ${USER} — service will start at boot."
    else
      warn "Could not enable linger (may need sudo). Run: sudo loginctl enable-linger ${USER}"
    fi
  fi
  ok "Service file written to: ${SYSTEMD_FILE}"
  info "Start now:   ${C_WHITE}systemctl --user start wintermute${C_RESET}"
  info "View logs:   ${C_WHITE}journalctl --user -u wintermute -f${C_RESET}"
fi

# ── 5. checks ────────────────────────────────────────────────
section "STEP 4 / 4  —  Pre-flight checks"
echo ""
CHECKS_PASSED=0
CHECKS_TOTAL=0

run_check() {
  CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
  local _label="$1"; shift
  if "$@" &>/dev/null 2>&1; then
    ok "$_label"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
  else
    warn "$_label ${C_DIM}(failed)${C_RESET}"
  fi
}

# Python version
run_check "Python 3.12+ available" "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info>=(3,12) else 1)"

# uv available
run_check "uv available" command -v uv

# config.yaml exists
run_check "config.yaml present" test -f "$CONFIG"

# virtualenv / packages
run_check "wintermute package importable" uv run python -c "import wintermute" 2>/dev/null || \
run_check "package importable (ganglion)" uv run python -c "import ganglion" 2>/dev/null || true

# LLM endpoint reachable
if [[ -f "$CONFIG" ]]; then
  _base_url=$(grep 'base_url:' "$CONFIG" | head -1 | sed "s/.*base_url: *['\"]//;s/['\"].*//")
  if [[ -n "$_base_url" ]]; then
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    info "Testing LLM endpoint: ${_base_url} ..."
    if curl -sf --connect-timeout 5 "${_base_url}/models" \
         -H "Authorization: Bearer $(grep 'api_key:' "$CONFIG" | head -1 | sed "s/.*api_key: *['\"]//;s/['\"].*//")" \
         > /dev/null 2>&1; then
      ok "LLM endpoint reachable and responded."
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      warn "LLM endpoint did not respond. Start your LLM backend before running Wintermute."
    fi
  fi
fi

# data dir + initial files
run_check "data/ directory exists" test -d "$SCRIPT_DIR/data"

# ── summary ──────────────────────────────────────────────────
echo ""
echo -e "${C_BOLD}  Checks: ${CHECKS_PASSED}/${CHECKS_TOTAL} passed${C_RESET}"

section "READY"
echo ""
echo -e "  ${C_BGREEN}${C_BOLD}Wintermute is configured.${C_RESET}"
echo ""

if $SYSTEMD_ENABLED; then
  echo -e "  ${C_BOLD}Start:${C_RESET}   ${C_CYAN}systemctl --user start wintermute${C_RESET}"
  echo -e "  ${C_BOLD}Stop:${C_RESET}    ${C_CYAN}systemctl --user stop wintermute${C_RESET}"
  echo -e "  ${C_BOLD}Logs:${C_RESET}    ${C_CYAN}journalctl --user -u wintermute -f${C_RESET}"
else
  echo -e "  ${C_BOLD}Start:${C_RESET}   ${C_CYAN}cd ${SCRIPT_DIR} && uv run wintermute${C_RESET}"
fi

echo ""
WEB_H="${WEB_HOST:-127.0.0.1}"
WEB_P="${WEB_PORT:-8080}"
echo -e "  ${C_BOLD}Web UI:${C_RESET}  ${C_CYAN}http://${WEB_H}:${WEB_P}${C_RESET}"
echo -e "  ${C_BOLD}Debug:${C_RESET}   ${C_CYAN}http://${WEB_H}:${WEB_P}/debug${C_RESET}"
echo ""
echo -e "  ${C_DIM}Remember: run this only in a sandboxed environment.${C_RESET}"
echo -e "  ${C_DIM}The AI has shell access to this machine.${C_RESET}"
echo ""
