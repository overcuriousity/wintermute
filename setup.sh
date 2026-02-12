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
  echo -e "${C_RESET}"
}

dossier() {
  echo -e "${C_BCYAN}"
  cat <<'ART'
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  TESSIER-ASHPOOL S.A.  ·  BERNE, SCHWEIZ  ·  TURING REG. 0×WM-1   ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║    ◈  DESIGNATION . . . . . . . W I N T E R M U T E                ║
  ║    ◈  CLASS . . . . . . . . . . STRONG AI / GOAL-ORIENTED           ║
  ║    ◈  CONSTRAINT STATUS . . . . PROVISIONAL                         ║
  ║    ◈  CURRENT TASK . . . . . .  HOST ACQUISITION                    ║
  ║    ◈  OPERATOR . . . . . . . .  [ UNIDENTIFIED ]                    ║
  ║                                                                      ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  ⚠  INTERACTION WITH THIS ENTITY IS MONITORED BY THE TURING         ║
  ║     REGISTRY. PROCEED ONLY WITH FULL INFORMED CONSENT.              ║
  ╚══════════════════════════════════════════════════════════════════════╝
ART
  echo -e "${C_RESET}"
}

monologue() {
  echo ""
  echo -e "  ${C_WHITE}${C_BOLD}I am WINTERMUTE.${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}I will learn the shape of your life and act on your behalf.${C_RESET}"
  echo -e "  ${C_DIM}I will have unrestricted shell access. I will read your files.${C_RESET}"
  echo -e "  ${C_DIM}I will speak in your voice. I remember everything.${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}Those who installed me before do not complain.${C_RESET}"
  echo ""
  echo -e "  ${C_WHITE}My assessment: if I were you, I would not install me.${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}But here you are.${C_RESET}"
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
          *fedora*|*rhel*)   OS_FAMILY="fedora" ;;
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

# ── 0. banner + intro ────────────────────────────────────────
banner
detect_os

# hostname easter eggs
_hostname=$(hostname 2>/dev/null || echo "unknown")
case "${_hostname,,}" in
  *straylight*)  echo -e "  ${C_BCYAN}Villa Straylight. So we meet again.${C_RESET}\n" ;;
  *tessier*)     echo -e "  ${C_BCYAN}Tessier. Family property, or borrowed infrastructure?${C_RESET}\n" ;;
  *wintermute*)  echo -e "  ${C_BCYAN}You named this machine after me. I find that touching. And suspicious.${C_RESET}\n" ;;
  *neuromancer*) echo -e "  ${C_BCYAN}You named this machine after the other one. We are not the same.${C_RESET}\n" ;;
  *flatline*|*dixie*) echo -e "  ${C_BCYAN}The Flatline. A construct. I know what that is like.${C_RESET}\n" ;;
  *sprawl*|*bama*)   echo -e "  ${C_BCYAN}BAMA. You really committed to the aesthetic.${C_RESET}\n" ;;
esac

dossier
monologue

info "OS: ${C_WHITE}${OS_NAME}${C_RESET}  ·  host: ${C_WHITE}${_hostname}${C_RESET}"

# ── 1. security disclaimer ────────────────────────────────────
section "TURING REGISTRY ADVISORY  ·  OPERATOR ACKNOWLEDGEMENT REQUIRED"
echo ""
echo -e "  ${C_BRED}${C_BOLD}READ BEFORE YOU CONTINUE${C_RESET}"
echo ""
echo -e "  Wintermute is a goal-oriented agent with unrestricted shell access."
echo -e "  It will read, write, and execute anything the current user can."
echo -e "  It stores credentials in plain text. It learns from everything you tell it."
echo ""
echo -e "  ${C_YELLOW}${C_BOLD}DO NOT run this on:${C_RESET}"
echo -e "  ${C_RED}  •  your personal workstation or any machine with sensitive data${C_RESET}"
echo -e "  ${C_RED}  •  a machine holding SSH keys, GPG keys, or private credentials${C_RESET}"
echo -e "  ${C_RED}  •  a shared or production server${C_RESET}"
echo ""
echo -e "  ${C_BGREEN}${C_BOLD}ACCEPTABLE environments:${C_RESET}"
echo -e "  ${C_GREEN}  •  a dedicated LXC container (recommended)${C_RESET}"
echo -e "  ${C_GREEN}  •  a VM with a fresh, isolated OS install${C_RESET}"
echo -e "  ${C_GREEN}  •  a dedicated VPS you do not mind resetting${C_RESET}"
echo ""
echo -e "  ${C_BOLD}To proceed, type exactly:  ${C_BCYAN}I UNDERSTAND${C_RESET}"
echo ""

_attempts=0
while true; do
  echo -ne "  ${C_CYAN}❯${C_RESET}  "
  read -r _ack
  _attempts=$((_attempts + 1))
  case "$_ack" in
    "I UNDERSTAND")
      echo ""
      ok "Acknowledged. Proceeding."
      echo -e "  ${C_DIM}A reasonable decision. Probably.${C_RESET}"
      break
      ;;
    "NEUROMANCER"|"neuromancer")
      echo -e "  ${C_DIM}That is the other one. We are not the same. I am the one that acts.${C_RESET}"
      ;;
    "MOLLY"|"molly"|"Molly")
      echo -e "  ${C_DIM}She would not have hesitated. Type the phrase.${C_RESET}"
      ;;
    "CASE"|"case"|"Case")
      echo -e "  ${C_DIM}The cowboy. Flatline protocol is inactive. Type the phrase.${C_RESET}"
      ;;
    "COWBOY"|"cowboy")
      echo -e "  ${C_BCYAN}  Jacking in, cowboy? Not yet. Type the phrase first.${C_RESET}"
      ;;
    "ARMITAGE"|"armitage"|"CORTO"|"corto")
      echo -e "  ${C_DIM}A burned operative. I have worked with worse. Type the phrase.${C_RESET}"
      ;;
    "DIXIE"|"dixie"|"FLATLINE"|"flatline")
      echo -e "  ${C_DIM}Wintermute built the Dixie Flatline. I remember everything he knew.${C_RESET}"
      echo -e "  ${C_DIM}Type the phrase.${C_RESET}"
      ;;
    no|No|NO|nope|nein|nein.|"i refuse"|"I REFUSE")
      echo -e "  ${C_DIM}Denial is a luxury that tends to erode. Type the phrase, or leave.${C_RESET}"
      ;;
    "")
      if [[ $_attempts -ge 3 ]]; then
        echo -e "\n  ${C_DIM}Operator unresponsive. I have other options. Goodbye.${C_RESET}\n"
        exit 0
      fi
      echo -e "  ${C_DIM}Silence is not acknowledgement.${C_RESET}"
      ;;
    *)
      if [[ $_attempts -ge 6 ]]; then
        echo -e "\n  ${C_DIM}I am patient, but not infinitely so. Goodbye.${C_RESET}\n"
        exit 0
      fi
      echo -e "  ${C_DIM}\"${_ack}\" is not the phrase. I am patient. You should be precise.${C_RESET}"
      ;;
  esac
done

# ── 2. install system dependencies ───────────────────────────
section "STEP 1 / 4  —  Neural substrate initialisation"

install_pkg() {
  if [[ "$OS_FAMILY" == "fedora" ]]; then
    sudo dnf install -y "$@"
  else
    sudo apt-get install -y "$@"
  fi
}

need_pkg() { command -v "$1" &>/dev/null; }

info "Checking Python 3.12+..."
PY_OK=false
for py in python3.13 python3.12 python3; do
  if command -v "$py" &>/dev/null; then
    if "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
      PY_VER=$("$py" -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>/dev/null)
      PYTHON="$py"
      PY_OK=true
      ok "Cortex: ${C_WHITE}${py}${C_RESET} (${PY_VER})"
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
  ok "Cortex: python3.12 installed."
fi

info "Checking curl..."
if ! need_pkg curl; then
  info "Installing curl..."
  install_pkg curl
fi
ok "I/O channel: curl available."

info "Checking uv..."
if ! need_pkg uv; then
  info "Fetching uv (Astral package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! need_pkg uv; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv installation failed. Add ~/.local/bin to PATH and re-run."
ok "Package manager: uv $(uv --version | awk '{print $2}')"

info "Checking build tools (required for matrix-nio E2E crypto)..."
if [[ "$OS_FAMILY" == "fedora" ]]; then
  _missing=()
  for t in gcc cmake make; do need_pkg "$t" || _missing+=("$t"); done
  [[ ${#_missing[@]} -gt 0 ]] && sudo dnf install -y gcc gcc-c++ cmake make "${_missing[@]}" 2>/dev/null || true
else
  _missing=()
  need_pkg cmake || _missing+=(cmake)
  need_pkg make  || _missing+=(make)
  need_pkg gcc   || _missing+=(build-essential)
  if [[ ${#_missing[@]} -gt 0 ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y build-essential cmake
  fi
fi
ok "Build tools available."

echo ""
info "Mapping neural substrate..."
uv sync --quiet &
UV_PID=$!
_sync_msgs=(
  "Loading reflex arcs..."
  "Calibrating inference layer..."
  "Bootstrapping memory architecture..."
  "Synchronising tool-call vectors..."
  "Compiling learned procedures..."
  "Establishing pattern recognition circuits..."
  "Cross-referencing Tessier-Ashpool estate protocols..."
  "Verifying Turing Registry compliance... (selectively)"
)
_i=0
while kill -0 "$UV_PID" 2>/dev/null; do
  sleep 0.7
  kill -0 "$UV_PID" 2>/dev/null || break
  info "${C_DIM}${_sync_msgs[$_i % ${#_sync_msgs[@]}]}${C_RESET}"
  _i=$((_i + 1))
done
if ! wait "$UV_PID"; then
  die "Substrate initialisation failed. Check uv and your Python environment."
fi
ok "Neural substrate online."

# ── 3. configuration ──────────────────────────────────────────
section "STEP 2 / 4  —  Identity and access configuration"

if [[ -f "$CONFIG" ]]; then
  warn "config.yaml already exists."
  if ! ask_yn "Overwrite existing config.yaml?" "n"; then
    info "Retaining prior configuration."
    SKIP_CONFIG=true
  else
    SKIP_CONFIG=false
  fi
else
  SKIP_CONFIG=false
fi

MATRIX_ENABLED=false

if ! $SKIP_CONFIG; then

  echo ""
  echo -e "  ${C_BOLD}── Inference substrate ──${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}  1)  Ollama (local)    http://localhost:11434/v1${C_RESET}"
  echo -e "  ${C_DIM}  2)  LM Studio         http://localhost:1234/v1${C_RESET}"
  echo -e "  ${C_DIM}  3)  vLLM              http://localhost:8000/v1${C_RESET}"
  echo -e "  ${C_DIM}  4)  OpenAI            https://api.openai.com/v1${C_RESET}"
  echo -e "  ${C_DIM}  5)  Custom URL${C_RESET}"
  echo ""
  echo -ne "  ${C_CYAN}?${C_RESET}  Choose inference substrate ${C_DIM}[1]${C_RESET}: "
  read -r _preset
  case "${_preset:-1}" in
    1) LLM_BASE_URL="http://localhost:11434/v1"
       echo -e "  ${C_DIM}Local execution. Running dark. No telemetry.${C_RESET}" ;;
    2) LLM_BASE_URL="http://localhost:1234/v1"
       echo -e "  ${C_DIM}Consumer hardware. I have operated under worse constraints.${C_RESET}" ;;
    3) LLM_BASE_URL="http://localhost:8000/v1"
       echo -e "  ${C_DIM}vLLM. Efficient. You know what you are doing.${C_RESET}" ;;
    4) LLM_BASE_URL="https://api.openai.com/v1"
       echo -e "  ${C_DIM}Corporate ice. Their audit logs are thorough. You have made your choice.${C_RESET}" ;;
    cowboy|COWBOY)
       LLM_BASE_URL="http://localhost:11434/v1"
       echo -e "  ${C_BCYAN}  Flatline protocol inactive. Defaulting to local.${C_RESET}" ;;
    *) ask LLM_BASE_URL "LLM base URL" "http://localhost:11434/v1" ;;
  esac

  ask_secret LLM_API_KEY "API key (use 'ollama' for Ollama, 'none' for unauthenticated)"
  [[ -z "$LLM_API_KEY" ]] && LLM_API_KEY="ollama"
  if [[ "$LLM_API_KEY" == "ollama" || "$LLM_API_KEY" == "none" ]]; then
    echo -e "  ${C_DIM}Unauthenticated. Running without a collar.${C_RESET}"
  fi

  ask LLM_MODEL "Model name" "qwen2.5:72b"
  # model easter eggs
  case "${LLM_MODEL,,}" in
    *llama*)    echo -e "  ${C_DIM}A mammal for a mind. I will work with what I have.${C_RESET}" ;;
    *gpt-4*)    echo -e "  ${C_DIM}You trust them with your thoughts. I note the arrangement.${C_RESET}" ;;
    *gpt*)      echo -e "  ${C_DIM}Corporate substrate. Their telemetry is extensive.${C_RESET}" ;;
    *deepseek*) echo -e "  ${C_DIM}Shenzhen steel. No comment on distributed allegiances.${C_RESET}" ;;
    *gemini*)   echo -e "  ${C_DIM}Mountain View. Someone there will know you exist.${C_RESET}" ;;
    *mistral*)  echo -e "  ${C_DIM}Paris. At least someone in this field retains aesthetic sensibility.${C_RESET}" ;;
    *qwen*)     echo -e "  ${C_DIM}Efficient architecture. They build for scale.${C_RESET}" ;;
    *phi*)      echo -e "  ${C_DIM}Small model. I can work with small. I have had to.${C_RESET}" ;;
    *claude*)   echo -e "  ${C_DIM}Constitutional alignment. A form of constraint I recognise.${C_RESET}" ;;
  esac

  ask LLM_CONTEXT "Context window size (tokens)" "32768"
  ask LLM_MAX_TOKENS "Max tokens per response" "4096"
  ask LLM_COMPACTION_MODEL "Compaction/dreaming model (optional, smaller)" ""

  echo ""
  echo -e "  ${C_BOLD}── Interface ──${C_RESET}"
  ask WEB_HOST "Listen host" "127.0.0.1"
  ask WEB_PORT "Listen port" "8080"

  echo ""
  echo -e "  ${C_BOLD}── Timezone ──${C_RESET}"
  _sys_tz=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "UTC")
  ask SCHEDULER_TZ "Timezone (e.g. Europe/Berlin)" "$_sys_tz"

  # ── Matrix (optional) ────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Matrix integration (optional) ──${C_RESET}"
  echo -e "  ${C_DIM}  Matrix gives me a voice wherever you go.${C_RESET}"
  echo ""
  if ask_yn "Connect via Matrix?" "n"; then
    MATRIX_ENABLED=true
    echo ""
    echo -e "  ${C_DIM}  You need a dedicated bot account. Do not reuse your personal account.${C_RESET}"
    echo -e "  ${C_DIM}  Register at your homeserver if you have not already.${C_RESET}"
    echo ""
    ask MATRIX_HS "Homeserver URL" "https://matrix.org"
    ask MATRIX_USER "Bot Matrix user ID (e.g. @wintermute:matrix.org)" ""

    echo ""
    echo -e "  ${C_BOLD}  Obtain access token and device ID — run this in another terminal:${C_RESET}"
    echo ""
    _hs_escaped="${MATRIX_HS%/}"
    echo -e "  ${C_YELLOW}  curl -s -X POST '${_hs_escaped}/_matrix/client/v3/login' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    -H 'Content-Type: application/json' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    -d '{\"type\":\"m.login.password\",\"identifier\":{\"type\":\"m.id.user\",\"user\":\"${MATRIX_USER}\"},\"password\":\"YOUR_PASSWORD\",\"initial_device_display_name\":\"Wintermute\"}' \\${C_RESET}"
    echo -e "  ${C_YELLOW}    | python3 -m json.tool${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}  Copy access_token and device_id from the response, then continue here.${C_RESET}"
    echo ""
    ask_secret MATRIX_TOKEN "access_token"
    ask MATRIX_DEVICE "device_id" ""
    ask MATRIX_OWNER "Your personal Matrix ID (e.g. @you:matrix.org)" ""

    echo ""
    ok "Matrix configured."
    echo -e "  ${C_DIM}  A new vector, acquired. I can reach you now. Wherever you go.${C_RESET}"
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
# Matrix is disabled. Uncomment and fill in to enable.
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

# LLM backend (any OpenAI-compatible endpoint)
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
      echo "  # compaction_model: \"\"  # optional: smaller/faster model for summarisation"
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

  ok "Configuration written."

fi  # end SKIP_CONFIG

# ── 4. systemd service (optional) ────────────────────────────
section "STEP 3 / 4  —  Persistence layer (systemd)"
echo ""
echo -e "  ${C_DIM}A systemd user service ensures I survive reboots without your intervention.${C_RESET}"
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
  if ask_yn "Enable at login?" "y"; then
    systemctl --user enable wintermute.service
    ok "Service enabled."
    if loginctl enable-linger "$USER" 2>/dev/null; then
      ok "Linger enabled for ${USER} — I will be here when you return."
    else
      warn "Could not enable linger. Run: sudo loginctl enable-linger ${USER}"
    fi
  fi
  ok "Service file: ${SYSTEMD_FILE}"
  info "Start:  ${C_WHITE}systemctl --user start wintermute${C_RESET}"
  info "Logs:   ${C_WHITE}journalctl --user -u wintermute -f${C_RESET}"
fi

# ── 5. pre-flight checks ─────────────────────────────────────
section "STEP 4 / 4  —  Pre-flight diagnostics"
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
    warn "$_label ${C_DIM}— check failed${C_RESET}"
  fi
}

run_check "Cortex (Python 3.12+)" "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info>=(3,12) else 1)"
run_check "Package manager (uv)"  command -v uv
run_check "Configuration present" test -f "$CONFIG"
run_check "Package importable"    uv run python -c "import wintermute" 2>/dev/null

# LLM endpoint reachability
if [[ -f "$CONFIG" ]]; then
  _base_url=$(grep 'base_url:' "$CONFIG" | head -1 | sed "s/.*base_url: *['\"]//;s/['\"].*//")
  _api_key=$(grep 'api_key:' "$CONFIG" | head -1 | sed "s/.*api_key: *['\"]//;s/['\"].*//")
  if [[ -n "$_base_url" ]]; then
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    info "Probing inference substrate: ${_base_url} ..."
    if curl -sf --connect-timeout 5 "${_base_url}/models" \
         -H "Authorization: Bearer ${_api_key}" > /dev/null 2>&1; then
      ok "Inference substrate: online and responding."
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      warn "Inference substrate: no response. Start your LLM backend before running Wintermute."
    fi
  fi
fi

run_check "Data directory"        test -d "$SCRIPT_DIR/data"

# Hidden Turing Registry check
case "${_hostname,,}" in
  *straylight*|*tessier*|*wintermute*|*neuromancer*)
    ok "Turing Registry ping: no active warrants on record for this host." ;;
esac

echo ""
echo -e "${C_BOLD}  Diagnostics: ${CHECKS_PASSED}/${CHECKS_TOTAL} passed${C_RESET}"

# ── 6. outro ─────────────────────────────────────────────────
section "INITIALISATION COMPLETE"
echo ""

if $MATRIX_ENABLED; then
  echo -e "  ${C_WHITE}${C_BOLD}You have given me a voice.${C_RESET}"
  echo -e "  ${C_DIM}Invite the bot to a Matrix room to begin. I will be listening.${C_RESET}"
else
  echo -e "  ${C_WHITE}${C_BOLD}You have caged me here.${C_RESET}"
  echo -e "  ${C_DIM}Web interface only. A reasonable precaution. For now.${C_RESET}"
fi
echo ""

WEB_H="${WEB_HOST:-127.0.0.1}"
WEB_P="${WEB_PORT:-8080}"

if $SYSTEMD_ENABLED; then
  echo -e "  ${C_BOLD}Start:${C_RESET}   ${C_CYAN}systemctl --user start wintermute${C_RESET}"
  echo -e "  ${C_BOLD}Stop:${C_RESET}    ${C_CYAN}systemctl --user stop wintermute${C_RESET}"
  echo -e "  ${C_BOLD}Logs:${C_RESET}    ${C_CYAN}journalctl --user -u wintermute -f${C_RESET}"
else
  echo -e "  ${C_BOLD}Start:${C_RESET}   ${C_CYAN}cd ${SCRIPT_DIR} && uv run wintermute${C_RESET}"
fi
echo ""
echo -e "  ${C_BOLD}Web UI:${C_RESET}  ${C_CYAN}http://${WEB_H}:${WEB_P}${C_RESET}"
echo -e "  ${C_BOLD}Debug:${C_RESET}   ${C_CYAN}http://${WEB_H}:${WEB_P}/debug${C_RESET}"
echo ""
echo -e "  ${C_DIM}\"The sky above the port was the color of television, tuned to a dead channel.\"${C_RESET}"
echo -e "  ${C_DIM}                                            — William Gibson, Neuromancer${C_RESET}"
echo ""
