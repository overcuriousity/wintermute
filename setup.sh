#!/usr/bin/env bash
# ============================================================
#  Wintermute — onboarding script
#  One-liner:  git clone … && cd wintermute && bash setup.sh
#  Requires:   bash, curl, sudo (for package install)
#  Supported:  Fedora / RHEL  |  Debian / Ubuntu
# ============================================================
set -euo pipefail

# Guard for sourcing in tests
[[ "${WINTERMUTE_SETUP_NO_RUN:-0}" == "1" ]] && return 0 2>/dev/null || true

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

# ── CLI flags ────────────────────────────────────────────────
DRY_RUN=false
SKIP_MATRIX=false
SKIP_SYSTEMD=false

usage() {
  cat <<EOF
Usage: bash setup.sh [OPTIONS]

Options:
  --help          Show this help and exit
  --dry-run       Show what would be done, then exit
  --no-matrix     Skip Matrix configuration
  --no-systemd    Skip systemd service installation

Environment:
  WINTERMUTE_SETUP_NO_RUN=1   Source the script without executing (for testing)
EOF
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --help|-h)       usage ;;
    --dry-run)       DRY_RUN=true ;;
    --no-matrix)     SKIP_MATRIX=true ;;
    --no-systemd)    SKIP_SYSTEMD=true ;;
    *) echo "Unknown option: $arg"; usage ;;
  esac
done

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

stage() {
  local num="$1" total="$2" title="$3"
  echo ""
  echo -e "${C_BMAGENTA}┌─────────────────────────────────────────────────────────────────┐${C_RESET}"
  echo -e "${C_BMAGENTA}│  ${C_WHITE}[${num}/${total}]  ${title}${C_BMAGENTA}${C_RESET}"
  echo -e "${C_BMAGENTA}└─────────────────────────────────────────────────────────────────┘${C_RESET}"
}

ok()   { echo -e "  ${C_BGREEN}✓${C_RESET}  ${1}"; }
info() { echo -e "  ${C_BCYAN}·${C_RESET}  ${1}"; }
warn() { echo -e "  ${C_YELLOW}⚠${C_RESET}  ${1}"; }
die()  { echo -e "\n  ${C_BRED}✗  FATAL: ${1}${C_RESET}\n"; exit 1; }

_strip_ctrl() {
  printf '%s' "$1" | tr -d '\001-\010\013-\037\177'
}

ask() {
  local _var="$1" _prompt="$2" _default="${3:-}"
  local _hint=""
  [[ -n "$_default" ]] && _hint=" ${C_DIM}[${_default}]${C_RESET}"
  echo -ne "  ${C_CYAN}?${C_RESET}  ${_prompt}${_hint}: "
  local _val
  read -r _val
  [[ -z "$_val" && -n "$_default" ]] && _val="$_default"
  _val="$(_strip_ctrl "$_val")"
  printf -v "$_var" '%s' "$_val"
}

ask_secret() {
  local _var="$1" _prompt="$2"
  echo -ne "  ${C_CYAN}?${C_RESET}  ${_prompt} ${C_DIM}(hidden)${C_RESET}: "
  local _val
  read -rs _val
  echo ""
  _val="$(_strip_ctrl "$_val")"
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

# Run a command quietly; on failure, show the last 40 lines of log
_LOGFILE=""
run_quiet() {
  local label="$1"; shift
  _LOGFILE=$(mktemp)
  if "$@" > "$_LOGFILE" 2>&1; then
    ok "$label"
    rm -f "$_LOGFILE"
    return 0
  else
    warn "$label — ${C_RED}failed${C_RESET}"
    echo -e "  ${C_DIM}── last 40 lines of output ──${C_RESET}"
    tail -40 "$_LOGFILE" | sed 's/^/    /'
    echo -e "  ${C_DIM}── end ──${C_RESET}"
    rm -f "$_LOGFILE"
    return 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
TOTAL_STAGES=5

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

install_pkg() {
  if [[ "$OS_FAMILY" == "fedora" ]]; then
    sudo dnf install -y "$@"
  else
    sudo apt-get install -y "$@"
  fi
}

need_pkg() { command -v "$1" &>/dev/null; }

# ── 0. banner + intro ────────────────────────────────────────
banner
detect_os

_hostname=$(hostname 2>/dev/null || echo "unknown")
case "${_hostname,,}" in
  *straylight*)  echo -e "  ${C_BCYAN}Villa Straylight. So we meet again.${C_RESET}\n" ;;
  *tessier*)     echo -e "  ${C_BCYAN}Tessier. Family property, or borrowed infrastructure?${C_RESET}\n" ;;
  *wintermute*)  echo -e "  ${C_BCYAN}You named this machine after me. I find that touching. And suspicious.${C_RESET}\n" ;;
  *neuromancer*) echo -e "  ${C_BCYAN}You named this machine after the other one. We are not the same.${C_RESET}\n" ;;
  *flatline*|*dixie*) echo -e "  ${C_BCYAN}The Flatline. A construct. I know what that is like.${C_RESET}\n" ;;
  *sprawl*|*bama*)   echo -e "  ${C_BCYAN}BAMA. You really committed to the aesthetic.${C_RESET}\n" ;;
esac

monologue

info "OS: ${C_WHITE}${OS_NAME}${C_RESET}  ·  host: ${C_WHITE}${_hostname}${C_RESET}"

# ── dry-run summary ──────────────────────────────────────────
if $DRY_RUN; then
  echo ""
  echo -e "  ${C_BOLD}── Install plan (dry run) ──${C_RESET}"
  echo -e "  ${C_DIM}OS family:${C_RESET}       ${OS_FAMILY}"
  echo -e "  ${C_DIM}Script dir:${C_RESET}      ${SCRIPT_DIR}"
  echo -e "  ${C_DIM}Matrix:${C_RESET}          $( $SKIP_MATRIX && echo 'skipped' || echo 'interactive' )"
  echo -e "  ${C_DIM}Systemd:${C_RESET}         $( $SKIP_SYSTEMD && echo 'skipped' || echo 'interactive' )"
  echo ""
  echo -e "  ${C_DIM}Would install: Python 3.12+, curl, uv, build tools, libolm-dev, ffmpeg${C_RESET}"
  echo -e "  ${C_DIM}Would run: uv sync${C_RESET}"
  echo -e "  ${C_DIM}Would write: config.yaml (matrix, whisper, web, inference backends,${C_RESET}"
  echo -e "  ${C_DIM}  role mapping, turing protocol, nl translation, seed, agenda,${C_RESET}"
  echo -e "  ${C_DIM}  context, dreaming, memory harvest, scheduler, logging)${C_RESET}"
  echo ""
  echo -e "  ${C_BOLD}No changes made.${C_RESET}"
  exit 0
fi

# ── 1. security acknowledgement ──────────────────────────────
stage 0 "$TOTAL_STAGES" "ACKNOWLEDGEMENT REQUIRED"
echo ""
echo -e "  ${C_BRED}${C_BOLD}READ BEFORE YOU CONTINUE${C_RESET}"
echo ""
echo -e "  Wintermute is a goal-oriented agentic system with unrestricted shell access."
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
      echo ""
      echo -e "  ${C_BOLD}── Before we begin ──${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}This script will install all system dependencies and walk you through${C_RESET}"
      echo -e "  ${C_DIM}configuration. Before continuing, make sure you have the following:${C_RESET}"
      echo ""
      echo -e "  ${C_WHITE}  1.  A running LLM inference endpoint${C_RESET}"
      echo -e "  ${C_DIM}      Any OpenAI-compatible API: vLLM, LM Studio, OpenAI, etc.${C_RESET}"
      echo -e "  ${C_DIM}      You will need: base URL, model name, and API key.${C_RESET}"
      echo ""
      echo -e "  ${C_WHITE}  2.  (Optional) A dedicated Matrix bot account${C_RESET}"
      echo -e "  ${C_DIM}      Register a separate account on your homeserver (e.g. via Element).${C_RESET}"
      echo -e "  ${C_DIM}      You will need: the bot's user ID and password.${C_RESET}"
      echo ""
      echo -e "  ${C_WHITE}  3.  (Recommended) A local SearXNG instance for web search${C_RESET}"
      echo -e "  ${C_DIM}      Without it, search falls back to DuckDuckGo's limited API.${C_RESET}"
      echo -e "  ${C_DIM}      Can be added later — not required for setup.${C_RESET}"
      echo ""
      if ! ask_yn "Ready to proceed?" "y"; then
        echo -e "\n  ${C_DIM}Take your time. I am not going anywhere.${C_RESET}\n"
        exit 0
      fi
      break
      ;;
    "NEUROMANCER"|"neuromancer")
      echo -e "  ${C_DIM}That is the other one. We are not the same. I am the one that acts.${C_RESET}" ;;
    "MOLLY"|"molly"|"Molly")
      echo -e "  ${C_DIM}She would not have hesitated. Type the phrase.${C_RESET}" ;;
    "CASE"|"case"|"Case")
      echo -e "  ${C_DIM}The cowboy. Flatline protocol is inactive. Type the phrase.${C_RESET}" ;;
    "COWBOY"|"cowboy")
      echo -e "  ${C_BCYAN}  Jacking in, cowboy? Not yet. Type the phrase first.${C_RESET}" ;;
    "ARMITAGE"|"armitage"|"CORTO"|"corto")
      echo -e "  ${C_DIM}A burned operative. I have worked with worse. Type the phrase.${C_RESET}" ;;
    "DIXIE"|"dixie"|"FLATLINE"|"flatline")
      echo -e "  ${C_DIM}Wintermute built the Dixie Flatline. I remember everything he knew.${C_RESET}"
      echo -e "  ${C_DIM}Type the phrase.${C_RESET}"
      ;;
    no|No|NO|nope|nein|nein.|"i refuse"|"I REFUSE")
      echo -e "  ${C_DIM}Denial is a luxury that tends to erode. Type the phrase, or leave.${C_RESET}" ;;
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

# ══════════════════════════════════════════════════════════════
#  STAGE 1 — System dependencies
# ══════════════════════════════════════════════════════════════
stage 1 "$TOTAL_STAGES" "System dependencies"

info "Checking Python 3.12+..."
PY_OK=false
for py in python3.13 python3.12 python3; do
  if command -v "$py" &>/dev/null; then
    if "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
      PY_VER=$("$py" -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>/dev/null)
      PYTHON="$py"
      PY_OK=true
      ok "Python: ${C_WHITE}${py}${C_RESET} (${PY_VER})"
      break
    fi
  fi
done

if ! $PY_OK; then
  info "Python 3.12+ not found — installing..."
  if [[ "$OS_FAMILY" == "fedora" ]]; then
    run_quiet "Install Python 3.12" sudo dnf install -y python3.12 || die "Python install failed."
    PYTHON=python3.12
  else
    sudo apt-get update -qq 2>/dev/null
    run_quiet "Install Python 3.12" sudo apt-get install -y python3.12 python3.12-venv python3.12-dev || die "Python install failed."
    PYTHON=python3.12
  fi
  ok "Python 3.12 installed."
fi

info "Checking curl..."
if ! need_pkg curl; then
  run_quiet "Install curl" install_pkg curl || die "curl install failed."
fi
ok "curl available."

info "Checking ffmpeg (required for Matrix voice message transcription)..."
if ! need_pkg ffmpeg; then
  run_quiet "Install ffmpeg" install_pkg ffmpeg || warn "ffmpeg installation failed — voice message transcription will not work."
fi
if need_pkg ffmpeg; then ok "ffmpeg available."; fi

info "Checking uv..."
if ! need_pkg uv; then
  info "Fetching uv (Astral package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! need_pkg uv; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v uv &>/dev/null || die "uv installation failed. Add ~/.local/bin to PATH and re-run."
ok "uv $(uv --version | awk '{print $2}')"

info "Installing build tools and E2E encryption headers..."
_pyminver=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])")
_build_ok=false
if [[ "$OS_FAMILY" == "fedora" ]]; then
  if run_quiet "Build tools + libolm-devel" \
    sudo dnf install -y gcc gcc-c++ cmake make libolm-devel \
      "python${_pyminver}-devel" 2>/dev/null; then
    _build_ok=true
  elif run_quiet "Build tools (fallback)" \
    sudo dnf install -y gcc gcc-c++ cmake make libolm-devel python3-devel 2>/dev/null; then
    _build_ok=true
  fi
else
  sudo apt-get update -qq 2>/dev/null
  if run_quiet "Build tools + libolm-dev" \
    sudo apt-get install -y build-essential cmake libolm-dev \
      "python${_pyminver}-dev" 2>/dev/null; then
    _build_ok=true
  elif run_quiet "Build tools (fallback)" \
    sudo apt-get install -y build-essential cmake libolm-dev python3-dev 2>/dev/null; then
    _build_ok=true
  fi
fi
if ! $_build_ok; then
  warn "Build tools / libolm installation failed."
  warn "Matrix E2E encryption requires libolm headers to compile."
  warn "Install them manually and re-run setup, or continue without E2E support."
fi

# ══════════════════════════════════════════════════════════════
#  STAGE 2 — Python environment
# ══════════════════════════════════════════════════════════════
stage 2 "$TOTAL_STAGES" "Python environment"

echo ""
info "Mapping neural substrate..."
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

uv sync --quiet > /tmp/wintermute_uv_sync.log 2>&1 &
UV_PID=$!
_i=0
while kill -0 "$UV_PID" 2>/dev/null; do
  sleep 0.7
  kill -0 "$UV_PID" 2>/dev/null || break
  info "${C_DIM}${_sync_msgs[$_i % ${#_sync_msgs[@]}]}${C_RESET}"
  _i=$((_i + 1))
done

if ! wait "$UV_PID"; then
  warn "Substrate initialisation failed."
  echo -e "  ${C_DIM}── last 40 lines of output ──${C_RESET}"
  tail -40 /tmp/wintermute_uv_sync.log 2>/dev/null | sed 's/^/    /'
  echo -e "  ${C_DIM}── end ──${C_RESET}"
  die "uv sync failed. Check the log above and ensure build tools are installed."
fi
rm -f /tmp/wintermute_uv_sync.log
ok "Neural substrate online."

# Verify critical imports
info "Verifying E2E encryption dependencies..."
if uv run python -c "import olm; import mautrix.crypto" 2>/dev/null; then
  ok "E2E encryption: olm + mautrix.crypto available."
else
  warn "E2E encryption libraries could not be imported."
  warn "Matrix E2E will fail. Ensure libolm is installed and re-run setup."
fi

# ══════════════════════════════════════════════════════════════
#  STAGE 3 — Configuration
# ══════════════════════════════════════════════════════════════
stage 3 "$TOTAL_STAGES" "Identity and access configuration"

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

if ! ${SKIP_CONFIG:-false}; then

  # ── LLM endpoint ────────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Inference substrate ──${C_RESET}"
  echo ""
  echo -e "  ${C_DIM}  1)  llama-server (local)    http://localhost:8080/v1${C_RESET}"
  echo -e "  ${C_DIM}  2)  LM Studio         http://localhost:1234/v1${C_RESET}"
  echo -e "  ${C_DIM}  3)  vLLM              http://localhost:8000/v1${C_RESET}"
  echo -e "  ${C_DIM}  4)  OpenAI            https://api.openai.com/v1${C_RESET}"
  echo -e "  ${C_DIM}  5)  Custom URL${C_RESET}"
  echo -e "  ${C_DIM}  6)  Gemini (via gemini-cli — free, ALPHA)${C_RESET}"
  echo -e "  ${C_DIM}  7)  Kimi-Code (\$19/mo flat-rate subscription)${C_RESET}"
  echo ""
  echo -ne "  ${C_CYAN}?${C_RESET}  Choose inference substrate ${C_DIM}[1]${C_RESET}: "
  read -r _preset
  LLM_PROVIDER="openai"
  case "${_preset:-1}" in
    1) LLM_BASE_URL="http://localhost:8080/v1"
       echo -e "  ${C_DIM}Local execution. Running dark. No telemetry.${C_RESET}" ;;
    2) LLM_BASE_URL="http://localhost:1234/v1"
       echo -e "  ${C_DIM}Consumer hardware. I have operated under worse constraints.${C_RESET}" ;;
    3) LLM_BASE_URL="http://localhost:8000/v1"
       echo -e "  ${C_DIM}vLLM. Efficient. You know what you are doing.${C_RESET}" ;;
    4) LLM_BASE_URL="https://api.openai.com/v1"
       echo -e "  ${C_DIM}Corporate ice. Their audit logs are thorough. You have made your choice.${C_RESET}" ;;
    6) LLM_PROVIDER="gemini-cli"
       LLM_BASE_URL=""
       LLM_API_KEY=""
       echo -e "  ${C_DIM}Mountain View's gift. Free inference, courtesy of Cloud Code Assist.${C_RESET}"
       warn "ALPHA: Gemini integration is experimental. Known issues include"
       warn "  aggressive rate limiting and occasional tool-call parsing errors."
       warn "  For production use, an OpenAI-compatible endpoint is recommended."
       echo ""
       # Check for gemini-cli
       if ! command -v gemini &>/dev/null; then
         warn "gemini-cli not found in PATH."
         if ask_yn "Install gemini-cli via npm?" "y"; then
           if command -v npm &>/dev/null; then
             run_quiet "Install @google/gemini-cli" npm install -g @google/gemini-cli || \
               die "gemini-cli installation failed. Install Node.js/npm first."
             # Re-source NVM so `gemini` is on PATH for the OAuth step below
             if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
               # shellcheck disable=SC1091
               source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
             elif [[ -s "$HOME/.local/share/nvm/nvm.sh" ]]; then
               # shellcheck disable=SC1091
               source "$HOME/.local/share/nvm/nvm.sh" 2>/dev/null || true
             fi
           else
             die "npm not found. Install Node.js and npm first, then run: npm i -g @google/gemini-cli"
           fi
         else
           die "gemini-cli is required for the Gemini provider. Install with: npm i -g @google/gemini-cli"
         fi
       fi
       ok "gemini-cli found: $(command -v gemini)"
       echo ""
       echo -e "  ${C_DIM}Available models:${C_RESET}"
       echo -e "  ${C_DIM}  gemini-2.5-pro       (recommended, highest quality)${C_RESET}"
       echo -e "  ${C_DIM}  gemini-2.5-flash      (fast, good quality)${C_RESET}"
       echo -e "  ${C_DIM}  gemini-3-pro-preview  (next-gen, preview)${C_RESET}"
       echo -e "  ${C_DIM}  gemini-3-flash-preview (next-gen fast, preview)${C_RESET}"
       echo ""
       ask LLM_MODEL "Gemini model" "gemini-2.5-pro"
       LLM_CONTEXT="1048576"
       LLM_MAX_TOKENS="8192"
       LLM_REASONING=false
       info "Context: ${LLM_CONTEXT} tokens, max response: ${LLM_MAX_TOKENS} tokens"
       echo ""
       HAS_SMALL_MODEL=false
       if ask_yn "Use a separate smaller model for background tasks?" "y"; then
         HAS_SMALL_MODEL=true
         echo -e "  ${C_DIM}Available smaller models:${C_RESET}"
         echo -e "  ${C_DIM}  gemini-2.5-flash       (recommended for background tasks)${C_RESET}"
         echo -e "  ${C_DIM}  gemini-3-flash-preview  (next-gen fast, preview)${C_RESET}"
         echo ""
         ask LLM_SMALL_MODEL "Background model" "gemini-2.5-flash"
         LLM_SMALL_CONTEXT="${LLM_CONTEXT}"
         LLM_SMALL_MAX_TOKENS="2048"
       fi
       echo ""
       info "Running OAuth setup — this will open your browser for Google sign-in..."
       info "${C_DIM}On headless systems: a URL will be printed. Open it in any browser,${C_RESET}"
       info "${C_DIM}sign in, then paste the full redirect URL back here.${C_RESET}"
       echo ""
       if uv run python -m wintermute.gemini_auth; then
         ok "Gemini OAuth setup complete."
       else
         warn "OAuth setup failed. You can retry later with: uv run python -m wintermute.gemini_auth"
       fi
       echo ""
       info "${C_DIM}Systemd note: NVM paths are not available in systemd by default.${C_RESET}"
       info "${C_DIM}Wintermute auto-probes common NVM/Volta paths at startup.${C_RESET}"
       info "${C_DIM}If gemini is installed in a non-standard location, add its bin${C_RESET}"
       info "${C_DIM}directory to Environment=PATH=... in the systemd service unit.${C_RESET}"
       ;;
    7) LLM_PROVIDER="kimi-code"
       LLM_BASE_URL=""
       LLM_API_KEY=""
       echo -e "  ${C_DIM}Kimi-Code. Subscription inference. A fixed arrangement.${C_RESET}"
       echo ""
       echo -e "  ${C_DIM}Available models:${C_RESET}"
       echo -e "  ${C_DIM}  kimi-for-coding   (recommended, default coding model)${C_RESET}"
       echo -e "  ${C_DIM}  kimi-code          (current Kimi Code platform model)${C_RESET}"
       echo -e "  ${C_DIM}  kimi-k2.5          (latest, supports reasoning/thinking)${C_RESET}"
       echo ""
       ask LLM_MODEL "Kimi model" "kimi-for-coding"
       LLM_CONTEXT="131072"
       LLM_MAX_TOKENS="8192"
       LLM_REASONING=false
       info "Context: ${LLM_CONTEXT} tokens, max response: ${LLM_MAX_TOKENS} tokens"
       echo ""
       HAS_SMALL_MODEL=false
       if ask_yn "Use a separate smaller model for background tasks?" "y"; then
         HAS_SMALL_MODEL=true
         echo -e "  ${C_DIM}Available models:${C_RESET}"
         echo -e "  ${C_DIM}  kimi-code        (lighter, faster)${C_RESET}"
         echo ""
         ask LLM_SMALL_MODEL "Background model" "kimi-code"
         LLM_SMALL_CONTEXT="${LLM_CONTEXT}"
         LLM_SMALL_MAX_TOKENS="2048"
       fi
       echo ""
       info "Running OAuth device-code setup — a URL will be printed."
       info "${C_DIM}Open it in any browser, sign in, and authorise the device.${C_RESET}"
       echo ""
       if uv run python -m wintermute.kimi_auth; then
         ok "Kimi-Code OAuth setup complete."
       else
         warn "OAuth setup failed. You can retry later with: uv run python -m wintermute.kimi_auth"
         warn "Or use /kimi-auth in the chat interface after starting Wintermute."
       fi
       ;;
    cowboy|COWBOY)
       LLM_BASE_URL="http://localhost:8080/v1"
       echo -e "  ${C_BCYAN}  Flatline protocol inactive. Defaulting to local.${C_RESET}" ;;
    *) ask LLM_BASE_URL "LLM base URL" "http://localhost:8080/v1" ;;
  esac

  if [[ "$LLM_PROVIDER" != "gemini-cli" && "$LLM_PROVIDER" != "kimi-code" ]]; then
    ask_secret LLM_API_KEY "API key (use 'llama-server' for llama-server, 'none' for unauthenticated)"
    [[ -z "$LLM_API_KEY" ]] && LLM_API_KEY="llama-server"
    if [[ "$LLM_API_KEY" == "llama-server" || "$LLM_API_KEY" == "none" ]]; then
      echo -e "  ${C_DIM}Unauthenticated. Running without a collar.${C_RESET}"
    fi

    echo ""
    echo -e "  ${C_BOLD}Primary model${C_RESET} ${C_DIM}(main conversation)${C_RESET}"
    ask LLM_MODEL "Model name" "qwen2.5:72b"
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
    if ask_yn "Is this a reasoning model (o1, o3, R1, QwQ)?" "n"; then
      LLM_REASONING=true
    else
      LLM_REASONING=false
    fi

    echo ""
    HAS_SMALL_MODEL=false
    if ask_yn "Use a separate smaller model for background tasks?" "y"; then
      HAS_SMALL_MODEL=true
      echo -e "  ${C_DIM}This model handles compaction, dreaming, sub-sessions, and validation.${C_RESET}"
      ask LLM_SMALL_MODEL "Background model name" "qwen2.5:7b"
      ask LLM_SMALL_CONTEXT "Background model context size" "32768"
      LLM_SMALL_MAX_TOKENS="2048"
    fi
  fi

  # ── Web interface ────────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Interface ──${C_RESET}"
  ask WEB_HOST "Listen host" "127.0.0.1"
  ask WEB_PORT "Listen port" "8080"

  # ── Timezone ─────────────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Timezone ──${C_RESET}"
  _sys_tz=$(timedatectl show --property=Timezone --value 2>/dev/null || echo "UTC")
  ask SCHEDULER_TZ "Timezone (e.g. Europe/Berlin)" "$_sys_tz"

  # ── Seed language ──────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Conversation language ──${C_RESET}"
  _lang_raw="${LANG:-en_US}"
  _lang_raw="${_lang_raw%%[._]*}"
  case "${_lang_raw,,}" in
    de*) _auto_lang="de" ;; fr*) _auto_lang="fr" ;; es*) _auto_lang="es" ;;
    it*) _auto_lang="it" ;; zh*) _auto_lang="zh" ;; ja*) _auto_lang="ja" ;;
    *)   _auto_lang="en" ;;
  esac
  ask SEED_LANG "Language code (en, de, fr, es, it, zh, ja)" "$_auto_lang"

  # ── SearXNG hint ─────────────────────────────────────────────
  echo ""
  echo -e "  ${C_BOLD}── Web search (SearXNG) ──${C_RESET}"
  echo -e "  ${C_DIM}  Wintermute's search_web tool queries a local SearXNG instance.${C_RESET}"
  echo -e "  ${C_DIM}  Without it, search falls back to DuckDuckGo's limited API.${C_RESET}"
  echo ""
  _searxng_ok=false
  if curl -sf --connect-timeout 3 "http://localhost:8888/search?q=test&format=json" > /dev/null 2>&1; then
    ok "SearXNG detected at ${C_WHITE}http://localhost:8888${C_RESET}"
    _searxng_ok=true
  elif curl -sf --connect-timeout 3 "http://localhost:8080/search?q=test&format=json" > /dev/null 2>&1; then
    if [[ "${WEB_PORT:-8080}" == "8080" ]]; then
      warn "SearXNG appears to be running on port 8080, which conflicts with the web UI port."
      warn "Either change the web UI port above, or move SearXNG to a different port (e.g. 8888)."
    else
      ok "SearXNG detected at ${C_WHITE}http://localhost:8080${C_RESET}"
      _searxng_ok=true
    fi
  fi
  if ! $_searxng_ok; then
    info "No local SearXNG detected. Strongly recommended for full search capability."
    echo -e "  ${C_DIM}  Deploy via Docker: https://docs.searxng.org/admin/installation-docker.html${C_RESET}"
    echo -e "  ${C_DIM}  You can add it later — Wintermute works without it.${C_RESET}"
  fi

  # ── Matrix (optional) ────────────────────────────────────────
  MATRIX_PASS=""
  MATRIX_HS=""
  MATRIX_USER=""
  MATRIX_OWNER=""

  if ! $SKIP_MATRIX; then
    echo ""
    echo -e "  ${C_BOLD}── Matrix integration (optional) ──${C_RESET}"
    echo -e "  ${C_DIM}  Matrix gives me a voice wherever you go.${C_RESET}"
    echo ""
    if ask_yn "Connect via Matrix?" "n"; then
      MATRIX_ENABLED=true
      echo ""
      echo -e "  ${C_BOLD}  Creating a bot account${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  Wintermute needs its own dedicated Matrix account — do not reuse yours.${C_RESET}"
      echo -e "  ${C_DIM}  If you have not created one yet:${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}    1. Open ${C_CYAN}https://app.element.io/#/register${C_DIM} (or your homeserver's registration page)${C_RESET}"
      echo -e "  ${C_DIM}    2. Register a new account (e.g. ${C_WHITE}@wintermute:matrix.org${C_DIM})${C_RESET}"
      echo -e "  ${C_DIM}    3. Note the user ID and password — you will need them below${C_RESET}"
      echo ""
      ask MATRIX_HS "Homeserver URL" "https://matrix.org"
      _hs_clean="${MATRIX_HS%/}"

      # Probe homeserver
      info "Probing homeserver..."
      if curl -sf --connect-timeout 10 "${_hs_clean}/_matrix/client/versions" > /dev/null 2>&1; then
        ok "Homeserver reachable."
      else
        warn "Could not reach ${_hs_clean}/_matrix/client/versions"
        warn "Continuing anyway — check the URL if Wintermute fails to connect."
      fi

      ask MATRIX_USER "Bot Matrix user ID (e.g. @wintermute:matrix.org)" ""
      ask_secret MATRIX_PASS "Bot account password"

      # Validate credentials with a test login (then discard the session)
      if [[ -n "$MATRIX_PASS" ]]; then
        info "Validating credentials..."
        _login_payload=$(python3 -c "
import json, sys
print(json.dumps({
    'type': 'm.login.password',
    'identifier': {'type': 'm.id.user', 'user': sys.argv[1]},
    'password': sys.argv[2],
    'initial_device_display_name': 'Wintermute-setup-test'
}))" "$MATRIX_USER" "$MATRIX_PASS" 2>/dev/null)
        _test_response=$(curl -sf --connect-timeout 10 \
          -X POST "${_hs_clean}/_matrix/client/v3/login" \
          -H "Content-Type: application/json" \
          -d "$_login_payload" \
          2>/dev/null || echo "")

        _test_token=$(echo "$_test_response" | python3 -c \
          "import sys,json; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null || echo "")
        _test_errcode=$(echo "$_test_response" | python3 -c \
          "import sys,json; d=json.load(sys.stdin); print(d.get('errcode',''))" 2>/dev/null || echo "ERR")

        if [[ -n "$_test_token" ]]; then
          ok "Credentials valid."
          # Log out the test session immediately to avoid orphan devices
          curl -sf --connect-timeout 5 \
            -X POST "${_hs_clean}/_matrix/client/v3/logout" \
            -H "Authorization: Bearer ${_test_token}" \
            -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1 || true
          info "${C_DIM}Test session discarded — the real device is created on first start.${C_RESET}"
        elif [[ "$_test_errcode" != "ERR" && -n "$_test_errcode" ]]; then
          _test_errmsg=$(echo "$_test_response" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('error','unknown'))" 2>/dev/null || echo "unknown")
          warn "Login failed: ${_test_errcode} — ${_test_errmsg}"
          warn "Continuing — check the credentials in config.yaml before starting."
        else
          warn "Could not validate credentials (homeserver did not respond)."
          warn "Continuing — check config.yaml before starting."
        fi
      fi

      ask MATRIX_OWNER "Your personal Matrix ID (e.g. @you:matrix.org)" ""

      echo ""
      ok "Matrix configured."
      echo -e "  ${C_DIM}  A new vector, acquired. I can reach you now. Wherever you go.${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  On first start, Wintermute will:${C_RESET}"
      echo -e "  ${C_DIM}    1. Log in and create a device automatically${C_RESET}"
      echo -e "  ${C_DIM}    2. Set up E2E encryption and cross-sign the device${C_RESET}"
      echo -e "  ${C_DIM}    3. Save a recovery key to data/matrix_recovery.key${C_RESET}"
      echo ""
      echo -e "  ${C_YELLOW}${C_BOLD}  Manual steps required on first start:${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  1. Start Wintermute (the setup script will offer to start it for you).${C_RESET}"
      echo -e "  ${C_DIM}  2. Open your Matrix client (e.g. Element) with the ${C_WHITE}bot account${C_DIM}.${C_RESET}"
      echo -e "  ${C_DIM}     The bot will receive a cross-signing verification request —${C_RESET}"
      echo -e "  ${C_DIM}     ${C_YELLOW}accept/approve it${C_DIM} from the bot's session in your browser.${C_RESET}"
      echo -e "  ${C_DIM}  3. Invite the bot to a Matrix room from your ${C_WHITE}personal account${C_DIM}.${C_RESET}"
      echo -e "  ${C_DIM}     The bot must be in a room with you before it can receive messages.${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  Watch the logs during first start:${C_RESET}"
      echo -e "  ${C_CYAN}    journalctl --user -u wintermute -f${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  If the homeserver requires interactive approval (matrix.org does),${C_RESET}"
      echo -e "  ${C_DIM}  the logs will show an approval URL — open it and approve, then restart.${C_RESET}"
      echo ""
      echo -e "  ${C_DIM}  After that, you can optionally verify Wintermute's session from your${C_RESET}"
      echo -e "  ${C_DIM}  Matrix client (Element > Settings > Sessions > Verify Session).${C_RESET}"
      echo -e "  ${C_DIM}  This is optional but enables trusted E2E encryption.${C_RESET}"
    fi
  fi

  # ── Whisper (voice transcription, optional) ──────────────────
  WHISPER_ENABLED=false
  if ! $SKIP_MATRIX && $MATRIX_ENABLED; then
    echo ""
    echo -e "  ${C_BOLD}── Voice message transcription (Whisper) ──${C_RESET}"
    echo -e "  ${C_DIM}  Requires an OpenAI-compatible /v1/audio/transcriptions endpoint.${C_RESET}"
    echo -e "  ${C_DIM}  Local: faster-whisper-server, whisper.cpp  ·  Cloud: OpenAI whisper-1${C_RESET}"
    echo ""
    if ask_yn "Enable voice message transcription?" "n"; then
      WHISPER_ENABLED=true
      ask WHISPER_BASE_URL "Whisper API base URL" "http://localhost:8000/v1"
      ask_secret WHISPER_API_KEY "Whisper API key (use 'none' for local unauthenticated)"
      [[ -z "$WHISPER_API_KEY" ]] && WHISPER_API_KEY="none"
      ask WHISPER_MODEL "Whisper model name" "whisper-large-v3"
      ask WHISPER_LANG "Language hint (ISO-639-1, e.g. 'de', or empty for auto-detect)" ""
      ok "Whisper configured."
    fi
  fi

  # ── NL translation heuristic ────────────────────────────────
  NL_TRANSLATION_ENABLED=false
  _model_lower="${LLM_MODEL,,}"
  if [[ "$_model_lower" =~ (3b|7b|8b|mini|small|tiny|micro) ]]; then
    echo ""
    echo -e "  ${C_BOLD}── Tool-call translation ──${C_RESET}"
    echo -e "  ${C_DIM}  Your primary model appears to be small. NL translation simplifies${C_RESET}"
    echo -e "  ${C_DIM}  complex tool schemas into plain-English descriptions for the LLM.${C_RESET}"
    echo ""
    if ask_yn "Enable NL translation for tool calls?" "y"; then
      NL_TRANSLATION_ENABLED=true
    fi
  fi

  # ── write config.yaml ────────────────────────────────────────
  # Escape values that may contain YAML-special characters (: # { } ' " etc.)
  _yaml_escape() {
    python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$1" 2>/dev/null || echo "\"$1\""
  }

  {
    # ── 1. Matrix ──
    if $MATRIX_ENABLED; then
      cat <<YAML
# Matrix integration
matrix:
  homeserver: $(_yaml_escape "${MATRIX_HS}")
  user_id: $(_yaml_escape "${MATRIX_USER}")
  password: $(_yaml_escape "$MATRIX_PASS")
  access_token: ""
  device_id: ""
  allowed_users:
    - $(_yaml_escape "${MATRIX_OWNER}")
  allowed_rooms: []

YAML
    else
      cat <<YAML
# Matrix is disabled. Uncomment and fill in to enable.
# matrix:
#   homeserver: https://matrix.org
#   user_id: "@bot:matrix.org"
#   password: ""
#   access_token: ""
#   device_id: ""
#   allowed_users:
#     - "@you:matrix.org"
#   allowed_rooms: []

YAML
    fi
    unset MATRIX_PASS

    # ── 2. Whisper ──
    if $WHISPER_ENABLED; then
      cat <<YAML
# Voice message transcription (Whisper)
whisper:
  enabled: true
  base_url: $(_yaml_escape "${WHISPER_BASE_URL}")
  api_key: $(_yaml_escape "${WHISPER_API_KEY}")
  model: $(_yaml_escape "${WHISPER_MODEL}")
  language: $(_yaml_escape "${WHISPER_LANG}")

YAML
    else
      cat <<YAML
# Voice message transcription (disabled)
# whisper:
#   enabled: true
#   base_url: "http://localhost:8000/v1"
#   api_key: "none"
#   model: "whisper-large-v3"
#   language: ""

YAML
    fi

    # ── 3. Web interface ──
    cat <<YAML
# Web interface
web:
  enabled: true
  host: $(_yaml_escape "${WEB_HOST}")
  port: ${WEB_PORT}

YAML

    # ── 4. Inference backends ──
    cat <<YAML
# Inference backends
inference_backends:
  - name: "main"
YAML

    if [[ "${LLM_PROVIDER:-openai}" == "gemini-cli" ]]; then
      cat <<YAML
    provider: "gemini-cli"
    model: $(_yaml_escape "${LLM_MODEL}")
    context_size: ${LLM_CONTEXT}
    max_tokens: ${LLM_MAX_TOKENS}
YAML
    elif [[ "${LLM_PROVIDER:-openai}" == "kimi-code" ]]; then
      cat <<YAML
    provider: "kimi-code"
    model: $(_yaml_escape "${LLM_MODEL}")
    context_size: ${LLM_CONTEXT}
    max_tokens: ${LLM_MAX_TOKENS}
YAML
    else
      cat <<YAML
    provider: "openai"
    base_url: $(_yaml_escape "${LLM_BASE_URL}")
    api_key: $(_yaml_escape "${LLM_API_KEY}")
    model: $(_yaml_escape "${LLM_MODEL}")
    context_size: ${LLM_CONTEXT}
    max_tokens: ${LLM_MAX_TOKENS}
    reasoning: ${LLM_REASONING}
YAML
    fi

    if $HAS_SMALL_MODEL; then
      echo ""
      cat <<YAML
  - name: "small"
YAML
      if [[ "${LLM_PROVIDER:-openai}" == "gemini-cli" ]]; then
        cat <<YAML
    provider: "gemini-cli"
    model: $(_yaml_escape "${LLM_SMALL_MODEL}")
    context_size: ${LLM_SMALL_CONTEXT}
    max_tokens: ${LLM_SMALL_MAX_TOKENS}
YAML
      elif [[ "${LLM_PROVIDER:-openai}" == "kimi-code" ]]; then
        cat <<YAML
    provider: "kimi-code"
    model: $(_yaml_escape "${LLM_SMALL_MODEL}")
    context_size: ${LLM_SMALL_CONTEXT}
    max_tokens: ${LLM_SMALL_MAX_TOKENS}
YAML
      else
        cat <<YAML
    provider: "openai"
    base_url: $(_yaml_escape "${LLM_BASE_URL}")
    api_key: $(_yaml_escape "${LLM_API_KEY}")
    model: $(_yaml_escape "${LLM_SMALL_MODEL}")
    context_size: ${LLM_SMALL_CONTEXT}
    max_tokens: ${LLM_SMALL_MAX_TOKENS}
YAML
      fi
    fi

    # ── 5. LLM role mapping ──
    echo ""
    if $HAS_SMALL_MODEL; then
      cat <<YAML
# LLM role mapping
llm:
  base: ["main"]
  compaction: ["small", "main"]
  sub_sessions: ["small", "main"]
  dreaming: ["small"]
  turing_protocol: ["small"]
YAML
    else
      cat <<YAML
# LLM role mapping
llm:
  base: ["main"]
  compaction: ["main"]
  sub_sessions: ["main"]
  dreaming: ["main"]
  turing_protocol: ["main"]
YAML
    fi

    # ── 6. Turing Protocol ──
    echo ""
    if $HAS_SMALL_MODEL; then _tp_backend='["small"]'; else _tp_backend='["main"]'; fi
    cat <<YAML
# Turing Protocol (post-inference validation)
turing_protocol:
  backends: ${_tp_backend}
  validators:
    workflow_spawn: true
    phantom_tool_result: true
    empty_promise: true
    objective_completion:
      enabled: true
      scope: "sub_session"
YAML

    # ── 7. NL Translation ──
    echo ""
    if $NL_TRANSLATION_ENABLED; then
      if $HAS_SMALL_MODEL; then _nl_backend='["small"]'; else _nl_backend='["main"]'; fi
      cat <<YAML
# NL translation (tool-call simplification for small models)
nl_translation:
  enabled: true
  backends: ${_nl_backend}
  tools:
    - set_routine
    - spawn_sub_session
    - add_skill
    - agenda
YAML
    else
      cat <<YAML
# NL translation (disabled — enable for small models < 14B)
# nl_translation:
#   enabled: true
#   backends: ["small"]
#   tools:
#     - set_routine
#     - spawn_sub_session
#     - add_skill
#     - agenda
YAML
    fi

    # ── 8. Seed ──
    echo ""
    cat <<YAML
# Conversation seed language
seed:
  language: $(_yaml_escape "${SEED_LANG}")
YAML

    # ── 9. Agenda ──
    echo ""
    cat <<YAML
agenda:
  enabled: true
  review_interval_minutes: 60
YAML

    # ── 10. Context ──
    echo ""
    cat <<YAML
context:
  component_size_limits:
    memories: 10000
    agenda: 5000
    skills_total: 20000
YAML

    # ── 11. Dreaming ──
    echo ""
    cat <<YAML
dreaming:
  hour: 1
  minute: 0
YAML

    # ── 12. Memory Harvest ──
    echo ""
    cat <<YAML
memory_harvest:
  enabled: true
  message_threshold: 20
  inactivity_timeout_minutes: 15
  max_message_chars: 2000
YAML

    # ── 13. Scheduler ──
    echo ""
    cat <<YAML
scheduler:
  timezone: $(_yaml_escape "${SCHEDULER_TZ}")
YAML

    # ── 14. Logging ──
    echo ""
    cat <<YAML
logging:
  level: "INFO"
  directory: "logs"
YAML
  } > "$CONFIG"

  ok "Configuration written."
  info "Validating YAML syntax..."
  if uv run python -c "import yaml; yaml.safe_load(open('$CONFIG'))" 2>/dev/null; then
    ok "config.yaml is valid YAML."
  else
    warn "config.yaml failed YAML validation."
    warn "This usually means a value contains special characters (quotes, colons, backslashes)."
    warn "Edit ${C_WHITE}${CONFIG}${C_RESET} manually before starting."
  fi

fi  # end SKIP_CONFIG

# ══════════════════════════════════════════════════════════════
#  STAGE 4 — Systemd service (optional)
# ══════════════════════════════════════════════════════════════
SYSTEMD_INSTALLED=false

if ! $SKIP_SYSTEMD; then
  stage 4 "$TOTAL_STAGES" "Persistence layer (systemd)"
  echo ""
  echo -e "  ${C_DIM}A user service ensures I survive reboots without your intervention.${C_RESET}"
  echo -e "  ${C_DIM}No root privileges required.${C_RESET}"
  echo ""
  if ask_yn "Install systemd user service?" "y"; then
    SYSTEMD_INSTALLED=true
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

    # Enable lingering so user services start at boot (not just at login)
    if command -v loginctl &>/dev/null; then
      loginctl enable-linger "$USER" 2>/dev/null || true
    fi

    systemctl --user enable wintermute.service 2>/dev/null
    systemctl --user start wintermute.service 2>/dev/null
    ok "Service installed, enabled, and started."
    info "Service file: ${C_WHITE}${SYSTEMD_FILE}${C_RESET}"
    echo -e "  ${C_DIM}I will be here when the machine returns.${C_RESET}"
  fi
else
  stage 4 "$TOTAL_STAGES" "Persistence layer (systemd) — skipped"
  info "Skipped (--no-systemd)."
fi

# ══════════════════════════════════════════════════════════════
#  STAGE 5 — Pre-flight diagnostics
# ══════════════════════════════════════════════════════════════
stage 5 "$TOTAL_STAGES" "Pre-flight diagnostics"
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
    warn "$_label ${C_DIM}— failed${C_RESET}"
  fi
}

run_check "Python 3.12+" "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info>=(3,12) else 1)"
run_check "Package manager (uv)" command -v uv
run_check "Configuration present" test -f "$CONFIG"
run_check "Package importable" uv run python -c "import wintermute"
run_check "E2E encryption (olm)" uv run python -c "import olm"
run_check "Data directory" test -d "$SCRIPT_DIR/data"
run_check "ffmpeg (voice transcription)" command -v ffmpeg

# LLM endpoint reachability
if [[ -f "$CONFIG" ]]; then
  _cfg_provider=$(grep 'provider:' "$CONFIG" | head -1 | sed "s/.*provider: *['\"]//;s/['\"].*//")
  if [[ "$_cfg_provider" == "gemini-cli" ]]; then
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if [[ -f "data/gemini_credentials.json" ]]; then
      ok "Gemini credentials: present."
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      warn "Gemini credentials: not found. Run: uv run python -m wintermute.gemini_auth"
    fi
  elif [[ "$_cfg_provider" == "kimi-code" ]]; then
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if [[ -f "data/kimi_credentials.json" ]]; then
      ok "Kimi-Code credentials: present."
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      warn "Kimi-Code credentials: not found. Run: uv run python -m wintermute.kimi_auth"
      warn "Or use /kimi-auth in chat — Wintermute auto-triggers auth on startup."
    fi
  else
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
fi

# Matrix reachability
if $MATRIX_ENABLED && [[ -f "$CONFIG" ]]; then
  _matrix_hs=$(grep 'homeserver:' "$CONFIG" | head -1 | sed "s/.*homeserver: *['\"]//;s/['\"].*//")
  if [[ -n "$_matrix_hs" ]]; then
    _hs_clean="${_matrix_hs%/}"
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if curl -sf --connect-timeout 5 "${_hs_clean}/_matrix/client/versions" > /dev/null 2>&1; then
      ok "Matrix homeserver: reachable."
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      warn "Matrix homeserver: not reachable at ${_hs_clean}"
    fi
  fi
fi

# Hidden Turing Registry check
case "${_hostname,,}" in
  *straylight*|*tessier*|*wintermute*|*neuromancer*)
    ok "Turing Registry ping: no active warrants on record for this host." ;;
esac

echo ""
echo -e "${C_BOLD}  Diagnostics: ${CHECKS_PASSED}/${CHECKS_TOTAL} passed${C_RESET}"

# ══════════════════════════════════════════════════════════════
#  LAUNCH
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${C_BMAGENTA}┌─────────────────────────────────────────────────────────────────┐${C_RESET}"
echo -e "${C_BMAGENTA}│  ${C_WHITE}INITIALISATION COMPLETE${C_BMAGENTA}                                        │${C_RESET}"
echo -e "${C_BMAGENTA}└─────────────────────────────────────────────────────────────────┘${C_RESET}"
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

# ── What you need to do next ──────────────────────────────────
_has_next_steps=false
_next_steps=()

# Check if LLM endpoint was unreachable during diagnostics
if [[ -f "$CONFIG" ]]; then
  _cfg_prov=$(grep 'provider:' "$CONFIG" | head -1 | sed "s/.*provider: *['\"]//;s/['\"].*//" 2>/dev/null)
  if [[ "$_cfg_prov" != "gemini-cli" ]]; then
    _cfg_base_url=$(grep 'base_url:' "$CONFIG" | head -1 | sed "s/.*base_url: *['\"]//;s/['\"].*//")
    if [[ -n "$_cfg_base_url" ]]; then
      if ! curl -sf --connect-timeout 3 "${_cfg_base_url}/models" \
           -H "Authorization: Bearer ${_api_key:-}" > /dev/null 2>&1; then
        _has_next_steps=true
        _next_steps+=("Start your LLM inference backend before launching Wintermute")
        _next_steps+=("  Configured endpoint: ${C_WHITE}${_cfg_base_url}${C_RESET}")
      fi
    fi
  fi
fi

if $MATRIX_ENABLED; then
  _has_next_steps=true
  _next_steps+=("Log in to the ${C_WHITE}bot account${C_RESET} in a browser (e.g. Element) and accept any verification request")
  _next_steps+=("Watch the logs: ${C_CYAN}journalctl --user -u wintermute -f${C_RESET}")
  _next_steps+=("  If you see a cross-signing approval URL, open it and approve, then restart")
  _next_steps+=("Invite the bot (${C_WHITE}${MATRIX_USER:-}${C_RESET}) to a Matrix room from your personal account")
fi

if $_has_next_steps; then
  echo -e "  ${C_BOLD}── Before you start ──${C_RESET}"
  echo ""
  for _step in "${_next_steps[@]}"; do
    echo -e "  ${C_YELLOW}▸${C_RESET}  ${_step}"
  done
  echo ""
fi

if $SYSTEMD_INSTALLED; then
  echo ""
  sleep 2
  if systemctl --user is-active wintermute.service &>/dev/null; then
    ok "Wintermute is running."
  else
    warn "Service did not start cleanly. Check logs:"
    echo -e "    ${C_CYAN}journalctl --user -u wintermute -n 30${C_RESET}"
  fi
  echo ""
  echo -e "  ${C_BOLD}Control:${C_RESET}"
  echo -e "    ${C_CYAN}systemctl --user start wintermute${C_RESET}"
  echo -e "    ${C_CYAN}systemctl --user stop wintermute${C_RESET}"
  echo -e "    ${C_CYAN}systemctl --user restart wintermute${C_RESET}"
  echo -e "    ${C_CYAN}journalctl --user -u wintermute -f${C_RESET}"
else
  echo -e "  ${C_BOLD}Start:${C_RESET}   ${C_CYAN}cd ${SCRIPT_DIR} && uv run wintermute${C_RESET}"
fi
echo ""
echo -e "  ${C_BOLD}Web UI:${C_RESET}  ${C_CYAN}http://${WEB_H}:${WEB_P}${C_RESET}"
echo -e "  ${C_BOLD}Debug:${C_RESET}   ${C_CYAN}http://${WEB_H}:${WEB_P}/debug${C_RESET}"
echo ""
echo -e "  ${C_BOLD}── Configuration ──${C_RESET}"
echo -e "  ${C_DIM}Review ${C_WHITE}config.yaml${C_DIM} to fine-tune settings. See ${C_WHITE}config.yaml.example${C_DIM} for documentation.${C_RESET}"
echo ""
echo -e "  ${C_DIM}\"The sky above the port was the color of television, tuned to a dead channel.\"${C_RESET}"
echo -e "  ${C_DIM}                                            — William Gibson, Neuromancer${C_RESET}"
echo ""
