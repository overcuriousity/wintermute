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
  echo -e "  ${C_DIM}Would install: Python 3.12+, curl, uv, build tools, libolm-dev${C_RESET}"
  echo -e "  ${C_DIM}Would run: uv sync${C_RESET}"
  echo -e "  ${C_DIM}Would write: config.yaml${C_RESET}"
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
if [[ "$OS_FAMILY" == "fedora" ]]; then
  run_quiet "Build tools + libolm-devel" \
    sudo dnf install -y gcc gcc-c++ cmake make libolm-devel \
      "python${_pyminver}-devel" 2>/dev/null || \
    run_quiet "Build tools (fallback)" \
      sudo dnf install -y gcc gcc-c++ cmake make libolm-devel python3-devel 2>/dev/null || true
else
  sudo apt-get update -qq 2>/dev/null
  run_quiet "Build tools + libolm-dev" \
    sudo apt-get install -y build-essential cmake libolm-dev \
      "python${_pyminver}-dev" 2>/dev/null || \
    run_quiet "Build tools (fallback)" \
      sudo apt-get install -y build-essential cmake libolm-dev python3-dev 2>/dev/null || true
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
    # Don't detect on 8080 if that's what they chose for Wintermute
    if [[ "${WEB_PORT:-8080}" != "8080" ]]; then
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
      echo -e "  ${C_DIM}  You need a dedicated bot account. Do not reuse your personal account.${C_RESET}"
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
      echo -e "  ${C_DIM}  Some homeservers (e.g. matrix.org) require a one-time browser approval${C_RESET}"
      echo -e "  ${C_DIM}  for cross-signing — Wintermute logs the exact URL on first start.${C_RESET}"
      echo -e "  ${C_DIM}  After that, everything is fully automatic.${C_RESET}"
    fi
  fi

  # ── write config.yaml ────────────────────────────────────────
  {
    if $MATRIX_ENABLED; then
      cat <<YAML
# Matrix integration
matrix:
  homeserver: "${MATRIX_HS}"
  user_id: "${MATRIX_USER}"
  password: $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$MATRIX_PASS" 2>/dev/null || echo "\"${MATRIX_PASS}\"")
  access_token: ""
  device_id: ""
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
#   password: ""
#   access_token: ""
#   device_id: ""
#   allowed_users:
#     - "@you:matrix.org"
#   allowed_rooms: []

YAML
    fi
    unset MATRIX_PASS

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
      echo "  compaction:"
      echo "    model: \"${LLM_COMPACTION_MODEL}\""
    else
      echo "  # compaction:"
      echo "  #   model: \"\"  # optional: smaller/faster model for summarisation"
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
    ok "Service installed and enabled."
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

if $SYSTEMD_INSTALLED; then
  echo ""
  if ask_yn "Start Wintermute now?" "y"; then
    systemctl --user start wintermute.service
    sleep 2
    if systemctl --user is-active wintermute.service &>/dev/null; then
      ok "Wintermute is running."
    else
      warn "Service did not start cleanly. Check logs:"
      echo -e "    ${C_CYAN}journalctl --user -u wintermute -n 30${C_RESET}"
    fi
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
echo -e "  ${C_DIM}\"The sky above the port was the color of television, tuned to a dead channel.\"${C_RESET}"
echo -e "  ${C_DIM}                                            — William Gibson, Neuromancer${C_RESET}"
echo ""
