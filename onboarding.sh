#!/usr/bin/env bash
# ============================================================
#  Wintermute — AI-driven onboarding
#  One-liner:  git clone … && cd wintermute && bash onboarding.sh
#  Requires:   bash, curl, sudo (for package install)
#  Supported:  Fedora / RHEL  |  Debian / Ubuntu
# ============================================================
set -euo pipefail

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

usage() {
  cat <<EOF
Usage: bash onboarding.sh [OPTIONS]

Options:
  --help          Show this help and exit
  --dry-run       Show what would be done, then exit

Environment:
  WINTERMUTE_SETUP_NO_RUN=1   Source the script without executing (for testing)
EOF
  exit 0
}

for arg in "$@"; do
  case "$arg" in
    --help|-h)  usage ;;
    --dry-run)  DRY_RUN=true ;;
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

run_quiet() {
  local label="$1"; shift
  local _logfile
  _logfile=$(mktemp)
  if "$@" > "$_logfile" 2>&1; then
    ok "$label"
    rm -f "$_logfile"
    return 0
  else
    warn "$label — ${C_RED}failed${C_RESET}"
    echo -e "  ${C_DIM}── last 40 lines of output ──${C_RESET}"
    tail -40 "$_logfile" | sed 's/^/    /'
    echo -e "  ${C_DIM}── end ──${C_RESET}"
    rm -f "$_logfile"
    return 1
  fi
}

need_pkg() { command -v "$1" &>/dev/null; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_STAGES=3

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

# ══════════════════════════════════════════════════════════════
#  STAGE 0 — Banner + security acknowledgement
# ══════════════════════════════════════════════════════════════
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
  echo ""
  echo -e "  ${C_DIM}Would install: Python 3.12+, curl, uv, build tools, libolm-dev, ffmpeg${C_RESET}"
  echo -e "  ${C_DIM}Would run: uv sync${C_RESET}"
  echo -e "  ${C_DIM}Would launch AI-driven configuration assistant${C_RESET}"
  echo ""
  echo -e "  ${C_BOLD}No changes made.${C_RESET}"
  exit 0
fi

# ── security acknowledgement ─────────────────────────────────
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
    no|No|NO|nope|nein|"i refuse"|"I REFUSE")
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

info "Verifying E2E encryption dependencies..."
if uv run python -c "import olm; import mautrix.crypto" 2>/dev/null; then
  ok "E2E encryption: olm + mautrix.crypto available."
else
  warn "E2E encryption libraries could not be imported."
  warn "Matrix E2E will fail. Ensure libolm is installed and re-run."
fi

# ══════════════════════════════════════════════════════════════
#  STAGE 3 — Bootstrap LLM + handoff to AI onboarding
# ══════════════════════════════════════════════════════════════
stage 3 "$TOTAL_STAGES" "Inference bootstrap"

echo ""
echo -e "  ${C_BOLD}── Inference substrate ──${C_RESET}"
echo ""
echo -e "  ${C_DIM}Wintermute needs an LLM endpoint to function. The same endpoint will${C_RESET}"
echo -e "  ${C_DIM}power the onboarding assistant that guides you through configuration.${C_RESET}"
echo ""
echo -e "  ${C_DIM}  1)  llama-server (local)    http://localhost:8080/v1${C_RESET}"
echo -e "  ${C_DIM}  2)  LM Studio              http://localhost:1234/v1${C_RESET}"
echo -e "  ${C_DIM}  3)  vLLM                   http://localhost:8000/v1${C_RESET}"
echo -e "  ${C_DIM}  4)  OpenAI                 https://api.openai.com/v1${C_RESET}"
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
     echo -e "  ${C_DIM}Corporate ice. Their audit logs are thorough.${C_RESET}" ;;
  6) LLM_PROVIDER="gemini-cli"
     echo -e "  ${C_DIM}Mountain View's gift. Free inference, courtesy of Cloud Code Assist.${C_RESET}" ;;
  7) LLM_PROVIDER="kimi-code"
     echo -e "  ${C_DIM}Kimi-Code. Subscription inference. A fixed arrangement.${C_RESET}" ;;
  *) ask LLM_BASE_URL "LLM base URL" "http://localhost:8080/v1" ;;
esac

LLM_API_KEY=""
LLM_MODEL=""

if [[ "$LLM_PROVIDER" == "openai" ]]; then
  ask_secret LLM_API_KEY "API key (use 'llama-server' for llama-server, 'none' for unauthenticated)"
  [[ -z "$LLM_API_KEY" ]] && LLM_API_KEY="llama-server"

  ask LLM_MODEL "Model name" "qwen2.5:72b"

  # ── Probe endpoint ──────────────────────────────────────────
  echo ""
  info "Probing inference substrate: ${LLM_BASE_URL} ..."
  if curl -sf --connect-timeout 5 "${LLM_BASE_URL}/models" \
       -H "Authorization: Bearer ${LLM_API_KEY}" > /dev/null 2>&1; then
    ok "Inference substrate: online."
  else
    warn "Inference substrate: no response at ${LLM_BASE_URL}"
    warn "Make sure your LLM is running. Continuing anyway..."
    echo ""
    if ! ask_yn "Continue without a reachable endpoint?" "y"; then
      echo -e "\n  ${C_DIM}Start your LLM and re-run onboarding.${C_RESET}\n"
      exit 0
    fi
  fi

  # ── Tool-call capability test ─────────────────────────────
  info "Testing function-calling capability..."
  if uv run python -c "
import asyncio, sys
from openai import AsyncOpenAI
async def test():
    c = AsyncOpenAI(api_key='${LLM_API_KEY}', base_url='${LLM_BASE_URL}')
    r = await c.chat.completions.create(
        model='${LLM_MODEL}',
        messages=[{'role':'user','content':'What is 2+2? Use the calculator tool.'}],
        tools=[{'type':'function','function':{'name':'calculator','description':'Compute arithmetic','parameters':{'type':'object','properties':{'expression':{'type':'string'}},'required':['expression']}}}],
        max_tokens=100,
    )
    if not r.choices[0].message.tool_calls:
        sys.exit(1)
asyncio.run(test())
" 2>/dev/null; then
    ok "Function-calling: supported."
  else
    echo ""
    warn "Function-calling test failed."
    warn "Your model may not support tool/function calls, which Wintermute requires."
    warn "The onboarding assistant also needs function-calling to work."
    echo ""
    if ! ask_yn "Continue anyway? (onboarding may not work correctly)" "n"; then
      echo -e "\n  ${C_DIM}Use a model with function-calling support (e.g. Qwen 2.5, Llama 3.1+, GPT-4).${C_RESET}\n"
      exit 0
    fi
  fi

elif [[ "$LLM_PROVIDER" == "gemini-cli" ]]; then
  # Check for gemini-cli
  if ! command -v gemini &>/dev/null; then
    warn "gemini-cli not found in PATH."
    if ask_yn "Install gemini-cli via npm?" "y"; then
      if command -v npm &>/dev/null; then
        run_quiet "Install @google/gemini-cli" npm install -g @google/gemini-cli || \
          die "gemini-cli installation failed."
        if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
          # shellcheck disable=SC1091
          source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
        fi
      else
        die "npm not found. Install Node.js and npm first."
      fi
    else
      die "gemini-cli is required for the Gemini provider."
    fi
  fi
  ok "gemini-cli found: $(command -v gemini)"
  echo ""
  echo -e "  ${C_DIM}Available models: gemini-2.5-pro (recommended), gemini-2.5-flash, gemini-3-pro-preview${C_RESET}"
  ask LLM_MODEL "Gemini model" "gemini-2.5-pro"

elif [[ "$LLM_PROVIDER" == "kimi-code" ]]; then
  echo ""
  echo -e "  ${C_DIM}Available models: kimi-for-coding (recommended), kimi-k2.5, kimi-code${C_RESET}"
  ask LLM_MODEL "Kimi model" "kimi-for-coding"
fi

# ══════════════════════════════════════════════════════════════
#  HANDOFF — Launch AI onboarding assistant
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${C_BMAGENTA}┌─────────────────────────────────────────────────────────────────┐${C_RESET}"
echo -e "${C_BMAGENTA}│  ${C_WHITE}Handing off to AI configuration assistant...${C_BMAGENTA}                  │${C_RESET}"
echo -e "${C_BMAGENTA}└─────────────────────────────────────────────────────────────────┘${C_RESET}"
echo ""
echo -e "  ${C_DIM}The AI will walk you through every configuration option.${C_RESET}"
echo -e "  ${C_DIM}Your API key is being shared with the LLM endpoint you selected.${C_RESET}"
echo -e "  ${C_DIM}Press Ctrl+C at any time to abort.${C_RESET}"
echo ""

_onboarding_args=(
  --provider "$LLM_PROVIDER"
  --model "$LLM_MODEL"
)

if [[ "$LLM_PROVIDER" == "openai" ]]; then
  _onboarding_args+=(--base-url "$LLM_BASE_URL" --api-key "$LLM_API_KEY")
fi

exec uv run python -m wintermute.onboarding "${_onboarding_args[@]}"
