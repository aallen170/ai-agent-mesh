#!/usr/bin/env bash
# =============================================================================
# AIMESH bootstrap — Linux & macOS
# Sets up the Python environment using uv. Safe to re-run at any time.
#
# Usage:
#   bash scripts/bootstrap.sh
# =============================================================================
set -euo pipefail

# -- Colours ------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}▶  $1${NC}"; }
ok()   { echo -e "${GREEN}✓  $1${NC}"; }
warn() { echo -e "${YELLOW}!  $1${NC}"; }
die()  { echo -e "${RED}✗  $1${NC}" >&2; exit 1; }

# -- Locate repo root (works when called from any directory) ------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  AIMESH — First-time setup${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# -- Step 1: install uv if missing --------------------------------------------
step "Checking for uv (Python package manager)"

if command -v uv &>/dev/null; then
    ok "uv already installed ($(uv --version))"
else
    warn "uv not found — installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # The installer adds uv to PATH in shell config, but we need it now too.
    UV_PATHS=("$HOME/.local/bin" "$HOME/.cargo/bin")
    for p in "${UV_PATHS[@]}"; do
        if [[ -x "$p/uv" ]]; then
            export PATH="$p:$PATH"
            break
        fi
    done

    command -v uv &>/dev/null || die "uv installation failed. Please install manually: https://docs.astral.sh/uv/"
    ok "uv installed ($(uv --version))"

    warn "Restart your shell (or run: source ~/.bashrc / source ~/.zshrc) to make uv available globally."
fi

# -- Step 2: Python 3.11+ -----------------------------------------------------
step "Checking Python 3.11+"

if uv python find 3.11 &>/dev/null; then
    ok "Python 3.11 found ($(uv python find 3.11))"
else
    warn "Python 3.11 not found — installing via uv..."
    uv python install 3.11
    ok "Python 3.11 installed"
fi

# -- Step 3: install project dependencies ------------------------------------
step "Installing project dependencies (uv sync)"

cd "$REPO_ROOT"
uv sync --quiet
ok "All dependencies installed into .venv"

# -- Step 4: verify core import -----------------------------------------------
step "Verifying installation"

uv run python -c "import redis, pydantic, psutil, yaml, openai; print('Core imports OK')" \
    && ok "Core packages verified" \
    || die "Import check failed — try running 'uv sync' again."

# -- Done: print next steps ---------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Setup complete!${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Run scripts with ${BOLD}uv run python <script>${NC} — no venv activation needed."
echo -e "  Or activate manually:  ${BOLD}source .venv/bin/activate${NC}"
echo ""
echo -e "${BOLD}Next steps depend on this device's role:${NC}"
echo ""
echo -e "  ${CYAN}Desktop PC (control plane + worker):${NC}"
echo -e "    1.  docker compose -f infra/docker-compose.yml up -d"
echo -e "    2.  uv run python scripts/detect_hardware.py --is-control-plane"
echo -e "    3.  uv run python scripts/run_worker.py --config config/<device-id>.yaml"
echo ""
echo -e "  ${CYAN}Any other device (laptop, etc.):${NC}"
echo -e "    1.  uv run python scripts/detect_hardware.py"
echo -e "    2.  uv run python scripts/run_worker.py --config config/<device-id>.yaml"
echo ""
echo -e "  ${CYAN}Mobile devices (iPad, iPhone, Android):${NC}"
echo -e "    See config/templates/tier0_mobile.yaml for prerequisites (mlx_lm / MLC-LLM),"
echo -e "    then run:  python scripts/detect_hardware.py"
echo ""
