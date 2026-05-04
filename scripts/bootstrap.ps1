# =============================================================================
# AIMESH bootstrap — Windows (PowerShell)
# Sets up the Python environment using uv. Safe to re-run at any time.
#
# Usage (from repo root):
#   .\scripts\bootstrap.ps1
#
# If you see "cannot be loaded because running scripts is disabled", run this
# once in an elevated PowerShell, then retry:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# =============================================================================
#Requires -Version 5.1

# -- Helpers ------------------------------------------------------------------
function Step  { param($msg) Write-Host "`n▶  $msg" -ForegroundColor Cyan }
function Ok    { param($msg) Write-Host "✓  $msg" -ForegroundColor Green }
function Warn  { param($msg) Write-Host "!  $msg" -ForegroundColor Yellow }
function Err   { param($msg) Write-Host "✗  $msg" -ForegroundColor Red; exit 1 }

# -- Locate repo root ---------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor White
Write-Host "  AIMESH — First-time setup" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor White

# -- Step 1: install uv if missing --------------------------------------------
Step "Checking for uv (Python package manager)"

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCmd) {
    $uvVer = & uv --version 2>&1
    Ok "uv already installed ($uvVer)"
} else {
    Warn "uv not found — installing now..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Err "Failed to download uv installer. Check your internet connection, then install manually: https://docs.astral.sh/uv/"
    }

    # Refresh PATH for this session
    $uvDefaultPath = "$env:USERPROFILE\.local\bin"
    if (Test-Path "$uvDefaultPath\uv.exe") {
        $env:PATH = "$uvDefaultPath;$env:PATH"
    }

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Err "uv installation failed or not in PATH. Please restart PowerShell and re-run this script."
    }

    $uvVer = & uv --version 2>&1
    Ok "uv installed ($uvVer)"
    Warn "Restart PowerShell to make uv available in all future sessions."
}

# -- Step 2: Python 3.11+ -----------------------------------------------------
Step "Checking Python 3.11+"

$pythonFound = & uv python find 3.11 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "Python 3.11 found ($pythonFound)"
} else {
    Warn "Python 3.11 not found — installing via uv..."
    & uv python install 3.11
    if ($LASTEXITCODE -ne 0) { Err "Python 3.11 installation failed." }
    Ok "Python 3.11 installed"
}

# -- Step 3: install project dependencies ------------------------------------
Step "Installing project dependencies (uv sync)"

Set-Location $RepoRoot
& uv sync --quiet
if ($LASTEXITCODE -ne 0) { Err "uv sync failed. Check the error output above." }
Ok "All dependencies installed into .venv"

# -- Step 4: verify core imports ----------------------------------------------
Step "Verifying installation"

$verifyResult = & uv run python -c "import redis, pydantic, psutil, yaml, openai; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "Core packages verified"
} else {
    Err "Import check failed: $verifyResult`nTry running 'uv sync' again."
}

# -- Done: print next steps ---------------------------------------------------
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""
Write-Host "  Run scripts with " -NoNewline
Write-Host "uv run python <script>" -ForegroundColor White -NoNewline
Write-Host " — no venv activation needed."
Write-Host "  Or activate manually: " -NoNewline
Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Next steps depend on this device's role:" -ForegroundColor White
Write-Host ""
Write-Host "  Desktop PC (control plane + worker):" -ForegroundColor Cyan
Write-Host "    1.  docker compose -f infra/docker-compose.yml up -d"
Write-Host "    2.  uv run python scripts/detect_hardware.py --is-control-plane"
Write-Host "    3.  uv run python scripts/run_worker.py --config config/<device-id>.yaml"
Write-Host ""
Write-Host "  Any other device (laptop, etc.):" -ForegroundColor Cyan
Write-Host "    1.  uv run python scripts/detect_hardware.py"
Write-Host "    2.  uv run python scripts/run_worker.py --config config/<device-id>.yaml"
Write-Host ""
Write-Host "  Mobile devices (iPad, iPhone, Android):" -ForegroundColor Cyan
Write-Host "    See config/templates/tier0_mobile.yaml for prerequisites (mlx_lm / MLC-LLM),"
Write-Host "    then run:  python scripts/detect_hardware.py"
Write-Host ""
