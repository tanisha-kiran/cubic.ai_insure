# Install cubic.ai from an offline bundle. Needs NO internet.
# Run inside the bundle folder (made by build_offline_bundle.ps1), after installing Python.
#
#   powershell -ExecutionPolicy Bypass -File .\install_offline.ps1
#   powershell -ExecutionPolicy Bypass -File .\install_offline.ps1 -VerifyModels     # also re-hash the model files (slow)
param([switch]$VerifyModels)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($n, $text) { Write-Host ""; Write-Host "[$n/5] $text" -ForegroundColor Cyan }

Step 1 "Looking for Python"
$wanted = if (Test-Path PYTHON_VERSION.txt) { (Get-Content PYTHON_VERSION.txt).Trim() } else { "3.12" }
$python = $null
foreach ($candidate in @("py -$wanted", "python")) {
    try {
        $version = (Invoke-Expression "$candidate -c `"import sys; print('%d.%d' % sys.version_info[:2])`"" 2>$null)
        if ($version -eq $wanted) { $python = $candidate; break }
    } catch { }
}
if (-not $python) {
    $installer = Get-ChildItem -Filter "python-*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    Write-Host "Python $wanted is not installed." -ForegroundColor Red
    if ($installer) { Write-Host "Install it from the bundle first: $($installer.Name)  (tick 'Add python.exe to PATH'), then run this again." }
    else { Write-Host "Install Python $wanted (64-bit), tick 'Add python.exe to PATH', then run this again." }
    Write-Host "The bundle's packages were built for Python $wanted; other versions will not match."
    exit 1
}
Write-Host "Using Python $wanted ($python)"

Step 2 "Creating the environment and installing packages from the bundle (no internet)"
if (-not (Test-Path ".venv\Scripts\python.exe")) { Invoke-Expression "$python -m venv .venv" }
& .venv\Scripts\python.exe -m pip install --no-index --find-links wheels -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Package install failed. Is the wheels folder complete?" -ForegroundColor Red; exit 1 }

Step 3 "Model server (Ollama)"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama -and -not (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
    if (Test-Path "OllamaSetup.exe") {
        Write-Host "Installing Ollama from the bundle (a window may appear)..."
        Start-Process ".\OllamaSetup.exe" -ArgumentList "/SILENT" -Wait
    } else { Write-Host "Ollama is not installed and the bundle has no installer. Install it, then run this again." -ForegroundColor Red; exit 1 }
} else { Write-Host "Ollama is already installed." }

Step 4 "Copying the AI models into place"
if (Test-Path "models\models.json") {
    $flag = if ($VerifyModels) { "--verify" } else { "" }
    & .venv\Scripts\python.exe package_models.py import --src models $flag
    if ($LASTEXITCODE -ne 0) { Write-Host "Model import failed." -ForegroundColor Red; exit 1 }
} else { Write-Host "No models folder in this bundle; skipping." -ForegroundColor DarkYellow }

Step 5 "Checking everything works offline"
& .venv\Scripts\python.exe verify_install.py
$ok = $LASTEXITCODE -eq 0

Write-Host ""
if ($ok) {
    Write-Host "Installed. Start it with:  powershell -ExecutionPolicy Bypass -File .\start_all.ps1" -ForegroundColor Green
    Write-Host "The first start prints the admin password. Add more people later with:  .venv\Scripts\python.exe manage_users.py add NAME"
} else { Write-Host "Some checks failed (see the FAIL lines above)." -ForegroundColor Red; exit 1 }
