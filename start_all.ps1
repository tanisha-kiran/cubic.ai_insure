# Start cubic.ai: the model server (Ollama) and the web app, then open the browser.
#
#   powershell -ExecutionPolicy Bypass -File .\start_all.ps1                  # this PC only (http://localhost:8000)
#   powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -Https           # + HTTPS (https://localhost:8443)
#   powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -Lan             # reachable from other devices (implies -Https)
#   powershell -ExecutionPolicy Bypass -File .\start_all.ps1 -NoBrowser       # do not open a browser
#
# Stop everything with:  powershell -ExecutionPolicy Bypass -File .\stop_all.ps1
param([switch]$Https, [switch]$Lan, [switch]$NoBrowser, [switch]$NoModels)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
# `docvqa_env` is the complete offline runtime (including PyMuPDF/fitz). A
# lightweight .venv can exist for insurance tests but does not run the web app.
$python = @("docvqa_env\Scripts\python.exe", ".venv\Scripts\python.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) { Write-Host "No Python environment found. Run install_offline.ps1 first." -ForegroundColor Red; exit 1 }

# 1. the model server -- low-power settings keep a laptop usable (see start_ollama_lowpower.ps1)
if (-not $NoModels) {
    & powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\start_ollama_lowpower.ps1" | Select-Object -Last 1
}

# 2. the web app, with models forced to load from the local cache only
$useHttps = $Https -or $Lan
$appArgs = @("app_backend.py")
if ($Lan) { $appArgs += "--lan" }
if ($useHttps) { $appArgs += "--https" }
$port = if ($useHttps) { 8443 } else { 8000 }
$scheme = if ($useHttps) { "https" } else { "http" }

New-Item -ItemType Directory -Force logs | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$log = "logs\server.log"
$proc = Start-Process -FilePath $python -ArgumentList $appArgs -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "logs\server.err.log"

# 3. wait until it answers
$url = "${scheme}://localhost:$port"
$ready = $false
for ($i = 0; $i -lt 90 -and -not $ready; $i++) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) { break }
    try {
        # "is the port open?" works the same for http and for https with a self-signed certificate,
        # and in both Windows PowerShell 5.1 and PowerShell 7
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $port)
        $ready = $tcp.Connected
        $tcp.Close()
    } catch { }
}
if ($proc.HasExited -or -not $ready) {
    Write-Host "cubic.ai did not start. Last lines of logs\server.err.log:" -ForegroundColor Red
    if (Test-Path "logs\server.err.log") { Get-Content "logs\server.err.log" -Tail 15 }
    if ($proc.HasExited -and $ready) {
        Write-Host "Port $port is already serving another local process. Stop that process or use its existing cubic.ai page." -ForegroundColor Yellow
    }
    exit 1
}

Write-Host ""
Write-Host "cubic.ai is running at $url" -ForegroundColor Green
if ($Lan) {
    $ips = Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -ExpandProperty IPAddress
    foreach ($ip in $ips) { Write-Host "  other devices on this network:  https://${ip}:8443   (accept the one-time certificate warning)" }
}
$creds = "data\first_run_credentials.txt"
if (Test-Path $creds) {
    Write-Host ""
    Write-Host "First sign-in details (delete data\first_run_credentials.txt afterwards):" -ForegroundColor Yellow
    Get-Content $creds | ForEach-Object { Write-Host "  $_" }
}
Write-Host ""
Write-Host "Models take about a minute to load on the first question. Stop everything with stop_all.ps1."
if (-not $NoBrowser) { Start-Process $url }
