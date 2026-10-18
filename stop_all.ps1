# Stop cubic.ai: the web app and the model server, freeing the CPU and memory.
#   powershell -ExecutionPolicy Bypass -File .\stop_all.ps1
Set-Location $PSScriptRoot

foreach ($port in 8000, 8443) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($l in $listeners) {
        Stop-Process -Id $l.OwningProcess -Force -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Stopped the web app (port $port)."
    }
}
& powershell -ExecutionPolicy Bypass -File "$PSScriptRoot\start_ollama_lowpower.ps1" -Stop
$left = @(Get-Process | Where-Object { $_.Name -in @("ollama", "llama-server", "ollama app") }).Count
Write-Host "Model processes still running: $left"
