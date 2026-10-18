# Start Ollama in laptop-friendly mode (CPU-only machines).
#
#   powershell -ExecutionPolicy Bypass -File start_ollama_lowpower.ps1
#   powershell -ExecutionPolicy Bypass -File start_ollama_lowpower.ps1 -Stop
#
# - Points Ollama at the folder the models actually live in (the user-level
#   OLLAMA_MODELS variable points at an empty E:\ollama_models).
# - Keeps the 7B reasoner AND the 3B writer in RAM (~10 GB of 24 GB): swapping
#   them reloads 6 GB from disk each time, which cost ~150 s per question.
#   Serves one request at a time.
# - Runs at BelowNormal priority so the desktop stays responsive. Thread count
#   is capped per request by hardware_profile.py (physical cores - 1).
# - Unloads models after 15 idle minutes so the CPU and RAM go back to idle.
param([switch]$Stop)

$ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

Get-Process | Where-Object { $_.Name -in @("ollama", "llama-server", "ollama app") } |
    Stop-Process -Force -Confirm:$false -ErrorAction SilentlyContinue
if ($Stop) { "Ollama stopped."; exit 0 }

$env:OLLAMA_MODELS = "$env:USERPROFILE\.ollama\models"
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_KEEP_ALIVE = "15m"

$proc = Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 3
$proc.PriorityClass = "BelowNormal"

# Child llama-server processes inherit the priority class from ollama serve.
& $ollama list
"Ollama running (PID $($proc.Id), BelowNormal priority). Stop with: -Stop"
