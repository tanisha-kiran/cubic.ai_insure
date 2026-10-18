# cubic.ai Insurance

An offline-first insurance assistant integrated into the cubic.ai chat interface.

It supports:

- Policy questions with clause citations
- Coverage-gap checks for described situations
- Claim checklists and claim drafts
- A mock insurer sandbox that requires explicit confirmation before a submission
- Permission tiers and a tamper-evident audit trail
- Continuing a selected policy and its analysis in the normal cubic.ai chat

## Run locally

1. Install the Python runtime and the local Ollama models required by your environment.
2. Start the application from PowerShell:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start_all.ps1
   ```

3. Open `http://localhost:8000/`.

Stop the backend and Ollama with:

```powershell
powershell -ExecutionPolicy Bypass -File .\stop_all.ps1
```

## Privacy

The repository intentionally excludes local accounts, sessions, uploaded policies, audit records, claim drafts, model files, and Python environments. Those remain on the local machine.

The submission workflow targets a sandbox insurer only. It does not send a claim to a real insurer.
