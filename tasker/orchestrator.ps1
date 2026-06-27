#Requires -Version 7.0
<#
.SYNOPSIS
    Master orchestrator — dispatches tasks, handles signals, validates results
.DESCRIPTION
    Run this from the main opencode window to manage the full workflow.

    Usage:
      .\orchestrator.ps1 -Action dispatch -TaskId fix_001
      .\orchestrator.ps1 -Action run-all
      .\orchestrator.ps1 -Action next
      .\orchestrator.ps1 -Action status
#>

param(
    [Parameter()]
    [ValidateSet("dispatch", "run-all", "next", "status", "reset", "batch")]
    [string]$Action = "status",

    [string]$TaskId = "",
    [switch]$DryRun
)

$TaskerRoot = $PSScriptRoot
$PendingDir = "$TaskerRoot\pending"
$InProgressDir = "$TaskerRoot\in_progress"
$CompletedDir = "$TaskerRoot\completed"
$FailedDir = "$TaskerRoot\failed"
$LogsDir = "$TaskerRoot\logs"
$LockScript = "$TaskerRoot\lock.ps1"
$SignalScript = "$TaskerRoot\signals.ps1"
$CompleteScript = "$TaskerRoot\complete.ps1"

$ProjectRoot = "C:\Users\rr\Desktop\Notllama-loc"

# ============================================================
# STEP 1: Load task from queue
# ============================================================
function Get-NextTask {
    $tasks = Get-ChildItem $PendingDir -Filter "*.json" -ErrorAction SilentlyContinue |
        Sort-Object { (Get-Content $_.FullName | ConvertFrom-Json).priority }

    if ($tasks.Count -eq 0) {
        return $null
    }

    $taskFile = $tasks[0]
    return Get-Content $taskFile.FullName -Raw | ConvertFrom-Json
}

# ============================================================
# STEP 2: Check file locks + claim files
# ============================================================
function Acquire-Locks {
    param([object]$Task)

    Write-Host "[LOCK] Checking file availability..." -ForegroundColor Yellow

    $check = & $LockScript -Action check -Files $Task.files_to_write
    if ($check -match "BLOCKED") {
        Write-Host "[LOCK] Files blocked — waiting..." -ForegroundColor Yellow
        $waited = 0
        while ($check -match "BLOCKED" -and $waited -lt 300) {
            Start-Sleep -Seconds 5
            $waited += 5
            $check = & $LockScript -Action check -Files $Task.files_to_write
        }
        if ($check -match "BLOCKED") {
            Write-Host "[LOCK] TIMEOUT — files still locked" -ForegroundColor Red
            return $false
        }
    }

    & $LockScript -Action claim -TaskId $Task.id -Files $Task.files_to_write
    return $true
}

# ============================================================
# STEP 3: Acquire build lock
# ============================================================
function Acquire-BuildLock {
    param([string]$TaskId)

    Write-Host "[BUILD] Acquiring build lock..." -ForegroundColor Yellow
    $result = & $LockScript -Action build-lock -TaskId $TaskId
    return $result
}

# ============================================================
# STEP 4: Generate prompt for DeepSeek
# ============================================================
function Generate-Prompt {
    param([object]$Task)

    $filesRead = ($Task.files_to_read | ForEach-Object { "- $_" }) -join "`n"
    $filesWrite = ($Task.files_to_write | ForEach-Object { "- $_" }) -join "`n"

    $prompt = @"
# Task: $($Task.title)

## Context

You are working on an RDNA4 LLM inference engine at `C:\Users\rr\Desktop\Notllama-loc`. The engine runs VibeThinker-3B.Q6_K (qwen2 architecture) end-to-end on Vulkan compute shaders.

## Goal

$($Task.description)

## Constraints

- Read ALL relevant source files before making changes
- Follow existing code conventions (see AGENTS.md)
- Do NOT add comments unless asked
- Do NOT modify files not listed in the task
- Build must pass: ``cd build && cmake --build . --config Release``
- If shader modified: recompile with ``glslc -V shader.comp -o shader.spv``

## Files to Read

$filesRead

## Files to Modify

$filesWrite

## Signal Protocol

When you are DONE (success or failure), output EXACTLY one of these lines:

```
SIGNAL:task_id=$($Task.id)|type=done|message=All changes applied and build passes
```

```
SIGNAL:task_id=$($Task.id)|type=fail|message=Reason for failure
```

Do NOT output anything after the SIGNAL line.

## Expected Output

Describe what you changed and why.
"@

    return $prompt
}

# ============================================================
# STEP 5: Dispatch to agent (returns prompt for Task tool)
# ============================================================
function Dispatch-Task {
    param([object]$Task)

    Write-Host "`n=== DISPATCHING: $($Task.id) ===" -ForegroundColor Cyan
    Write-Host "  Title: $($Task.title)" -ForegroundColor Gray
    Write-Host "  Files: $($Task.files_to_write -join ', ')" -ForegroundColor Gray

    # Generate prompt
    $prompt = Generate-Prompt -Task $Task

    # Save prompt for reference
    $promptFile = "$LogsDir\$($Task.id)_prompt.md"
    $prompt | Set-Content $promptFile
    Write-Host "  Prompt: $promptFile" -ForegroundColor Gray

    # Move to in_progress
    $destFile = "$InProgressDir\$($Task.id).json"
    $Task.status = "in_progress"
    $Task.started = Get-Date -Format "o"
    $Task | ConvertTo-Json -Depth 10 | Set-Content $destFile
    Remove-Item "$PendingDir\$($Task.id).json" -Force -ErrorAction SilentlyContinue

    # Signal that we're dispatching
    & $SignalScript -Action signal -TaskId $Task.id -Type go -Message "Main dispatching task" -From main

    return $prompt
}

# ============================================================
# STEP 6: Validate result
# ============================================================
function Validate-Result {
    param([string]$TaskId, [bool]$AgentSuccess)

    Write-Host "`n=== VALIDATING: $TaskId ===" -ForegroundColor Cyan

    if (-not $AgentSuccess) {
        Write-Host "Agent reported failure" -ForegroundColor Red
        return $false
    }

    # Build check
    Write-Host "[1/3] Building..." -ForegroundColor Yellow
    Push-Location "$ProjectRoot\build"
    $buildOutput = & cmake --build . --config Release 2>&1
    $buildOk = ($LASTEXITCODE -eq 0)
    Pop-Location

    if (-not $buildOk) {
        Write-Host "BUILD FAILED" -ForegroundColor Red
        $buildOutput | Select-Object -Last 10 | ForEach-Object { Write-Host $_ -ForegroundColor DarkRed }
        return $false
    }
    Write-Host "BUILD OK" -ForegroundColor Green

    # Smoke test
    Write-Host "[2/3] Smoke test..." -ForegroundColor Yellow
    $exe = "$ProjectRoot\build\Release\rdna4_llama.exe"
    $json = "$ProjectRoot\build\model\VibeThinker-3B.Q6_K.weights.json"
    $bin = "$ProjectRoot\build\model\VibeThinker-3B.Q6_K.weights.bin"

    if ((Test-Path $exe) -and (Test-Path $json) -and (Test-Path $bin)) {
        $smoke = & $exe $json $bin "Hello" 2>&1
        $smokeStr = $smoke -join "`n"

        if ($smokeStr -match "DEVICE_LOST|CRASH|Access violation") {
            Write-Host "SMOKE TEST FAILED" -ForegroundColor Red
            return $false
        }
        Write-Host "SMOKE TEST OK" -ForegroundColor Green
    } else {
        Write-Host "Skipping smoke test (missing files)" -ForegroundColor Yellow
    }

    Write-Host "[3/3] All checks passed" -ForegroundColor Green
    return $true
}

# ============================================================
# STEP 7: Complete task (release locks, move to completed)
# ============================================================
function Complete-Task {
    param([string]$TaskId, [bool]$Success, [string]$Notes)

    & $CompleteScript -TaskId $TaskId -Success $Success

    if ($Success) {
        & $SignalScript -Action signal -TaskId $TaskId -Type done -Message "Task completed successfully" -From main
    } else {
        & $SignalScript -Action signal -TaskId $TaskId -Type fail -Message $Notes -From main
    }
}

# ============================================================
# ACTIONS
# ============================================================

switch ($Action) {
    "status" {
        & $SignalScript -Action status

        Write-Host "`n=== Task Queue ===" -ForegroundColor Cyan
        $pending = (Get-ChildItem $PendingDir -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
        $inProgress = (Get-ChildItem $InProgressDir -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
        $completed = (Get-ChildItem $CompletedDir -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
        $failed = (Get-ChildItem $FailedDir -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count

        Write-Host "  Pending: $pending | In Progress: $inProgress | Completed: $completed | Failed: $failed" -ForegroundColor Gray
    }

    "next" {
        $task = Get-NextTask
        if (-not $task) {
            Write-Host "No pending tasks" -ForegroundColor Yellow
            break
        }

        Write-Host "Next task: $($task.id) — $($task.title)" -ForegroundColor Cyan
        Write-Host "Run: .\orchestrator.ps1 -Action dispatch -TaskId $($task.id)" -ForegroundColor Gray
    }

    "dispatch" {
        if (-not $TaskId) {
            Write-Host "Usage: .\orchestrator.ps1 -Action dispatch -TaskId fix_001" -ForegroundColor Yellow
            break
        }

        $taskFile = "$PendingDir\$TaskId.json"
        if (-not (Test-Path $taskFile)) {
            Write-Host "Task not found: $TaskId" -ForegroundColor Red
            break
        }

        $task = Get-Content $taskFile -Raw | ConvertFrom-Json

        # Acquire locks
        $locked = Acquire-Locks -Task $task
        if (-not $locked) {
            Write-Host "Could not acquire locks" -ForegroundColor Red
            break
        }

        # Generate prompt (returns it for you to send to agent)
        $prompt = Dispatch-Task -Task $task

        Write-Host "`n--- PROMPT FOR DEEPSEEK ---" -ForegroundColor White
        $prompt
        Write-Host "--- END PROMPT ---" -ForegroundColor White

        Write-Host "`nAfter agent completes, run:" -ForegroundColor Cyan
        Write-Host "  .\orchestrator.ps1 -Action validate -TaskId $TaskId" -ForegroundColor Gray
    }

    "run-all" {
        Write-Host "Running all tasks sequentially..." -ForegroundColor Cyan

        while ($true) {
            $task = Get-NextTask
            if (-not $task) {
                Write-Host "All tasks completed!" -ForegroundColor Green
                break
            }

            Write-Host "`n--- Next task: $($task.id) ---" -ForegroundColor Cyan

            # Dispatch
            $locked = Acquire-Locks -Task $task
            if (-not $locked) {
                Write-Host "Skipping — locks unavailable" -ForegroundColor Yellow
                continue
            }

            $prompt = Dispatch-Task -Task $task

            # Here you would send the prompt to the agent via Task tool
            # For now, just save it and wait
            Write-Host "Prompt ready. Send to agent manually." -ForegroundColor Yellow
            Write-Host "  Prompt file: $LogsDir\$($task.id)_prompt.md" -ForegroundColor Gray
        }
    }

    "batch" {
        # Batch mode: create multiple tasks at once
        $tasks = Get-ChildItem $PendingDir -Filter "*.json" -ErrorAction SilentlyContinue |
            Sort-Object { (Get-Content $_.FullName | ConvertFrom-Json).priority }

        Write-Host "=== Batch Queue ===" -ForegroundColor Cyan
        $i = 1
        foreach ($t in $tasks) {
            $task = Get-Content $t.FullName -Raw | ConvertFrom-Json
            Write-Host "  $i. [$($task.id)] $($task.title)" -ForegroundColor Gray
            $i++
        }

        Write-Host "`nTotal: $($tasks.Count) tasks" -ForegroundColor Gray
        Write-Host "Run: .\orchestrator.ps1 -Action run-all" -ForegroundColor Yellow
    }

    "reset" {
        & $LockScript -Action status
        & $SignalScript -Action clear
        & $LockScript -Action build-unlock

        # Move all failed back to pending
        Get-ChildItem $FailedDir -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
            $task = Get-Content $_.FullName | ConvertFrom-Json
            $task.status = "pending"
            $task.retries = 0
            $task | ConvertTo-Json -Depth 10 | Set-Content "$PendingDir\$($task.id).json"
            Remove-Item $_.FullName -Force
            Write-Host "Reset: $($task.id)" -ForegroundColor Green
        }

        # Move in_progress back to pending
        Get-ChildItem $InProgressDir -Filter "*.json" -ErrorAction SilentlyContinue | ForEach-Object {
            $task = Get-Content $_.FullName | ConvertFrom-Json
            $task.status = "pending"
            $task | ConvertTo-Json -Depth 10 | Set-Content "$PendingDir\$($task.id).json"
            Remove-Item $_.FullName -Force
            Write-Host "Reset (from in_progress): $($task.id)" -ForegroundColor Yellow
        }

        Write-Host "All tasks reset" -ForegroundColor Green
    }
}
