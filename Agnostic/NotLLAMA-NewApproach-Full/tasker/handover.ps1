#Requires -Version 7.0
<#
.SYNOPSIS
    Handover + Compact protocol for dual-agent workflow
.DESCRIPTION
    Allows both agents to:
    1. Handover work to each other mid-task
    2. Signal when they need to compact
    3. Transfer state so the other agent can continue
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("handover", "compact", "resume", "state", "peek")]
    [string]$Action,

    [string]$From = "",

    [string]$To = "",

    [string]$TaskId = "",

    [string]$Reason = "",

    [string]$StateFile = ""
)

$HandoverDir = "$PSScriptRoot\handover"
$SignalDir = "$PSScriptRoot\signals"
$StateDir = "$PSScriptRoot\state"

@($HandoverDir, $StateDir) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

# ============================================================
# HANDOVER: One agent passes work to the other
# ============================================================
function Handover-Work {
    param([string]$From, [string]$To, [string]$TaskId, [string]$Reason)

    $handover = @{
        from = $From
        to = $To
        task_id = $TaskId
        reason = $Reason
        timestamp = Get-Date -Format "o"
        status = "pending"
    }

    $handoverFile = "$HandoverDir\handover_$TaskId`_$From`_to_$To.json"
    $handover | ConvertTo-Json -Depth 5 | Set-Content $handoverFile

    # Signal the other agent
    $signalFile = "$SignalDir\handover_$TaskId`_$To.signal"
    $signalFile | Set-Content $handoverFile

    Write-Host "HANDOVER: $From -> $To" -ForegroundColor Cyan
    Write-Host "  Task: $TaskId" -ForegroundColor Gray
    Write-Host "  Reason: $Reason" -ForegroundColor Gray
    Write-Host "  File: $handoverFile" -ForegroundColor Gray
}

# ============================================================
# COMPACT: Agent signals it needs to compact context
# ============================================================
function Signal-Compact {
    param([string]$Agent, [string]$TaskId, [string]$Reason)

    # Save current state before compacting
    $state = @{
        agent = $Agent
        task_id = $TaskId
        reason = $Reason
        timestamp = Get-Date -Format "o"
        files_modified = @()
        files_read = @()
        progress = ""
        next_steps = @()
        issues = @()
    }

    $stateFile = "$StateDir\compact_${Agent}_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    $state | ConvertTo-Json -Depth 10 | Set-Content $stateFile

    # Signal compact
    $signalFile = "$SignalDir\compact_${Agent}.signal"
    @{
        agent = $Agent
        task_id = $TaskId
        reason = $Reason
        state_file = $stateFile
        timestamp = Get-Date -Format "o"
    } | ConvertTo-Json -Depth 5 | Set-Content $signalFile

    Write-Host "COMPACT SIGNAL: $Agent" -ForegroundColor Yellow
    Write-Host "  Task: $TaskId" -ForegroundColor Gray
    Write-Host "  Reason: $Reason" -ForegroundColor Gray
    Write-Host "  State saved: $stateFile" -ForegroundColor Gray
    Write-Host "  Run /compact now, then continue with:" -ForegroundColor Cyan
    Write-Host "    .\tasker\handover.ps1 -Action resume -From $Agent -TaskId $TaskId" -ForegroundColor Gray
}

# ============================================================
# RESUME: Agent picks up where the other left off
# ============================================================
function Resume-Task {
    param([string]$From, [string]$TaskId)

    # Find the most recent state file for this task
    $stateFiles = Get-ChildItem $StateDir -Filter "compact_*_${TaskId}*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    if ($stateFiles.Count -eq 0) {
        Write-Host "No state found for task $TaskId" -ForegroundColor Red
        return
    }

    $stateFile = $stateFiles[0]
    $state = Get-Content $stateFile.FullName -Raw | ConvertFrom-Json

    Write-Host "RESUMING: $TaskId (from $From)" -ForegroundColor Green
    Write-Host "  Timestamp: $($state.timestamp)" -ForegroundColor Gray
    Write-Host "  Reason: $($state.reason)" -ForegroundColor Gray

    if ($state.files_modified.Count -gt 0) {
        Write-Host "  Files modified:" -ForegroundColor Gray
        $state.files_modified | ForEach-Object { Write-Host "    - $_" -ForegroundColor Gray }
    }

    if ($state.next_steps.Count -gt 0) {
        Write-Host "  Next steps:" -ForegroundColor Yellow
        $state.next_steps | ForEach-Object { Write-Host "    - $_" -ForegroundColor Yellow }
    }

    if ($state.issues.Count -gt 0) {
        Write-Host "  Issues:" -ForegroundColor Red
        $state.issues | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    }

    # Clean up handover file
    $handoverFiles = Get-ChildItem $HandoverDir -Filter "handover_${TaskId}_*" -ErrorAction SilentlyContinue
    $handoverFiles | Remove-Item -Force -ErrorAction SilentlyContinue
}

# ============================================================
# STATE: Save/load current work state
# ============================================================
function Save-State {
    param([string]$Agent, [string]$TaskId)

    $state = @{
        agent = $Agent
        task_id = $TaskId
        timestamp = Get-Date -Format "o"
        files_modified = @()
        files_read = @()
        progress = ""
        next_steps = @()
        issues = @()
    }

    # Read the task file to get context
    $taskFile = "C:\Users\rr\Desktop\Notllama-loc\tasker\in_progress\$TaskId.json"
    if (Test-Path $taskFile) {
        $task = Get-Content $taskFile -Raw | ConvertFrom-Json
        $state.files_read = $task.files_to_read
        $state.files_modified = $task.files_to_write
    }

    $stateFile = "$StateDir\state_${Agent}_${TaskId}.json"
    $state | ConvertTo-Json -Depth 10 | Set-Content $stateFile

    Write-Host "State saved: $stateFile" -ForegroundColor Green
    return $stateFile
}

function Load-State {
    param([string]$Agent, [string]$TaskId)

    $stateFile = "$StateDir\state_${Agent}_${TaskId}.json"
    if (Test-Path $stateFile) {
        $state = Get-Content $stateFile -Raw | ConvertFrom-Json
        Write-Host "State loaded: $stateFile" -ForegroundColor Green
        return $state
    }

    Write-Host "No state found for $Agent / $TaskId" -ForegroundColor Yellow
    return $null
}

# ============================================================
# PEEK: Check what the other agent is doing
# ============================================================
function Peek-Agent {
    param([string]$Agent)

    # Check signals
    $signals = Get-ChildItem $SignalDir -Filter "*${Agent}*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    # Check state
    $states = Get-ChildItem $StateDir -Filter "state_${Agent}_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    # Check handovers
    $handovers = Get-ChildItem $HandoverDir -Filter "*to_${Agent}*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    Write-Host "=== $Agent Status ===" -ForegroundColor Cyan

    if ($signals.Count -gt 0) {
        Write-Host "Recent signals:" -ForegroundColor Yellow
        $signals | Select-Object -First 3 | ForEach-Object {
            Write-Host "  $($_.LastWriteTime): $($_.Name)" -ForegroundColor Gray
        }
    }

    if ($states.Count -gt 0) {
        $latestState = Get-Content $states[0].FullName -Raw | ConvertFrom-Json
        Write-Host "Latest state:" -ForegroundColor Yellow
        Write-Host "  Task: $($latestState.task_id)" -ForegroundColor Gray
        Write-Host "  Time: $($latestState.timestamp)" -ForegroundColor Gray
        Write-Host "  Progress: $($latestState.progress)" -ForegroundColor Gray
    }

    if ($handovers.Count -gt 0) {
        Write-Host "Pending handovers:" -ForegroundColor Yellow
        $handovers | ForEach-Object {
            $h = Get-Content $_.FullName -Raw | ConvertFrom-Json
            Write-Host "  From: $($h.from) | Task: $($h.task_id) | Reason: $($h.reason)" -ForegroundColor Gray
        }
    }

    if ($signals.Count -eq 0 -and $states.Count -eq 0 -and $handovers.Count -eq 0) {
        Write-Host "  (no activity)" -ForegroundColor Gray
    }
}

# Execute
switch ($Action) {
    "handover" {
        if (-not $From -or -not $To -or -not $TaskId) {
            Write-Host "Usage: handover.ps1 -Action handover -From main -To deepseek -TaskId fix_001 -Reason 'stuck on X'" -ForegroundColor Yellow
            break
        }
        Handover-Work -From $From -To $To -TaskId $TaskId -Reason $Reason
    }

    "compact" {
        if (-not $From -or -not $TaskId) {
            Write-Host "Usage: handover.ps1 -Action compact -From main -TaskId fix_001 -Reason 'context too long'" -ForegroundColor Yellow
            break
        }
        Signal-Compact -Agent $From -TaskId $TaskId -Reason $Reason
    }

    "resume" {
        if (-not $From -or -not $TaskId) {
            Write-Host "Usage: handover.ps1 -Action resume -From main -TaskId fix_001" -ForegroundColor Yellow
            break
        }
        Resume-Task -From $From -TaskId $TaskId
    }

    "state" {
        if (-not $From -or -not $TaskId) {
            Write-Host "Usage: handover.ps1 -Action state -From main -TaskId fix_001" -ForegroundColor Yellow
            break
        }
        Save-State -Agent $From -TaskId $TaskId
    }

    "peek" {
        if (-not $From) {
            Write-Host "Usage: handover.ps1 -Action peek -From deepseek" -ForegroundColor Yellow
            break
        }
        Peek-Agent -Agent $From
    }
}
