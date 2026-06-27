#Requires -Version 7.0
<#
.SYNOPSIS
    Safe agent dispatcher — prevents concurrent writes and build conflicts
.DESCRIPTION
    Usage from my side (opencode main window):
      1. Add tasks to queue
      2. Call safe-dispatch.ps1 to send work to DeepSeek
      3. It locks files, sends task, waits for completion, validates, unlocks
#>

param(
    [Parameter(Mandatory)]
    [string]$TaskId,

    [string]$TaskDir = "$PSScriptRoot\pending"
)

$ErrorActionPreference = "Stop"
$TaskerRoot = $PSScriptRoot
$InProgressDir = "$TaskerRoot\in_progress"
$CompletedDir = "$TaskerRoot\completed"
$FailedDir = "$TaskerRoot\failed"
$LogsDir = "$TaskerRoot\logs"
$PromptsDir = "$TaskerRoot\prompts"
$LockScript = "$TaskerRoot\lock.ps1"

# Load task
$taskFile = "$TaskDir\$TaskId.json"
if (-not (Test-Path $taskFile)) {
    Write-Host "Task not found: $taskFile" -ForegroundColor Red
    exit 1
}

$task = Get-Content $taskFile -Raw | ConvertFrom-Json

Write-Host "`n=== DISPATCHING: $TaskId ===" -ForegroundColor Cyan
Write-Host "Title: $($task.title)" -ForegroundColor Gray
Write-Host "Files to write: $($task.files_to_write -join ', ')" -ForegroundColor Gray

# Step 1: Check file locks
Write-Host "`n[1/6] Checking file locks..." -ForegroundColor Yellow
$checkResult = & $LockScript -Action check -Files $task.files_to_write
if (-not $checkResult.CanProceed) {
    Write-Host "BLOCKED: Other task owns these files. Waiting..." -ForegroundColor Yellow

    # Wait for files to become available (max 300s)
    $waited = 0
    while (-not $checkResult.CanProceed -and $waited -lt 300) {
        Start-Sleep -Seconds 5
        $waited += 5
        $checkResult = & $LockScript -Action check -Files $task.files_to_write
    }

    if (-not $checkResult.CanProceed) {
        Write-Host "TIMEOUT: Files still locked after 300s" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Claim files
Write-Host "[2/6] Claiming files..." -ForegroundColor Yellow
& $LockScript -Action claim -TaskId $TaskId -Files $task.files_to_write

# Step 3: Acquire build lock
Write-Host "[3/6] Acquiring build lock..." -ForegroundColor Yellow
$buildLocked = & $LockScript -Action build-lock -TaskId $TaskId
if (-not $buildLocked) {
    Write-Host "Failed to acquire build lock" -ForegroundColor Red
    & $LockScript -Action release -TaskId $TaskId
    exit 1
}

# Step 4: Move task to in_progress
Write-Host "[4/6] Moving task to in_progress..." -ForegroundColor Yellow
$destFile = "$InProgressDir\$TaskId.json"
$task.status = "in_progress"
$task.started = Get-Date -Format "o"
$task | ConvertTo-Json -Depth 10 | Set-Content $destFile
Remove-Item $taskFile -Force -ErrorAction SilentlyContinue

# Step 5: Generate prompt
Write-Host "[5/6] Generating prompt..." -ForegroundColor Yellow

# Read base template
$templateFile = "$PromptsDir\template_base.md"
$template = ""
if (Test-Path $templateFile) {
    $template = Get-Content $templateFile -Raw
} else {
    # Inline template
    $template = @"
# Task: {TITLE}

## Context
Working on RDNA4 LLM inference engine at C:\Users\rr\Desktop\Notllama-loc

## Goal
{DESCRIPTION}

## Constraints
- Read ALL relevant files first
- Follow existing code conventions
- Do NOT add comments unless asked
- Build must pass: cd build && cmake --build . --config Release

## Files to Read
{FILES_TO_READ}

## Files to Modify
{FILES_TO_WRITE}
"@
}

# Build file lists
$filesRead = ($task.files_to_read | ForEach-Object { "- $_" }) -join "`n"
$filesWrite = ($task.files_to_write | ForEach-Object { "- $_" }) -join "`n"

$prompt = $template `
    -replace '\{TITLE\}', $task.title `
    -replace '\{DESCRIPTION\}', $task.description `
    -replace '\{FILES_TO_READ\}', $filesRead `
    -replace '\{FILES_TO_WRITE\}', $filesWrite

# Save prompt
$promptFile = "$LogsDir\$TaskId`_prompt.md"
$prompt | Set-Content $promptFile
Write-Host "Prompt: $promptFile" -ForegroundColor Gray

# Step 6: Return prompt for agent dispatch
Write-Host "[6/6] Ready for dispatch" -ForegroundColor Green
Write-Host ""
Write-Host "PROMPT_START" -ForegroundColor White
$prompt
Write-Host "PROMPT_END" -ForegroundColor White
Write-Host ""
Write-Host "NEXT: After agent completes, run:" -ForegroundColor Cyan
Write-Host "  .\tasker\complete.ps1 -TaskId $TaskId -Success `$true" -ForegroundColor Gray
Write-Host "  .\tasker\complete.ps1 -TaskId $TaskId -Success `$false" -ForegroundColor Gray
