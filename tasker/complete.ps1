#Requires -Version 7.0
<#
.SYNOPSIS
    Completes a task — validates, releases locks, moves to completed/failed
.DESCRIPTION
    Called after agent finishes work. Validates build, runs tests, releases locks.
#>

param(
    [Parameter(Mandatory)]
    [string]$TaskId,

    [Parameter(Mandatory)]
    [bool]$Success,

    [string]$Notes = ""
)

$TaskerRoot = $PSScriptRoot
$InProgressDir = "$TaskerRoot\in_progress"
$CompletedDir = "$TaskerRoot\completed"
$FailedDir = "$TaskerRoot\failed"
$LogsDir = "$TaskerRoot\logs"
$LockScript = "$TaskerRoot\lock.ps1"
$ProjectRoot = "C:\Users\rr\Desktop\Notllama-loc"

# Load task
$taskFile = "$InProgressDir\$TaskId.json"
if (-not (Test-Path $taskFile)) {
    Write-Host "Task $TaskId not found in in_progress" -ForegroundColor Red
    exit 1
}

$task = Get-Content $taskFile -Raw | ConvertFrom-Json

Write-Host "`n=== COMPLETING: $TaskId ===" -ForegroundColor Cyan

if ($Success) {
    # Step 1: Validate build
    Write-Host "[1/3] Building project..." -ForegroundColor Yellow
    Push-Location "$ProjectRoot\build"
    $buildOutput = & cmake --build . --config Release 2>&1
    $buildOk = ($LASTEXITCODE -eq 0)
    Pop-Location

    if (-not $buildOk) {
        Write-Host "BUILD FAILED — marking as failed" -ForegroundColor Red
        $buildOutput | Select-Object -Last 15 | ForEach-Object { Write-Host $_ -ForegroundColor DarkRed }
        $Success = $false
        $Notes = "Build failed after agent changes"
    } else {
        Write-Host "BUILD OK" -ForegroundColor Green
    }

    # Step 2: Shader compilation check
    if ($buildOk) {
        Write-Host "[2/3] Checking shaders..." -ForegroundColor Yellow
        $spirvDir = "$ProjectRoot\build\Release\shaders"
        $shaderDir = "$ProjectRoot\src\kernels"
        $shaderCount = (Get-ChildItem "$shaderDir\*.comp" -ErrorAction SilentlyContinue | Measure-Object).Count
        $spirvCount = (Get-ChildItem "$spirvDir\*.spv" -ErrorAction SilentlyContinue | Measure-Object).Count

        if ($spirvCount -lt $shaderCount) {
            Write-Host "WARNING: Missing SPIR-V ($spirvCount/$shaderCount)" -ForegroundColor Yellow
        } else {
            Write-Host "All $shaderCount shaders compiled" -ForegroundColor Green
        }
    }

    # Step 3: Smoke test
    if ($buildOk) {
        Write-Host "[3/3] Smoke test..." -ForegroundColor Yellow
        $exe = "$ProjectRoot\build\Release\rdna4_llama.exe"
        $json = "$ProjectRoot\build\model\VibeThinker-3B.Q6_K.weights.json"
        $bin = "$ProjectRoot\build\model\VibeThinker-3B.Q6_K.weights.bin"

        if ((Test-Path $exe) -and (Test-Path $json) -and (Test-Path $bin)) {
            $smokeOutput = & $exe $json $bin "Hello" 2>&1
            $smokeStr = $smokeOutput -join "`n"

            if ($smokeStr -match "DEVICE_LOST|CRASH|Access violation") {
                Write-Host "SMOKE TEST FAILED" -ForegroundColor Red
                $Success = $false
                $Notes = "Smoke test failed — device lost"
            } else {
                Write-Host "SMOKE TEST OK" -ForegroundColor Green
            }
        } else {
            Write-Host "Skipping smoke test (missing files)" -ForegroundColor Yellow
        }
    }
}

# Release build lock
& $LockScript -Action build-unlock

# Release file locks
& $LockScript -Action release -TaskId $TaskId

# Save log
$logContent = @"
Task: $TaskId
Title: $($task.title)
Success: $Success
Notes: $Notes
Completed: $(Get-Date -Format "o")
Build: $(if ($Success) { 'PASS' } else { 'FAIL' })
"@

$logFile = "$LogsDir\$TaskId`_result.log"
$logContent | Set-Content $logFile

# Move task
if ($Success) {
    $task.status = "completed"
    $task.completed = Get-Date -Format "o"
    $task.notes = $Notes
    $destFile = "$CompletedDir\$TaskId.json"
    $task | ConvertTo-Json -Depth 10 | Set-Content $destFile
    Remove-Item $taskFile -Force
    Write-Host "`nCOMPLETED: $TaskId" -ForegroundColor Green
} else {
    $task.retries++
    if ($task.retries -ge $task.max_retries) {
        $task.status = "failed"
        $task.failed = Get-Date -Format "o"
        $task.notes = $Notes
        $destFile = "$FailedDir\$TaskId.json"
        Write-Host "`nFAILED (max retries): $TaskId" -ForegroundColor Red
    } else {
        $task.status = "pending"
        $task.notes = $Notes
        $destFile = "$TaskerRoot\pending\$TaskId.json"
        Write-Host "`nRETRY ($($task.retries)/$($task.max_retries)): $TaskId" -ForegroundColor Yellow
    }
    $task | ConvertTo-Json -Depth 10 | Set-Content $destFile
    Remove-Item $taskFile -Force
}

Write-Host "Result: $logFile" -ForegroundColor Gray
