#Requires -Version 7.0
<#
.SYNOPSIS
    File lock manager for auto-tasker
.DESCRIPTION
    Prevents concurrent writes to same files and coordinates builds.
    Only ONE agent can build at a time. Agents must claim files before writing.
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("claim", "release", "check", "build-lock", "build-unlock", "status")]
    [string]$Action,

    [string]$TaskId = "",
    [string[]]$Files = @()
)

$LockDir = "$PSScriptRoot\locks"
$FileLocks = "$LockDir\file_claims"
$BuildLock = "$LockDir\build.lock"
$Manifest = "$LockDir\manifest.json"

# Ensure dirs
@($LockDir, $FileLocks) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

function Get-Manifest {
    if (Test-Path $Manifest) {
        return Get-Content $Manifest -Raw | ConvertFrom-Json
    }
    return @{ files = @{}; build_lock = $null; build_lock_time = $null }
}

function Save-Manifest($m) {
    $m | ConvertTo-Json -Depth 10 | Set-Content $Manifest
}

function Claim-Files {
    param([string]$TaskId, [string[]]$Files)

    $m = Get-Manifest

    # Check if any files are already claimed by another task
    $conflicts = @()
    foreach ($f in $Files) {
        $norm = $f.Replace('\', '/').ToLower()
        if ($m.files.$norm -and $m.files.$norm -ne $TaskId) {
            $conflicts += [PSCustomObject]@{
                File = $f
                ClaimedBy = $m.files.$norm
            }
        }
    }

    if ($conflicts.Count -gt 0) {
        Write-Host "CONFLICT: Files claimed by another task:" -ForegroundColor Red
        $conflicts | ForEach-Object {
            Write-Host "  $($_.File) -> claimed by $($_.ClaimedBy)" -ForegroundColor Yellow
        }
        return $false
    }

    # Claim all files
    foreach ($f in $Files) {
        $norm = $f.Replace('\', '/').ToLower()
        $m.files.$norm = $TaskId
    }
    Save-Manifest $m

    Write-Host "Claimed $($Files.Count) files for task $TaskId" -ForegroundColor Green
    return $true
}

function Release-Files {
    param([string]$TaskId)

    $m = Get-Manifest
    $toRemove = @()

    foreach ($prop in $m.files.PSObject.Properties) {
        if ($prop.Value -eq $TaskId) {
            $toRemove += $prop.Name
        }
    }

    foreach ($r in $toRemove) {
        $m.files.PSObject.Properties.Remove($r)
    }
    Save-Manifest $m

    Write-Host "Released $($toRemove.Count) files from task $TaskId" -ForegroundColor Green
}

function Check-Files {
    param([string[]]$Files)

    $m = Get-Manifest
    $available = @()
    $blocked = @()

    foreach ($f in $Files) {
        $norm = $f.Replace('\', '/').ToLower()
        if ($m.files.$norm) {
            $blocked += [PSCustomObject]@{
                File = $f
                ClaimedBy = $m.files.$norm
            }
        } else {
            $available += $f
        }
    }

    return @{
        Available = $available
        Blocked = $blocked
        CanProceed = ($blocked.Count -eq 0)
    }
}

function Get-BuildLock {
    param([string]$TaskId)

    if (Test-Path $BuildLock) {
        $existing = Get-Content $BuildLock
        Write-Host "BUILD LOCKED by: $existing" -ForegroundColor Yellow
        Write-Host "Waiting..." -ForegroundColor Gray

        # Wait up to 120 seconds
        $waited = 0
        while ((Test-Path $BuildLock) -and $waited -lt 120) {
            Start-Sleep -Seconds 2
            $waited += 2
        }

        if (Test-Path $BuildLock) {
            Write-Host "BUILD LOCK TIMEOUT" -ForegroundColor Red
            return $false
        }
    }

    # Acquire lock
    $TaskId | Set-Content $BuildLock
    Write-Host "Build locked by: $TaskId" -ForegroundColor Green
    return $true
}

function Release-BuildLock {
    if (Test-Path $BuildLock) {
        Remove-Item $BuildLock -Force
        Write-Host "Build lock released" -ForegroundColor Green
    }
}

function Show-Status {
    $m = Get-Manifest

    Write-Host "`n=== Lock Status ===" -ForegroundColor Cyan

    Write-Host "`nFile Claims:" -ForegroundColor Yellow
    if ($m.files.PSObject.Properties.Count -eq 0) {
        Write-Host "  (none)" -ForegroundColor Gray
    } else {
        $m.files.PSObject.Properties | ForEach-Object {
            Write-Host "  $($_.Name) -> $($_.Value)" -ForegroundColor Gray
        }
    }

    Write-Host "`nBuild Lock: " -ForegroundColor Yellow
    if (Test-Path $BuildLock) {
        $holder = Get-Content $BuildLock
        Write-Host $holder -ForegroundColor Red
    } else {
        Write-Host "(available)" -ForegroundColor Green
    }
}

# Execute
switch ($Action) {
    "claim" {
        if (-not $TaskId -or $Files.Count -eq 0) {
            Write-Host "Usage: lock.ps1 -Action claim -TaskId task_001 -Files @('file1.cpp','file2.hpp')" -ForegroundColor Yellow
            break
        }
        Claim-Files -TaskId $TaskId -Files $Files
    }
    "release" {
        if (-not $TaskId) {
            Write-Host "Usage: lock.ps1 -Action release -TaskId task_001" -ForegroundColor Yellow
            break
        }
        Release-Files -TaskId $TaskId
    }
    "check" {
        if ($Files.Count -eq 0) {
            Write-Host "Usage: lock.ps1 -Action check -Files @('file1.cpp','file2.hpp')" -ForegroundColor Yellow
            break
        }
        $result = Check-Files -Files $Files
        if ($result.CanProceed) {
            Write-Host "All files available" -ForegroundColor Green
        } else {
            Write-Host "BLOCKED files:" -ForegroundColor Red
            $result.Blocked | ForEach-Object {
                Write-Host "  $($_.File) -> $($_.ClaimedBy)" -ForegroundColor Yellow
            }
        }
    }
    "build-lock" {
        if (-not $TaskId) {
            Write-Host "Usage: lock.ps1 -Action build-lock -TaskId task_001" -ForegroundColor Yellow
            break
        }
        Get-BuildLock -TaskId $TaskId
    }
    "build-unlock" {
        Release-BuildLock
    }
    "status" {
        Show-Status
    }
}
