#Requires -Version 7.0
<#
.SYNOPSIS
    Bidirectional signal system between opencode windows
.DESCRIPTION
    Main window and DeepSeek window communicate via shared signal files.
    No polling needed — both watch the signal directory.

    Signal flow:
    ┌──────────────┐    signal files    ┌──────────────┐
    │  MAIN (me)   │ ────────────────> │  DEEPSEEK    │
    │              │ <──────────────── │              │
    │  • dispatch  │    task_001.go     │  • read task  │
    │  • validate  │    task_001.done   │  • write code │
    │  • fix       │    task_001.fail   │  • signal done│
    └──────────────┘    status.json     └──────────────┘
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("signal", "wait", "watch", "status", "clear", "init", "mailbox")]
    [string]$Action,

    [string]$TaskId = "",

    [ValidateSet("go", "done", "fail", "need-help", "waiting", "ready")]
    [string]$Type = "",

    [string]$Message = "",

    [int]$TimeoutSec = 300
)

$SignalDir = "$PSScriptRoot\signals"
$MailboxDir = "$PSScriptRoot\mailbox"
$StatusFile = "$SignalDir\status.json"

function Ensure-Dirs {
    @($SignalDir, $MailboxDir) | ForEach-Object {
        New-Item -ItemType Directory -Force -Path $_ | Out-Null
    }
}

function Init-Status {
    $status = @{
        main = @{
            state = "idle"
            current_task = $null
            last_signal = $null
            last_signal_time = $null
        }
        deepseek = @{
            state = "idle"
            current_task = $null
            last_signal = $null
            last_signal_time = $null
        }
    }
    $status | ConvertTo-Json -Depth 10 | Set-Content $StatusFile
    return $status
}

function Get-Status {
    if (-not (Test-Path $StatusFile)) {
        return Init-Status
    }
    return Get-Content $StatusFile -Raw | ConvertFrom-Json
}

function Save-Status($s) {
    $s | ConvertTo-Json -Depth 10 | Set-Content $StatusFile
}

function Send-Signal {
    param([string]$TaskId, [string]$Type, [string]$Message, [string]$From)

    Ensure-Dirs

    $signal = @{
        task_id = $TaskId
        type = $Type
        message = $Message
        from = $From
        timestamp = Get-Date -Format "o"
    }

    # Write signal file
    $signalFile = "$SignalDir\$TaskId`_$Type.signal"
    $signal | ConvertTo-Json -Depth 5 | Set-Content $signalFile

    # Update status
    $status = Get-Status
    $status.$From.state = $Type
    $status.$From.current_task = $TaskId
    $status.$From.last_signal = $Type
    $status.$From.last_signal_time = Get-Date -Format "o"
    Save-Status $status

    Write-Host "SIGNAL SENT: $TaskId -> $Type" -ForegroundColor Green
    Write-Host "  From: $From" -ForegroundColor Gray
    Write-Host "  Message: $Message" -ForegroundColor Gray
}

function Wait-Signal {
    param([string]$TaskId, [string]$ExpectedType, [int]$Timeout)

    Ensure-Dirs

    $start = Get-Date
    Write-Host "Waiting for signal: $TaskId`_$ExpectedType (timeout: ${Timeout}s)" -ForegroundColor Yellow

    while ($true) {
        # Check for signal
        $signalFile = "$SignalDir\$TaskId`_$ExpectedType.signal"
        if (Test-Path $signalFile) {
            $signal = Get-Content $signalFile -Raw | ConvertFrom-Json
            Write-Host "SIGNAL RECEIVED: $TaskId -> $ExpectedType" -ForegroundColor Green
            Write-Host "  From: $($signal.from)" -ForegroundColor Gray
            Write-Host "  Message: $($signal.message)" -ForegroundColor Gray

            # Clean up signal file
            Remove-Item $signalFile -Force
            return $signal
        }

        # Check timeout
        $elapsed = ((Get-Date) - $start).TotalSeconds
        if ($elapsed -ge $Timeout) {
            Write-Host "TIMEOUT waiting for $TaskId`_$ExpectedType" -ForegroundColor Red
            return $null
        }

        Start-Sleep -Milliseconds 500
    }
}

function Watch-Signals {
    param([int]$Timeout)

    Ensure-Dirs

    Write-Host "Watching for signals (Ctrl+C to stop)..." -ForegroundColor Cyan
    $start = Get-Date

    while ($true) {
        $signals = Get-ChildItem $SignalDir -Filter "*.signal" -ErrorAction SilentlyContinue

        if ($signals.Count -gt 0) {
            foreach ($s in $signals) {
                $content = Get-Content $s.FullName -Raw | ConvertFrom-Json
                Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Signal: $($content.task_id) -> $($content.type)" -ForegroundColor Cyan
                Write-Host "  From: $($content.from)" -ForegroundColor Gray
                Write-Host "  Message: $($content.message)" -ForegroundColor Gray
            }
        }

        # Show status
        $status = Get-Status
        Write-Host "`r[$(Get-Date -Format 'HH:mm:ss')] Main: $($status.main.state) | DeepSeek: $($status.deepseek.state)   " -NoNewline

        # Check timeout
        $elapsed = ((Get-Date) - $start).TotalSeconds
        if ($elapsed -ge $Timeout) {
            Write-Host "`nWatch timeout" -ForegroundColor Yellow
            break
        }

        Start-Sleep -Seconds 1
    }
}

function Send-Mailbox {
    param([string]$From, [string]$To, [string]$Subject, [string]$Body)

    Ensure-Dirs

    $msg = @{
        from = $From
        to = $To
        subject = $Subject
        body = $Body
        timestamp = Get-Date -Format "o"
        read = $false
    }

    $msgId = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $msgFile = "$MailboxDir\$msgId.json"
    $msg | ConvertTo-Json -Depth 5 | Set-Content $msgFile

    Write-Host "MAILBOX: $From -> $To: $Subject" -ForegroundColor Green
}

function Read-Mailbox {
    param([string]$For)

    Ensure-Dirs

    $messages = Get-ChildItem $MailboxDir -Filter "*.json" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime

    $unread = @()
    foreach ($m in $messages) {
        $msg = Get-Content $m.FullName -Raw | ConvertFrom-Json
        if ($msg.to -eq $For -and -not $msg.read) {
            $unread += $msg
            Write-Host "`nFROM: $($msg.from)" -ForegroundColor Cyan
            Write-Host "SUBJECT: $($msg.subject)" -ForegroundColor White
            Write-Host "BODY:" -ForegroundColor Gray
            Write-Host $msg.body

            # Mark as read
            $msg.read = $true
            $msg | ConvertTo-Json -Depth 5 | Set-Content $m.FullName
        }
    }

    if ($unread.Count -eq 0) {
        Write-Host "No unread messages for $For" -ForegroundColor Gray
    }

    return $unread
}

function Show-Status {
    $status = Get-Status

    Write-Host "`n=== Agent Status ===" -ForegroundColor Cyan

    Write-Host "`n  MAIN (opencode):" -ForegroundColor Yellow
    Write-Host "    State:  $($status.main.state)" -ForegroundColor Gray
    Write-Host "    Task:   $($status.main.current_task)" -ForegroundColor Gray
    Write-Host "    Signal: $($status.main.last_signal) @ $($status.main.last_signal_time)" -ForegroundColor Gray

    Write-Host "`n  DEEPSEEK:" -ForegroundColor Yellow
    Write-Host "    State:  $($status.deepseek.state)" -ForegroundColor Gray
    Write-Host "    Task:   $($status.deepseek.current_task)" -ForegroundColor Gray
    Write-Host "    Signal: $($status.deepseek.last_signal) @ $($status.deepseek.last_signal_time)" -ForegroundColor Gray

    Write-Host "`n  Pending Signals:" -ForegroundColor Yellow
    $signals = Get-ChildItem $SignalDir -Filter "*.signal" -ErrorAction SilentlyContinue
    if ($signals.Count -eq 0) {
        Write-Host "    (none)" -ForegroundColor Gray
    } else {
        $signals | ForEach-Object {
            $s = Get-Content $_.FullName -Raw | ConvertFrom-Json
            Write-Host "    $($s.task_id) -> $($s.type) (from $($s.from))" -ForegroundColor Gray
        }
    }
}

function Clear-Signals {
    Ensure-Dirs
    Get-ChildItem $SignalDir -Filter "*.signal" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem $MailboxDir -Filter "*.json" -ErrorAction SilentlyContinue | Remove-Item -Force
    Init-Status | Out-Null
    Write-Host "All signals and mailbox cleared" -ForegroundColor Green
}

# Execute
Ensure-Dirs

switch ($Action) {
    "init" {
        Init-Status | Out-Null
        Write-Host "Signal system initialized" -ForegroundColor Green
        Show-Status
    }

    "signal" {
        if (-not $TaskId -or -not $Type) {
            Write-Host "Usage: signals.ps1 -Action signal -TaskId task_001 -Type go|done|fail|need-help|waiting|ready -Message 'msg' -From main|deepseek" -ForegroundColor Yellow
            break
        }
        $from = if ($Message -match "^from:(.+)") { $Matches[1] } else { "main" }
        Send-Signal -TaskId $TaskId -Type $Type -Message $Message -From $from
    }

    "wait" {
        if (-not $TaskId -or -not $Type) {
            Write-Host "Usage: signals.ps1 -Action wait -TaskId task_001 -Type done|fail|ready" -ForegroundColor Yellow
            break
        }
        Wait-Signal -TaskId $TaskId -ExpectedType $Type -Timeout $TimeoutSec
    }

    "watch" {
        Watch-Signals -Timeout $TimeoutSec
    }

    "status" {
        Show-Status
    }

    "clear" {
        Clear-Signals
    }

    "mailbox" {
        if (-not $Message) {
            # Read mode
            Read-Mailbox -For "main"
        } else {
            # Send mode: -Message "to:deepseek|subject:Fix bug|body:Please fix rope.comp"
            $parts = $Message -split '\|'
            $to = ($parts | Where-Object { $_ -match "^to:" }) -replace "^to:", ""
            $subject = ($parts | Where-Object { $_ -match "^subject:" }) -replace "^subject:", ""
            $body = ($parts | Where-Object { $_ -match "^body:" }) -replace "^body:", ""
            Send-Mailbox -From "main" -To $to -Subject $subject -Body $body
        }
    }
}
