#Requires -Version 7.0
<#
.SYNOPSIS
    Graphify integration for tasker — enriches prompts with codebase knowledge graph
.DESCRIPTION
    Uses graphify's knowledge graph to:
    1. Find all files related to a task
    2. Map dependencies between components
    3. Enrich prompts with context nodes
    4. Validate changes don't break relationships
#>

param(
    [Parameter(Mandatory)]
    [ValidateSet("context", "deps", "validate", "enrich", "find-related", "community", "god-nodes")]
    [string]$Action,

    [string]$Query = "",

    [string[]]$Files = @(),

    [int]$Depth = 2,

    [int]$MaxNodes = 30
)

$ProjectRoot = "C:\Users\rr\Desktop\Notllama-loc"
$GraphJson = "$ProjectRoot\graphify-out\graph.json"
$RagScript = "$ProjectRoot\tools\rag.py"
$GraphReport = "$ProjectRoot\graphify-out\GRAPH_REPORT.md"

# ============================================================
# Get context nodes for a file or query
# ============================================================
function Get-Context {
    param([string]$Query, [int]$Depth, [int]$MaxNodes)

    # Use rag.py to query the graph
    $result = python $RagScript context $Query 2>&1
    return $result
}

# ============================================================
# Find all files related to a set of files
# ============================================================
function Find-Related {
    param([string[]]$Files, [int]$Depth)

    $allRelated = @{}

    foreach ($file in $Files) {
        # Query graph for this file
        $result = python $RagScript code $file 2>&1
        $allRelated[$file] = $result
    }

    return $allRelated
}

# ============================================================
# Get dependencies for a component
# ============================================================
function Get-Deps {
    param([string]$Query)

    # Query for dependencies
    $result = python $RagScript graph $Query 2>&1
    return $result
}

# ============================================================
# Validate changes don't break relationships
# ============================================================
function Validate-Changes {
    param([string[]]$ModifiedFiles)

    $issues = @()

    foreach ($file in $ModifiedFiles) {
        # Check if file is a god node (highly connected)
        $godNodes = python $RagScript query $file 2>&1
        if ($godNodes -match "god.node|critical|hub") {
            $issues += "WARNING: $file is a highly connected node — changes may have wide impact"
        }
    }

    return $issues
}

# ============================================================
# Enrich a prompt with graph context
# ============================================================
function Enrich-Prompt {
    param([string]$TaskTitle, [string]$TaskDescription, [string[]]$FilesToRead, [string[]]$FilesToWrite)

    # Get context for the task
    $contextQuery = "$TaskTitle $TaskDescription"
    $context = Get-Context -Query $contextQuery -Depth $Depth -MaxNodes $MaxNodes

    # Get related files
    $related = Find-Related -Files ($FilesToRead + $FilesToWrite) -Depth $Depth

    # Get dependencies
    $deps = @()
    foreach ($file in $FilesToWrite) {
        $fileDeps = Get-Deps -Query $file
        $deps += $fileDeps
    }

    # Build enriched context
    $enriched = @"
## Codebase Context (from knowledge graph)

### Related Components
$($context -join "`n")

### File Dependencies
$($related.Keys | ForEach-Object { "- $_: $($_.$_)" } | Out-String)

### Dependency Analysis
$($deps | Select-Object -Unique | Out-String)

### Impact Assessment
$($Validate-Changes -ModifiedFiles $FilesToWrite | Out-String)
"@

    return $enriched
}

# ============================================================
# Get community summary
# ============================================================
function Get-CommunitySummary {
    $result = python $RagScript query "community" 2>&1
    return $result
}

# ============================================================
# Get god nodes (most connected)
# ============================================================
function Get-GodNodes {
    $result = python $RagScript query "god node hub critical" 2>&1
    return $result
}

# ============================================================
# Scan entire codebase and return summary
# ============================================================
function Scan-Codebase {
    Write-Host "Scanning codebase with graphify..." -ForegroundColor Cyan

    $summary = python $RagScript query "summary architecture" 2>&1
    return $summary
}

# Execute
switch ($Action) {
    "context" {
        if (-not $Query) {
            Write-Host "Usage: graphify-integration.ps1 -Action context -Query 'softmax reduction'" -ForegroundColor Yellow
            break
        }
        $result = Get-Context -Query $Query -Depth $Depth -MaxNodes $MaxNodes
        Write-Host $result
    }

    "deps" {
        if (-not $Query) {
            Write-Host "Usage: graphify-integration.ps1 -Action deps -Query 'attention'" -ForegroundColor Yellow
            break
        }
        $result = Get-Deps -Query $Query
        Write-Host $result
    }

    "validate" {
        if ($Files.Count -eq 0) {
            Write-Host "Usage: graphify-integration.ps1 -Action validate -Files @('src/kernels/gemm.comp')" -ForegroundColor Yellow
            break
        }
        $issues = Validate-Changes -ModifiedFiles $Files
        if ($issues.Count -eq 0) {
            Write-Host "No issues found" -ForegroundColor Green
        } else {
            $issues | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
        }
    }

    "enrich" {
        if (-not $Query) {
            Write-Host "Usage: graphify-integration.ps1 -Action enrich -Query 'fix flash attention tiling'" -ForegroundColor Yellow
            break
        }
        $result = Enrich-Prompt -TaskTitle $Query -TaskDescription $Query -FilesToRead $Files -FilesToWrite $Files
        Write-Host $result
    }

    "find-related" {
        if ($Files.Count -eq 0) {
            Write-Host "Usage: graphify-integration.ps1 -Action find-related -Files @('src/kernels/gemm.comp')" -ForegroundColor Yellow
            break
        }
        $related = Find-Related -Files $Files -Depth $Depth
        $related.Keys | ForEach-Object {
            Write-Host "`n$_:" -ForegroundColor Cyan
            Write-Host $related[$_] -ForegroundColor Gray
        }
    }

    "community" {
        $result = Get-CommunitySummary
        Write-Host $result
    }

    "god-nodes" {
        $result = Get-GodNodes
        Write-Host $result
    }
}
