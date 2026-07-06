param(
    [switch]$ShowResults,
    [int]$HelpPreviewLines = 14
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Get-ConsoleWidth {
    try {
        return [Math]::Min([Math]::Max([Console]::WindowWidth, 72), 110)
    }
    catch {
        return 88
    }
}

function Write-Rule {
    param(
        [string]$Title,
        [ConsoleColor]$Color = [ConsoleColor]::DarkCyan
    )

    $width = Get-ConsoleWidth
    Write-Host ""
    Write-Host ("=" * $width) -ForegroundColor $Color
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * $width) -ForegroundColor $Color
}

function Write-Status {
    param(
        [string]$Label,
        [string]$Value,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )

    Write-Host ("{0,-18} " -f $Label) -NoNewline -ForegroundColor DarkGray
    Write-Host $Value -ForegroundColor $Color
}

function ConvertTo-ProjectPath {
    param([string]$Path)

    $fullPath = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    }
    else {
        Join-Path $ProjectRoot $Path
    }

    try {
        $fullPath = (Resolve-Path $fullPath).Path
    }
    catch {
        $fullPath = [System.IO.Path]::GetFullPath($fullPath)
    }

    if ($fullPath.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $fullPath.Substring($ProjectRoot.Length).TrimStart(
            [char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        )
    }

    return $fullPath
}

function Invoke-HelpPreview {
    param([string]$ScriptPath)

    Write-Host ""
    Write-Host ("python {0} --help" -f $ScriptPath) -ForegroundColor White

    $output = & python $ScriptPath --help 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Status "status" ("failed with exit code {0}" -f $exitCode) Red
        $output | ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor DarkRed }
        throw "Entry point check failed: $ScriptPath"
    }

    Write-Status "status" "ready" Green

    if ($HelpPreviewLines -gt 0) {
        $preview = @($output | Select-Object -First $HelpPreviewLines)
        foreach ($line in $preview) {
            Write-Host ("  {0}" -f $line) -ForegroundColor DarkGray
        }

        if (@($output).Count -gt $preview.Count) {
            Write-Host ("  ... {0} more lines hidden" -f (@($output).Count - $preview.Count)) -ForegroundColor DarkGray
        }
    }
}

Write-Rule "Few-Shot FroFA Demo Check"
Write-Status "project root" $ProjectRoot Gray
Write-Status "timestamp" (Get-Date -Format "yyyy-MM-dd HH:mm:ss") Gray

$pythonVersion = (& python --version 2>&1)
if ($LASTEXITCODE -eq 0) {
    Write-Status "python" ($pythonVersion -join " ") Green
}
else {
    Write-Status "python" "not available" Red
    throw "Python is required for the demo check."
}

Write-Rule "Project Files"

$requiredPaths = @(
    @{ Item = "README"; Path = "README.md" },
    @{ Item = "Runbook"; Path = "docs/runbook.md" },
    @{ Item = "Requirements"; Path = "environment/requirements.txt" },
    @{ Item = "Baseline entry"; Path = "scripts/baseline/run_libfewshot_config.py" },
    @{ Item = "Cloud entry"; Path = "scripts/baseline/run_libfewshot_config_cloud.py" },
    @{ Item = "Final report"; Path = "report/final_report/report.md" }
)

$requiredPaths |
    ForEach-Object {
        [PSCustomObject]@{
            Item = $_.Item
            Status = if (Test-Path $_.Path) { "ready" } else { "missing" }
            Path = $_.Path
        }
    } |
    Format-Table -AutoSize |
    Out-Host

Write-Rule "Python Entry Points"

Invoke-HelpPreview "scripts/baseline/run_libfewshot_config.py"
Invoke-HelpPreview "scripts/baseline/run_libfewshot_config_cloud.py"

Write-Rule "Experiment Results"

$experimentRows = Get-ChildItem -Path experiments -Directory |
    Sort-Object Name |
    ForEach-Object {
        $resultPath = Join-Path $_.FullName "results"
        $csvCount = if (Test-Path $resultPath) {
            @(Get-ChildItem -Path $resultPath -Filter *.csv -File).Count
        }
        else {
            0
        }

        [PSCustomObject]@{
            Experiment = $_.Name
            Guide = if (Test-Path (Join-Path $_.FullName "run_guide.md")) { "ready" } else { "missing" }
            CSVs = $csvCount
            Results = ConvertTo-ProjectPath $resultPath
        }
    }

$experimentRows | Format-Table -AutoSize | Out-Host

if ($ShowResults) {
    Write-Rule "Result CSV Files"

    Get-ChildItem -Path experiments -Recurse -Filter *.csv -File |
        Sort-Object FullName |
        ForEach-Object {
            $relativePath = ConvertTo-ProjectPath $_.FullName
            $parts = $relativePath -split "[\\/]"

            [PSCustomObject]@{
                Experiment = if ($parts.Count -gt 1) { $parts[1] } else { "-" }
                File = $_.Name
                SizeKB = [Math]::Round($_.Length / 1KB, 1)
            }
        } |
        Format-Table -AutoSize |
        Out-Host
}

Write-Rule "Done"
Write-Status "next step" ".\demo\demo_run.ps1 -ShowResults" Yellow
