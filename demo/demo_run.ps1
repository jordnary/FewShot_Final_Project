param(
    [switch]$ShowResults
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Project root: $ProjectRoot"
Write-Host "Checking lightweight Python entry points..."

python scripts/baseline/run_libfewshot_config.py --help | Out-Host
python scripts/baseline/run_libfewshot_config_cloud.py --help | Out-Host

Write-Host "Core result directories:"
Get-ChildItem -Path experiments -Directory | Select-Object FullName | Format-Table -AutoSize

if ($ShowResults) {
    Write-Host "Result CSV files:"
    Get-ChildItem -Path experiments -Recurse -Filter *.csv | Select-Object FullName | Format-Table -AutoSize
}
