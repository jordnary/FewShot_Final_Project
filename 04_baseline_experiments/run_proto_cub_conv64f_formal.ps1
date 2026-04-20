$ErrorActionPreference = "Continue"

(& "D:\Software\miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate ml
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$workspace = "D:\Codes\Python\Machine_Learning\FewShot_Final_Project"
$runner = Join-Path $workspace "04_baseline_experiments\run_libfewshot_config.py"
$logs = Join-Path $workspace "04_baseline_experiments\logs"

New-Item -ItemType Directory -Force -Path $logs | Out-Null

$configs = @(
    "04_baseline_experiments\configs\proto_cub_conv64f_formal_5way_1shot.yaml",
    "04_baseline_experiments\configs\proto_cub_conv64f_formal_5way_5shot.yaml"
)

$names = @(
    "proto_cub_conv64f_formal_5way_1shot",
    "proto_cub_conv64f_formal_5way_5shot"
)

for ($i = 0; $i -lt $configs.Count; $i++) {
    $config = Join-Path $workspace $configs[$i]
    $log = Join-Path $logs ($names[$i] + "_console.log")
    python $runner $config *> $log
}
