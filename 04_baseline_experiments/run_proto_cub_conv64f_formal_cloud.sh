#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/04_baseline_experiments/logs"
mkdir -p "$PROJECT_ROOT/04_baseline_experiments/results"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

python "$PROJECT_ROOT/04_baseline_experiments/run_libfewshot_config_cloud.py" \
  "$PROJECT_ROOT/04_baseline_experiments/configs/proto_cub_conv64f_formal_5way_1shot_cloud.yaml" \
  2>&1 | tee "$PROJECT_ROOT/04_baseline_experiments/logs/proto_cub_conv64f_formal_5way_1shot_cloud_console.log"

python "$PROJECT_ROOT/04_baseline_experiments/run_libfewshot_config_cloud.py" \
  "$PROJECT_ROOT/04_baseline_experiments/configs/proto_cub_conv64f_formal_5way_5shot_cloud.yaml" \
  2>&1 | tee "$PROJECT_ROOT/04_baseline_experiments/logs/proto_cub_conv64f_formal_5way_5shot_cloud_console.log"
