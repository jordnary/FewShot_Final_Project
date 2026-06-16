#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/artifacts/logs/baseline"
mkdir -p "$PROJECT_ROOT/artifacts/runs/baseline"
mkdir -p "$PROJECT_ROOT/experiments/baseline/results"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
  export MKL_NUM_THREADS=8
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
PY

run_one() {
  local config="$1"
  local log_name="$2"
  echo "============================================================"
  echo "Running $log_name"
  echo "Config: $config"
  echo "============================================================"
  python "$PROJECT_ROOT/scripts/baseline/run_libfewshot_config_cloud.py" \
    "$PROJECT_ROOT/configs/baseline/$config" \
    2>&1 | tee "$PROJECT_ROOT/artifacts/logs/baseline/${log_name}_console.log"
}

MODE="${1:-resnet12}"

if [[ "$MODE" != "resnet12" ]]; then
  echo "Usage: bash scripts/baseline/run_proto_cub_boost_cloud.sh [resnet12]" >&2
  exit 2
fi

run_one "proto_cub_resnet12_boost_5way_1shot_cloud.yaml" "proto_cub_resnet12_boost_5way_1shot_cloud"
run_one "proto_cub_resnet12_boost_5way_5shot_cloud.yaml" "proto_cub_resnet12_boost_5way_5shot_cloud"

python "$PROJECT_ROOT/scripts/baseline/summarize_boost_results.py"
