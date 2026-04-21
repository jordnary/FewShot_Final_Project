#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/04_baseline_experiments/logs"
mkdir -p "$PROJECT_ROOT/04_baseline_experiments/results"

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
  python "$PROJECT_ROOT/04_baseline_experiments/run_libfewshot_config_cloud.py" \
    "$PROJECT_ROOT/04_baseline_experiments/configs/$config" \
    2>&1 | tee "$PROJECT_ROOT/04_baseline_experiments/logs/${log_name}_console.log"
}

MODE="${1:-resnet12}"

case "$MODE" in
  resnet12)
    run_one "proto_cub_resnet12_boost_5way_1shot_cloud.yaml" "proto_cub_resnet12_boost_5way_1shot_cloud"
    run_one "proto_cub_resnet12_boost_5way_5shot_cloud.yaml" "proto_cub_resnet12_boost_5way_5shot_cloud"
    ;;
  conv64f)
    run_one "proto_cub_conv64f_boost_5way_1shot_cloud.yaml" "proto_cub_conv64f_boost_5way_1shot_cloud"
    run_one "proto_cub_conv64f_boost_5way_5shot_cloud.yaml" "proto_cub_conv64f_boost_5way_5shot_cloud"
    ;;
  all)
    run_one "proto_cub_resnet12_boost_5way_1shot_cloud.yaml" "proto_cub_resnet12_boost_5way_1shot_cloud"
    run_one "proto_cub_resnet12_boost_5way_5shot_cloud.yaml" "proto_cub_resnet12_boost_5way_5shot_cloud"
    run_one "proto_cub_conv64f_boost_5way_1shot_cloud.yaml" "proto_cub_conv64f_boost_5way_1shot_cloud"
    run_one "proto_cub_conv64f_boost_5way_5shot_cloud.yaml" "proto_cub_conv64f_boost_5way_5shot_cloud"
    ;;
  *)
    echo "Usage: bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh [resnet12|conv64f|all]" >&2
    exit 2
    ;;
esac

python "$PROJECT_ROOT/04_baseline_experiments/summarize_boost_results.py"
