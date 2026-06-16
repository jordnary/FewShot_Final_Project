#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/artifacts/logs/frofa"
mkdir -p "$PROJECT_ROOT/artifacts/runs/frofa"
mkdir -p "$PROJECT_ROOT/experiments/frofa_reproduction/results"

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
    "$PROJECT_ROOT/configs/frofa/$config" \
    2>&1 | tee "$PROJECT_ROOT/artifacts/logs/frofa/${log_name}_console.log"
}

MODE="${1:-joint}"

case "$MODE" in
  joint)
    run_one "frofa_proto_cub_resnet12_5way_1shot_cloud.yaml" "frofa_proto_cub_resnet12_5way_1shot_cloud"
    run_one "frofa_proto_cub_resnet12_5way_5shot_cloud.yaml" "frofa_proto_cub_resnet12_5way_5shot_cloud"
    ;;
  frozen)
    run_one "frofa_proto_cub_resnet12_frozen_5way_1shot_cloud.yaml" "frofa_proto_cub_resnet12_frozen_5way_1shot_cloud"
    run_one "frofa_proto_cub_resnet12_frozen_5way_5shot_cloud.yaml" "frofa_proto_cub_resnet12_frozen_5way_5shot_cloud"
    ;;
  all)
    run_one "frofa_proto_cub_resnet12_5way_1shot_cloud.yaml" "frofa_proto_cub_resnet12_5way_1shot_cloud"
    run_one "frofa_proto_cub_resnet12_5way_5shot_cloud.yaml" "frofa_proto_cub_resnet12_5way_5shot_cloud"
    run_one "frofa_proto_cub_resnet12_frozen_5way_1shot_cloud.yaml" "frofa_proto_cub_resnet12_frozen_5way_1shot_cloud"
    run_one "frofa_proto_cub_resnet12_frozen_5way_5shot_cloud.yaml" "frofa_proto_cub_resnet12_frozen_5way_5shot_cloud"
    ;;
  *)
    echo "Usage: bash scripts/frofa/run_frofa_cub_cloud.sh [joint|frozen|all]" >&2
    exit 2
    ;;
esac

python "$PROJECT_ROOT/scripts/frofa/summarize_frofa_results.py"
