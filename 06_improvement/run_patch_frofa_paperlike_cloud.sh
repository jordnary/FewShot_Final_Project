#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS=8
fi
if [[ ! "${OPENBLAS_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OPENBLAS_NUM_THREADS=8
fi
if [[ ! "${NUMEXPR_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export NUMEXPR_NUM_THREADS=8
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p 06_improvement/logs 06_improvement/results/features

python 06_improvement/extract_clip_patch_tokens.py \
  --data-root 03_datasets/CUB_200_2011 \
  --output-dir 06_improvement/results/features \
  --output-prefix clip_vit_b16_postln_patch_tokens \
  --model-name ViT-B-16 \
  --pretrained "${CLIP_PRETRAINED:-openai}" \
  --splits val test \
  --batch-size 24 \
  --workers 8 \
  --feature-dtype float16 \
  --token-stage post_ln \
  2>&1 | tee 06_improvement/logs/extract_clip_vit_b16_postln_patch_tokens.log

PYTHONUNBUFFERED=1 python -u 06_improvement/run_frofa_map_sweep.py \
  --val-token-file 06_improvement/results/features/cub_val_clip_vit_b16_postln_patch_tokens.npz \
  --test-token-file 06_improvement/results/features/cub_test_clip_vit_b16_postln_patch_tokens.npz \
  --output-prefix 06_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep \
  --episodes 120 \
  --test-episodes 600 \
  --shots 1 5 \
  --alphas 0.10 0.20 0.30 \
  --num-augs 4 8 \
  --augmentation-sets brightness brightness+posterize brightness+contrast \
  --train-steps-grid 40 \
  --weight-decays 0.01 \
  --select-by mean_gain \
  --log-interval 20 \
  --no-md \
  2>&1 | tee 06_improvement/logs/clip_vit_b16_postln_patch_frofa_map_sweep.log
