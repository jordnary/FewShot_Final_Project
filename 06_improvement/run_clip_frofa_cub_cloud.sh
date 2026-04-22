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
  --output-prefix clip_vit_b16_patch_tokens \
  --model-name ViT-B-16 \
  --pretrained "${CLIP_PRETRAINED:-openai}" \
  --splits test \
  --batch-size 32 \
  --workers 8 \
  --feature-dtype float16 \
  2>&1 | tee 06_improvement/logs/extract_clip_vit_b16_patch_tokens.log

python 06_improvement/run_frofa_map_eval.py \
  --token-file 06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz \
  --output-prefix 06_improvement/results/clip_vit_b16_patch_frofa_map_cub \
  --episodes 600 \
  --way 5 \
  --shots 1 5 \
  --query 15 \
  --train-steps 80 \
  --lr 0.001 \
  --weight-decay 0.01 \
  --map-heads 8 \
  --map-queries 1 \
  --alpha 0.20 \
  --num-aug 8 \
  --augmentations brightness \
  2>&1 | tee 06_improvement/logs/clip_vit_b16_patch_frofa_map_eval.log
