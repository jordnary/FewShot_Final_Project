# Stage 6 改进实验运行指南

本文档面向需要复现第 6 阶段实验的人员，默认从项目根目录 `FewShot_Final_Project/` 开始执行命令。

## 运行前检查

确认项目结构完整：

```text
FewShot_Final_Project/
├─ 01_environment/requirements.txt
├─ 03_datasets/CUB_200_2011/
└─ 06_improvement/
```

确认 CUB 数据目录至少包含：

```text
03_datasets/CUB_200_2011/
├─ images/
├─ train.csv
├─ val.csv
└─ test.csv
```

安装依赖：

```bash
pip install -r 01_environment/requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## CLIP 权重设置

云端脚本默认使用 `open_clip` 的 `ViT-B-16` + `openai` 权重，并设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果服务器不能自动下载权重，可以先本地下载并上传到：

```text
06_improvement/pretrained/open_clip_model.safetensors
```

然后运行：

```bash
export CLIP_PRETRAINED=06_improvement/pretrained/open_clip_model.safetensors
```

如果 CLIP 权重不可用，`extract_clip_features.py` 还提供 `torchvision_vit` 兜底 backend。该兜底不是 CLIP，但仍可用于验证 frozen ViT feature pipeline。

## 主路线：Patch-Token MAP

推荐优先运行 patch-token MAP 路线，因为它最接近 FroFA 原论文强调的 patch-token feature space augmentation。

```bash
bash 06_improvement/run_clip_frofa_cub_cloud.sh
```

脚本会执行两步：

```text
1. extract_clip_patch_tokens.py
2. run_frofa_map_eval.py
```

默认关键设置：

```text
CLIP model: ViT-B-16
split: test
episodes: 600
way: 5
shots: 1 5
query: 15
MAP train steps: 80
MAP heads: 8
FroFA alpha: 0.20
FroFA num_aug: 8
augmentation: brightness
paired episodes: enabled
```

主要输出：

```text
06_improvement/logs/extract_clip_vit_b16_patch_tokens.log
06_improvement/logs/clip_vit_b16_patch_frofa_map_eval.log
06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz
06_improvement/results/clip_vit_b16_patch_frofa_map_cub.csv
```

当前报告采用的 paired 正式结果保存在：

```text
06_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired.csv
```

## 更接近原论文：Post-LN Tokens + Val Sweep

如果要进一步复现 FroFA 原论文的实验逻辑，推荐运行 paperlike 路线：

```bash
export CLIP_PRETRAINED=06_improvement/pretrained/open_clip_model.safetensors
bash 06_improvement/run_patch_frofa_paperlike_cloud.sh
```

这条路线会执行：

```text
1. 提取 val/test 的 CLIP ViT-B/16 post-LN patch tokens
2. 在 val split 上搜索 alpha、num_aug 和增强组合
3. 用 val 上选出的最佳配置在 test split 上汇报
```

与主路线相比，它更接近原论文的地方是：

- 使用 projection 之前的 patch-token feature grid，而不是 projected CLIP embedding tokens；
- 使用 validation split 做超参选择，避免直接根据 test 调参；
- 增加 `brightness+posterize`、`brightness+contrast` 等组合增强。

主要输出：

```text
06_improvement/results/features/cub_val_clip_vit_b16_postln_patch_tokens.npz
06_improvement/results/features/cub_test_clip_vit_b16_postln_patch_tokens.npz
06_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep.csv
06_improvement/logs/clip_vit_b16_postln_patch_frofa_map_sweep.log
```

云端 paperlike 脚本默认带 `--no-md`，只保存 `.csv` 和 `.log`。

当前 paperlike 结果中，validation sweep 选出的最佳配置是：

```text
alpha: 0.30
num_aug: 8
augmentation: brightness
train_steps: 40
weight_decay: 0.01
```

对应 test split 结果：

| Method | 5-way 1-shot | 5-way 5-shot |
| :--- | ---: | ---: |
| MAP | 42.187 +/- 0.816 | 68.878 +/- 0.884 |
| FroFA + MAP | 43.960 +/- 0.854 | 72.893 +/- 0.833 |
| Gain | +1.773 | +4.016 |

## Global Feature Linear Probe

如果要复现 global CLIP feature + linear probe 消融，先提取 global features：

```bash
python 06_improvement/extract_clip_features.py \
  --data-root 03_datasets/CUB_200_2011 \
  --output-dir 06_improvement/results/features \
  --output-prefix clip_vit_b16 \
  --model-name ViT-B-16 \
  --pretrained "${CLIP_PRETRAINED:-openai}" \
  --backend open_clip \
  --splits train val test \
  --batch-size 64 \
  --workers 8
```

再运行 linear probe：

```bash
python 06_improvement/run_frofa_linear_eval.py \
  --feature-file 06_improvement/results/features/cub_test_clip_vit_b16.npz \
  --output-prefix 06_improvement/results/clip_vit_b16_frofa_linear_cub \
  --episodes 1000 \
  --way 5 \
  --shots 1 5 \
  --query 15 \
  --ridge-lambda 1.0 \
  --alpha 0.20 \
  --num-aug 8 \
  --augmentations brightness
```

输出：

```text
06_improvement/results/clip_vit_b16_frofa_linear_cub.csv
```

## 手动运行 Patch-Token MAP

如果 patch tokens 已经存在，可以直接运行：

```bash
python 06_improvement/run_frofa_map_eval.py \
  --token-file 06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz \
  --output-prefix 06_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired \
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
  --augmentations brightness
```

输出：

```text
06_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired.csv
```

## 结果文件

当前需要保留的结果 CSV：

```text
06_improvement/results/clip_vit_b16_frofa_linear_cub.csv
06_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired.csv
06_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep.csv
06_improvement/results/final_stage6_summary.csv
```

`results/features/` 下的 `.npz` 是可复用特征缓存，文件较大，通常作为本地或云端运行产物保留。

## 后台运行建议

云服务器上建议使用 `tmux`：

```bash
tmux new -s stage6-improvement
bash 06_improvement/run_clip_frofa_cub_cloud.sh
```

训练中断开终端但保留任务：按 `Ctrl+B`，再按 `D`。

恢复会话：

```bash
tmux attach -t stage6-improvement
```

## 注意事项

重新运行脚本会覆盖同名 CSV 和日志。如果需要保留旧结果，请先备份对应文件或换一个 `--output-prefix`。

CPU 可以运行部分流程，但 CLIP 特征提取和 600 episode MAP evaluation 会很慢，正式复现建议使用 GPU。
