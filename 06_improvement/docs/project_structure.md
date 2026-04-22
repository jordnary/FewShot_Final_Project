# 06_improvement 项目结构

本目录用于完成第 6 阶段个人改进实验：将 FroFA 的实验假设从“训练 CUB episodic backbone”进一步对齐到“强预训练 frozen feature + 轻量分类头”，重点比较 CLIP ViT-B/16 frozen features、global feature FroFA、patch-token FroFA 和 MAP head。

## 目录概览

```text
06_improvement/
├─ docs/
│  ├─ project_structure.md
│  ├─ run_guide.md
│  └─ experiment_summary.md
├─ logs/
│  ├─ extract_clip_vit_b16_features.log
│  ├─ extract_clip_vit_b16_patch_tokens.log
│  └─ clip_vit_b16_patch_frofa_map_paired.log
├─ pretrained/
├─ results/
│  ├─ features/
│  ├─ clip_vit_b16_frofa_linear_cub.csv
│  ├─ clip_vit_b16_patch_frofa_map_cub_paired.csv
│  └─ final_stage6_summary.csv
├─ extract_clip_features.py
├─ extract_clip_patch_tokens.py
├─ run_frofa_linear_eval.py
├─ run_frofa_map_eval.py
├─ run_frofa_map_sweep.py
├─ run_clip_frofa_cub_cloud.sh
└─ run_patch_frofa_paperlike_cloud.sh
```

## 文件职责

`docs/` 保存本阶段整理后的说明文档。`project_structure.md` 介绍目录结构，`run_guide.md` 面向他人说明如何复现实验，`experiment_summary.md` 归纳实验过程、数据和结论。

`logs/` 保存云端运行日志。特征提取日志记录缓存文件形状；MAP 评测日志记录每个方法在 600 个 episode 上的 running mean 和最终精度。

`pretrained/` 用于放置本地下载后上传的预训练权重，例如 `open_clip_model.safetensors`。该目录下的大模型权重通常属于本地资源，不建议纳入 Git。

`results/features/` 保存 frozen feature 缓存，包括 global CLIP feature 和 patch token feature。当前已有：

```text
cub_train_clip_vit_b16.npz
cub_val_clip_vit_b16.npz
cub_test_clip_vit_b16.npz
cub_test_clip_vit_b16_patch_tokens.npz
cub_val_clip_vit_b16_postln_patch_tokens.npz
cub_test_clip_vit_b16_postln_patch_tokens.npz
metadata_clip_vit_b16.json
metadata_clip_vit_b16_patch_tokens.json
metadata_clip_vit_b16_postln_patch_tokens.json
```

`results/*.csv` 保存可追踪的实验结果。当前核心结果包括 global feature linear probe、projected patch-token MAP paired evaluation、post-LN paperlike validation sweep 和最终 Stage 6 汇总。

`extract_clip_features.py` 读取 CUB split CSV 和图像，使用 frozen CLIP ViT-B/16 提取 pooled global image features，输出 `.npz` 缓存。

`run_frofa_linear_eval.py` 在 global feature 缓存上采样 5-way episode，对比 no-FroFA 与 FroFA 的 closed-form L2 linear probe。

`extract_clip_patch_tokens.py` 使用 frozen CLIP ViT-B/16 提取最终层 patch tokens。默认输出 projected `N x 196 x 512` token 缓存；使用 `--token-stage post_ln` 时输出 projection 之前的 `N x 196 x 768` token 缓存，更接近 FroFA 原论文的 frozen ViT feature grid。

`run_frofa_map_eval.py` 在 patch token 缓存上按 episode 训练 MAP head，对比 MAP 和 FroFA + MAP。默认使用 paired episodes，让同一 shot 下两种方法面对相同 support/query episode。

`run_clip_frofa_cub_cloud.sh` 是云端主入口，默认执行 patch-token 路线：提取 test split patch tokens，然后运行 MAP 与 FroFA + MAP 的 5-way 1-shot / 5-shot 评测。

`run_frofa_map_sweep.py` 是更接近论文流程的 validation sweep：在 val token cache 上搜索 FroFA/MAP 超参，再可选地用最佳配置评测 test token cache。

`run_patch_frofa_paperlike_cloud.sh` 是 paperlike 云端入口：提取 `post_ln` patch tokens，运行 validation sweep，并在 test split 汇报最佳验证配置。

## 实验路线

| 路线 | 脚本 | 输出 | 作用 |
| :--- | :--- | :--- | :--- |
| Global CLIP feature | `extract_clip_features.py` | `results/features/cub_*_clip_vit_b16.npz` | 提取 pooled global feature |
| Global linear probe | `run_frofa_linear_eval.py` | `results/clip_vit_b16_frofa_linear_cub.csv` | 比较 no-FroFA 与 FroFA vector feature |
| Patch-token cache | `extract_clip_patch_tokens.py` | `results/features/cub_test_clip_vit_b16_patch_tokens.npz` | 提取 ViT patch tokens |
| Patch-token MAP | `run_frofa_map_eval.py` | `results/clip_vit_b16_patch_frofa_map_cub_paired.csv` | 比较 MAP 与 FroFA + MAP |
| Paperlike sweep | `run_frofa_map_sweep.py` | `results/clip_vit_b16_postln_patch_frofa_map_sweep.csv` | post-LN tokens 上 val sweep 后 test 汇报 |
| Final summary | 手工整理 CSV | `results/final_stage6_summary.csv` | 汇总最终报告结论 |

## 数据流

1. 准备 `03_datasets/CUB_200_2011` 数据目录。
2. 使用 frozen CLIP ViT-B/16 提取 global features 或 patch tokens。
3. 在缓存特征上采样 5-way 1-shot / 5-shot episode。
4. 对 global features 运行 closed-form L2 linear probe。
5. 对 patch tokens 运行 episode-trained MAP head。
6. 在 paperlike 路线中，先用 validation split 选择 FroFA 超参，再在 test split 上汇报。
7. 对比 no-FroFA / FroFA，或 MAP / FroFA + MAP。
8. 将核心结果汇总到 `results/final_stage6_summary.csv`，并在 `docs/experiment_summary.md` 中解释。
