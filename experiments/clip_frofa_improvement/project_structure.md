# CLIP-FroFA 改进实验结构

本实验线将 FroFA 的实验假设从“训练 CUB episodic backbone”进一步对齐到“强预训练 frozen feature + 轻量分类头”，重点比较 CLIP ViT-B/16 frozen features、global feature FroFA、patch-token FroFA 和 MAP head。

## 相关目录

```text
scripts/clip_frofa/                         # CLIP 特征提取、MAP 评估和 sweep 入口
experiments/clip_frofa_improvement/         # 实验说明与可提交 CSV 结果
artifacts/features/clip_frofa/              # CLIP feature cache，不纳入 Git
artifacts/logs/clip_frofa/                  # 控制台日志，不纳入 Git
artifacts/pretrained/                       # 本地预训练权重，不纳入 Git
```

## 文件职责

`scripts/clip_frofa/extract_clip_features.py` 读取 CUB split CSV 和图像，使用 frozen CLIP ViT-B/16 提取 pooled global image features，输出 `.npz` 缓存。

`scripts/clip_frofa/extract_clip_patch_tokens.py` 使用 frozen CLIP ViT-B/16 提取最终层 patch tokens。默认输出 projected `N x 196 x 512` token 缓存；使用 `--token-stage post_ln` 时输出 projection 之前的 `N x 196 x 768` token 缓存，更接近 FroFA 原论文的 frozen ViT feature grid。

`scripts/clip_frofa/run_frofa_linear_eval.py` 在 global feature 缓存上采样 5-way episode，对比 no-FroFA 与 FroFA 的 closed-form L2 linear probe。

`scripts/clip_frofa/run_frofa_map_eval.py` 在 patch token 缓存上按 episode 训练 MAP head，对比 MAP 和 FroFA + MAP。默认使用 paired episodes，让同一 shot 下两种方法面对相同 support/query episode。

`scripts/clip_frofa/run_frofa_map_sweep.py` 在 validation token cache 上搜索 FroFA/MAP 超参，再可选地用最佳配置评测 test token cache。

`experiments/clip_frofa_improvement/results/*.csv` 保存可追踪的实验结果。特征缓存、日志和预训练权重保存在 `artifacts/`。

## 实验路线

| 路线 | 脚本 | 输出 | 作用 |
| :--- | :--- | :--- | :--- |
| Global CLIP feature | `extract_clip_features.py` | `artifacts/features/clip_frofa/cub_*_clip_vit_b16.npz` | 提取 pooled global feature |
| Global linear probe | `run_frofa_linear_eval.py` | `experiments/clip_frofa_improvement/results/clip_vit_b16_frofa_linear_cub.csv` | 比较 no-FroFA 与 FroFA vector feature |
| Patch-token cache | `extract_clip_patch_tokens.py` | `artifacts/features/clip_frofa/cub_test_clip_vit_b16_patch_tokens.npz` | 提取 ViT patch tokens |
| Patch-token MAP | `run_frofa_map_eval.py` | `experiments/clip_frofa_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired.csv` | 比较 MAP 与 FroFA + MAP |
| Paperlike sweep | `run_frofa_map_sweep.py` | `experiments/clip_frofa_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep.csv` | post-LN tokens 上 val sweep 后 test 汇报 |
| Final summary | 手工整理 CSV | `experiments/clip_frofa_improvement/results/final_summary.csv` | 汇总最终报告结论 |

## 数据流

1. 准备 `data/CUB_200_2011` 数据目录。
2. 使用 frozen CLIP ViT-B/16 提取 global features 或 patch tokens。
3. 在缓存特征上采样 5-way 1-shot / 5-shot episode。
4. 对 global features 运行 closed-form L2 linear probe。
5. 对 patch tokens 运行 episode-trained MAP head。
6. 在 paperlike 路线中，先用 validation split 选择 FroFA 超参，再在 test split 上汇报。
7. 将核心结果汇总到 `experiments/clip_frofa_improvement/results/final_summary.csv`。
