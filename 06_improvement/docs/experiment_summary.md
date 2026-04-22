# Stage 6 改进实验过程与结论

本阶段目标是进一步对齐 FroFA 原论文的核心假设：使用强预训练 frozen representation，在少样本阶段只训练轻量分类头，并在 frozen feature space 中做 FroFA-style augmentation。

## 改进动机

前 4-5 阶段已经完成 CUB 上的 ProtoNet baseline 和 FroFA-style ProtoNet 接入，但它们主要仍在 Conv64F 或 CUB episodic ResNet12 backbone 上训练。这样的设置更像“在传统 few-shot backbone 上增加特征扰动”，没有充分体现 FroFA 强调的 frozen feature setting。

第 6 阶段因此切换到 CLIP ViT-B/16 frozen encoder。这样可以将问题拆成两部分：

```text
CUB images
-> frozen CLIP ViT-B/16 features
-> lightweight few-shot head
-> no-FroFA / FroFA comparison
```

## 实验过程

第一步提取 CLIP global image features，运行 closed-form L2 linear probe，对比 no-FroFA 和 FroFA vector-feature 版本。

第二步发现 pooled global embedding 上的 FroFA 没有带来提升，说明直接扰动 512 维全局向量并不符合 FroFA 原论文的 patch-token 假设。

第三步改为提取 CLIP ViT-B/16 最后一层 patch tokens，得到 `N x 196 x 512` 的 token grid。

第四步在 patch tokens 上进行 paired episode 评测。每个 episode 训练一个 MAP head，并比较不增强的 MAP 与 support patch-token FroFA + MAP。

第五步将 global feature 结果、patch-token paired 结果和最终解释汇总到 `results/final_stage6_summary.csv`。

## 实验结果

| Experiment | Method | 5-way 1-shot | 5-way 5-shot | Main conclusion |
| :--- | :--- | ---: | ---: | :--- |
| Global CLIP feature | no-FroFA linear probe | 86.975 +/- 0.569 | 96.379 +/- 0.248 | Strong frozen feature baseline |
| Global CLIP feature | FroFA linear probe | 85.717 +/- 0.574 | 95.733 +/- 0.295 | Pooled embedding FroFA hurts |
| Patch-token paired | MAP | 46.213 +/- 0.875 | 75.478 +/- 0.765 | Patch-token MAP baseline |
| Patch-token paired | FroFA + MAP | 45.402 +/- 0.890 | 77.056 +/- 0.819 | FroFA improves 5-shot by +1.578 |

数据来源：

```text
06_improvement/results/clip_vit_b16_frofa_linear_cub.csv
06_improvement/results/clip_vit_b16_patch_frofa_map_cub_paired.csv
06_improvement/results/final_stage6_summary.csv
```

## 关键分析

CLIP frozen global feature + no-FroFA linear probe 已经明显超过前面阶段的 task-trained ProtoNet-ResNet12 baseline。与 04 阶段 ResNet12 baseline 的 73.376 / 85.945 相比，global CLIP no-FroFA 达到 86.975 / 96.379，分别提升 13.599 和 10.434 个百分点。

但是，在 pooled global feature 上加入 FroFA 后，1-shot 从 86.975 降到 85.717，5-shot 从 96.379 降到 95.733。这说明强 frozen feature 本身有效，但把 brightness FroFA 直接作用在压缩后的全局向量上并不合适。

Patch-token 路线更接近 FroFA 原始假设。paired 正式实验中，FroFA + MAP 在 5-shot 上从 75.478 提升到 77.056，提升 1.578 个百分点；但在 1-shot 上从 46.213 降到 45.402，下降 0.811 个百分点。

这个结果说明 patch-token FroFA 比 global-vector FroFA 更合理，尤其在 5-shot 中能改善 MAP head；但 1-shot 下 support 样本过少，增强强度、MAP 训练步数、weight decay 或 episode 采样仍需继续调参。

## 最终结论

第 6 阶段最明确的改进结论是：使用 CLIP ViT-B/16 frozen features 能显著改善 CUB few-shot 表现，说明强预训练 representation 比继续训练小型 episodic backbone 更有价值。

FroFA 的有效载体更可能是 ViT patch-token grid，而不是 pooled global embedding。当前 patch-token FroFA + MAP 已经在 5-shot 上体现正向收益，但 1-shot 仍不稳定。

后续如果继续推进，可以优先尝试：

1. 调整 `alpha`、`num_aug` 和 augmentation 类型。
2. 增加 MAP head train steps 或改进正则化。
3. 使用 validation split 做系统调参。
4. 继续保持 paired episode 评测，减少 episode 难度差异带来的噪声。

## 进一步贴近原论文的新增路线

为进一步靠近 FroFA 原论文，当前代码新增了 paperlike 路线：

```text
06_improvement/run_patch_frofa_paperlike_cloud.sh
06_improvement/run_frofa_map_sweep.py
```

它相比已有 paired patch-token 实验有三点改进：

1. 使用 `post_ln` patch tokens，即 CLIP ViT 最后一层、projection 之前的 patch features，形状约为 `N x 196 x 768`。这比 projected 512 维 CLIP embedding tokens 更接近论文中的 frozen ViT feature grid。
2. 使用 validation split 选择超参。先在 val 上搜索 `alpha`、`num_aug` 和增强组合，再用最佳配置在 test split 汇报，避免直接根据 test 结果调参。
3. 增加组合增强：`brightness`、`brightness+posterize`、`brightness+contrast`。这对应论文中 sequential FroFA 和多增强组合的思路。

推荐运行：

```bash
export CLIP_PRETRAINED=06_improvement/pretrained/open_clip_model.safetensors
bash 06_improvement/run_patch_frofa_paperlike_cloud.sh
```

如果这条路线在 test 上继续提升，可以作为最终报告的“更接近原论文复现”结果；如果提升不稳定，则可解释为本项目的 CUB episodic 评测、简化 MAP head 和原论文的大规模少样本迁移协议仍有差异。
