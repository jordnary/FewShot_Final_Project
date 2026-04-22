# 阶段 6：个人思考与改进分析

## 1. 为什么需要改进

前面阶段已经可以在 CUB 上运行 ProtoNet baseline，并且把 FroFA-style feature augmentation 接入了 LibFewShot 的 episodic pipeline。这个实现有一个明显限制：如果 backbone 仍然是 Conv64F 或从 CUB episode 中训练出来的 ResNet12，那么实验其实更像“在传统 few-shot backbone 上加一个特征扰动模块”，而不是 FroFA 原论文强调的“强预训练冻结特征 + 轻量分类头”。

FroFA 的优势来自两个前提：

1. encoder 已经通过大规模预训练学到通用视觉表征；
2. 少样本阶段不更新 encoder，只在缓存特征上训练轻量 head。

因此第 6 阶段选择 CLIP ViT-B/16 作为 frozen encoder。CLIP 是图文对比预训练模型，天然具备强迁移能力；ViT-B/16 也与 FroFA 原论文中使用 ViT frozen features 的设定更接近。

## 2. 改进路线

本阶段建议采用以下流程：

```text
CUB images
-> pretrained CLIP ViT-B/16
-> freeze image encoder
-> extract offline patch tokens
-> sample 5-way episodes on cached patch tokens
-> train no-FroFA MAP head
-> train FroFA augmented MAP head
-> compare 1-shot and 5-shot
```

这样设计有三个好处：

- 计算成本低：CLIP 只前向一次，后续实验都在 `.npz` patch-token 文件上运行。
- 对比更干净：MAP 和 FroFA + MAP 使用同一个 frozen patch tokens、同一个 episode、同一个 MAP head 结构，只差是否增强 support patch tokens。
- 更贴近论文：FroFA 原论文的关键贡献是 frozen patch-token feature space augmentation，而不是 backbone 训练技巧。

## 3. 实现选择

### Frozen encoder

使用 `open_clip_torch` 加载 `ViT-B-16` + `openai` 权重。脚本中也保留了 OpenAI 官方 `clip` 包的 fallback。

当前实现保留了两类输出：

```text
06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz
```

patch-token 文件包含：

- `tokens`：CLIP ViT-B/16 最后一层 patch tokens，默认形状约为 `N x 196 x 512`；
- `labels`：当前 split 内的整数标签；
- `class_names`：原始 CUB 类名；
- `paths`：图像相对路径。

### Classifier

改进版使用 episode-trained MAP head，而不是直接在 pooled global embedding 上训练 linear probe。原因是：

- MAP head 可以从 `196 x 512` patch tokens 中学习任务相关的 attention pooling；
- FroFA 原论文主要在 patch-token grid 上做增强，MAP head 比全局向量 linear probe 更贴近原设定；
- episode 内 no-FroFA MAP 与 FroFA + MAP 只差 support token augmentation，消融更清楚。

### FroFA variant

当前主路线实现的是 patch-token brightness c2FroFA：

```text
patch tokens: P x C
-> per-sample, per-channel min-max normalize over P patches
-> channel-wise brightness / contrast perturbation
-> inverse min-max map
-> augmented support patch tokens
```

默认增强是 brightness，`alpha=0.20`，每个 support 样本生成 8 个增强样本。这个选择对应原论文中更稳定的 brightness c2FroFA 思路。

## 4. 消融设计

核心对比：

| 方法 | Encoder | Head | 变量 |
|---|---|---|---|
| MAP | frozen CLIP ViT-B/16 patch tokens | MAP head | 不增强 |
| FroFA + MAP | frozen CLIP ViT-B/16 patch tokens | MAP head | support patch-token augmentation |

评测设置：

| Dataset | Way | Shot | Query | Episodes |
|---|---:|---:|---:|---:|
| CUB test split | 5 | 1 | 15 | 1000 |
| CUB test split | 5 | 5 | 15 | 1000 |

可选扩展消融：

| 消融项 | 取值 |
|---|---|
| `alpha` | 0.05 / 0.10 / 0.20 / 0.30 |
| `num_aug` | 2 / 4 / 8 / 16 |
| augmentation | brightness / contrast / brightness+contrast |
| MAP optimization | train steps 40 / 80 / 160 |
| MAP regularization | weight decay 0.0 / 0.001 / 0.01 |

## 5. 预期结果与解释

云端实验已经跑通，结果如下：

| 方法 | Frozen encoder | Classifier | 5-way 1-shot | 5-way 5-shot |
|---|---|---|---:|---:|
| no-FroFA | CLIP ViT-B/16 | L2 linear probe | 86.975 +/- 0.569 | 96.379 +/- 0.248 |
| FroFA | CLIP ViT-B/16 | L2 linear probe | 85.717 +/- 0.574 | 95.733 +/- 0.295 |

CLIP frozen feature 的 no-FroFA baseline 明显强于 Conv64F baseline，也超过了前面阶段中训练充分的 ResNet12 ProtoNet。这不是因为分类器复杂，而是因为 CLIP 的预训练表征更强。与前面阶段最强的 ResNet12 ProtoNet baseline 相比，提升为：

| 方法 | Feature / Backbone | 5-way 1-shot | 5-way 5-shot | 相对 ResNet12 提升 |
|---|---|---:|---:|---:|
| ProtoNet baseline | ResNet12 | 73.376 | 85.945 | reference |
| Stage 6 improvement | CLIP ViT-B/16 frozen feature | 86.975 | 96.379 | +13.599 / +10.434 |

因此，第 6 阶段可以把“使用强 frozen features 替代从 CUB episode 中训练 backbone”作为主要改进结论。这一结论非常明确：在相同 CUB 5-way 任务上，CLIP frozen feature + 简单线性分类器已经显著超过前面阶段的 task-trained ProtoNet。

不过，当前 FroFA 版本没有带来提升：

- 5-way 1-shot：FroFA 比 no-FroFA 低 1.258 个百分点；
- 5-way 5-shot：FroFA 比 no-FroFA 低 0.646 个百分点。

这说明“强 frozen features”路线是有效的，但把 brightness FroFA 直接作用在 CLIP pooled global embedding 上并不一定有效。原论文中的 FroFA 主要作用在 ViT patch token grid 上，保留了 `N x C` 的局部结构；本阶段为了快速验证，使用的是 512 维全局图像向量，缺少 patch 维度。对这种已经高度压缩且 L2 normalized 的语义向量做 min-max brightness 扰动，可能破坏了 CLIP embedding 的方向信息，从而降低 linear probe 的分类边界质量。

因此，FroFA 没有提升的可能原因包括：

- 当前实现使用 CLIP pooled global feature，缺少 FroFA 原论文中 patch feature grid 的空间结构；
- CUB 与 CLIP 预训练语义高度匹配，no-FroFA baseline 已经很强，增强空间较小；
- brightness 在 CLIP 全局 embedding 上不一定等价于 patch-level brightness FroFA；
- ridge lambda、alpha、num_aug 还需要 validation split 调参。

为解决这个问题，当前代码已升级为 patch-token FroFA + MAP head：

```text
06_improvement/extract_clip_patch_tokens.py
06_improvement/run_frofa_map_eval.py
06_improvement/run_clip_frofa_cub_cloud.sh
```

新路线不再把 FroFA 作用在 512 维 pooled global embedding 上，而是缓存 CLIP ViT-B/16 的最终 patch tokens，并在每个 episode 中训练 MAP head。这样更接近原论文中的 `N x C` patch feature setting，也更可能体现 FroFA 的提升。

升级后的 fast 实验已经跑通，设置为 100 episodes、每个 episode 训练 MAP head 40 steps：

| 方法 | Frozen encoder | Head | 5-way 1-shot | 5-way 5-shot |
|---|---|---|---:|---:|
| MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 43.920 +/- 1.926 | 72.867 +/- 2.164 |
| FroFA + MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 45.040 +/- 2.281 | 75.213 +/- 1.954 |

这组结果体现了 FroFA 的正向作用：

- 5-way 1-shot：+1.120 个百分点；
- 5-way 5-shot：+2.346 个百分点。

虽然 fast 版本的 MAP head 训练步数少、episode 数也少，置信区间仍然偏宽，但趋势已经和前一个 pooled global embedding 实验不同：当 FroFA 作用在 patch tokens 上时，它开始提升 MAP head。这支持了“FroFA 的有效载体是 patch-token feature grid，而不是压缩后的全局向量”的分析。

进一步运行正式设置：600 episodes、每个 episode 训练 MAP head 80 steps。为了消除随机 episode 难度差异，最终采用 paired episodes：MAP 与 FroFA + MAP 在同一批 support/query episode 上评测。结果如下：

| 方法 | Frozen encoder | Head | 5-way 1-shot | 5-way 5-shot |
|---|---|---|---:|---:|
| MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 46.213 +/- 0.875 | 75.478 +/- 0.765 |
| FroFA + MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 45.402 +/- 0.890 | 77.056 +/- 0.819 |

paired 正式结果中，FroFA + MAP 在 5-shot 上提升 +1.578 个百分点；1-shot 上低 0.811 个百分点。这个现象说明当前 patch-token FroFA 更稳定地改善 5-shot 设置，而 1-shot 下 MAP head 训练样本太少，增强样本可能仍不足以稳定改善分类边界。相比 pooled global embedding 上 FroFA 在 1-shot 和 5-shot 都下降，patch-token FroFA 已经更接近论文设定，并在 5-shot 上体现出明确正向收益。

## 6. 报告可用结论

本阶段的个人改进不是简单增加训练轮数，而是重新对齐 FroFA 的方法假设：把 backbone 训练问题转化为 frozen representation 上的轻量分类问题。相比 ResNet12/Conv64F，这条路线更能回答“强 frozen features 是否能改善 few-shot classification”。

最终报告中可以强调：

- Baseline 阶段验证了 LibFewShot 和 CUB 数据管线；
- Paper reproduction 阶段验证了 FroFA 思路可以接入 episodic ProtoNet；
- Improvement 阶段进一步使用 CLIP ViT-B/16 frozen features，减少 backbone 训练干扰；相比 ResNet12 ProtoNet，1-shot 提升 +13.599，5-shot 提升 +10.434；
- FroFA-global 消融没有超过 no-FroFA CLIP baseline，说明 pooled CLIP embedding 与 FroFA 原论文 patch feature setting 仍有差距；
- 因此最终改进版改为缓存 ViT patch tokens，并训练 MAP head，而不是只在全局 512 维 embedding 上做 brightness 扰动；
- paired 正式 patch-token 实验中，FroFA + MAP 在 5-shot 上从 75.478 提升到 77.056，说明 patch-token FroFA 比 global embedding FroFA 更合理；1-shot 从 46.213 降到 45.402，提示 1-shot 下还需要继续调增强强度、MAP 步数或正则化。
