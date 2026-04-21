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
-> extract offline features
-> sample 5-way episodes on cached features
-> train no-FroFA linear probe
-> train FroFA augmented linear probe
-> compare 1-shot and 5-shot
```

这样设计有三个好处：

- 计算成本低：CLIP 只前向一次，后续实验都在 `.npz` 特征文件上运行。
- 对比更干净：no-FroFA 和 FroFA 使用同一个 frozen feature、同一个 episode、同一个 linear classifier，只差是否增强 support features。
- 更贴近论文：FroFA 原论文的关键贡献是 frozen feature space augmentation，而不是 backbone 训练技巧。

## 3. 实现选择

### Frozen encoder

使用 `open_clip_torch` 加载 `ViT-B-16` + `openai` 权重。脚本中也保留了 OpenAI 官方 `clip` 包的 fallback。

输出特征文件：

```text
06_improvement/results/features/cub_train_clip_vit_b16.npz
06_improvement/results/features/cub_val_clip_vit_b16.npz
06_improvement/results/features/cub_test_clip_vit_b16.npz
```

每个文件包含：

- `features`：归一化后的 CLIP image features；
- `labels`：当前 split 内的整数标签；
- `class_names`：原始 CUB 类名；
- `paths`：图像相对路径。

### Classifier

使用 closed-form L2 linear probe，而不是训练很多步的 SGD/Adam 线性层。原因是：

- episode 数可以较大，闭式解速度更稳定；
- 结果可复现，不容易受优化步数影响；
- FroFA 原论文也将 linear probe 作为重要对照。

### FroFA variant

由于当前脚本使用 CLIP 的全局 image embedding，特征形状是 `D`，不是 ViT patch grid 的 `N x C`。因此这里实现的是 vector-feature FroFA：

```text
feature vector
-> per-sample min-max normalize to [0, 1]
-> brightness / contrast perturbation
-> inverse min-max map
-> augmented support feature
```

默认增强是 brightness，`alpha=0.20`，每个 support 样本生成 8 个增强样本。这个选择保守、简单，也符合 FroFA 论文中 brightness 类增强最稳定的观察。

## 4. 消融设计

核心对比：

| 方法 | Encoder | Classifier | 变量 |
|---|---|---|---|
| no-FroFA | frozen CLIP ViT-B/16 | L2 linear probe | 不增强 |
| FroFA | frozen CLIP ViT-B/16 | L2 linear probe | support feature augmentation |

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
| classifier regularization | ridge lambda 0.1 / 1.0 / 10.0 |

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

## 6. 报告可用结论

本阶段的个人改进不是简单增加训练轮数，而是重新对齐 FroFA 的方法假设：把 backbone 训练问题转化为 frozen representation 上的轻量分类问题。相比 ResNet12/Conv64F，这条路线更能回答“强 frozen features 是否能改善 few-shot classification”。

最终报告中可以强调：

- Baseline 阶段验证了 LibFewShot 和 CUB 数据管线；
- Paper reproduction 阶段验证了 FroFA 思路可以接入 episodic ProtoNet；
- Improvement 阶段进一步使用 CLIP ViT-B/16 frozen features，减少 backbone 训练干扰；相比 ResNet12 ProtoNet，1-shot 提升 +13.599，5-shot 提升 +10.434；
- FroFA-global 消融没有超过 no-FroFA CLIP baseline，说明 pooled CLIP embedding 与 FroFA 原论文 patch feature setting 仍有差距；
- 后续如果继续改进，应缓存 ViT patch tokens，并训练 MAP head 或 attention pooling classifier，而不是只在全局 512 维 embedding 上做 brightness 扰动。
