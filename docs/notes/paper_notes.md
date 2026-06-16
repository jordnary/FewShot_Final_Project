# 论文笔记：Frozen Feature Augmentation for Few-Shot Image Classification

## 基本信息

- 论文：Frozen Feature Augmentation for Few-Shot Image Classification
- 作者：Andreas Baer, Neil Houlsby, Mostafa Dehghani, Manoj Kumar
- 会议：CVPR 2024
- 项目主页：https://frozen-feature-augmentation.github.io
- 主题：少样本图像分类、预训练视觉模型、冻结特征、特征空间数据增强

## 总结

这篇论文提出 Frozen Feature Augmentation（FroFA）：先用大规模预训练 ViT 提取并缓存冻结特征，再把传统图像增强映射到冻结特征空间，在训练轻量分类头时对特征做随机增强。实验表明，简单的逐点风格增强，尤其是 brightness 的通道级变体，在多种模型规模、预训练数据和少样本数据集上能稳定提升性能。

## 研究动机

大规模预训练视觉模型的冻结特征已经能在下游少样本任务上取得很强结果，例如只训练 linear probe 或轻量 MAP head。与此同时，图像级数据增强是低数据场景中提升泛化能力的常用手段。

论文关注的问题是：既然少样本迁移时通常只训练冻结特征之上的小模型，能否不回到图像输入和大模型反向传播，而是直接在缓存后的冻结特征空间中做数据增强？

这个设定的吸引力在于：

- 预训练 backbone 完全冻结，训练和超参搜索成本低。
- 特征可以提前缓存，实验能在分钟级完成。
- 数据增强作用在轻量模型训练阶段，不需要重新跑大模型。
- 对少样本任务尤其有价值，因为训练样本少，轻量头容易过拟合。

## 方法概览

整体流程分为三步：

1. 选择一个预训练 ViT，并确定要缓存的层。论文主要缓存最后一个 Transformer block 输出。
2. 把少样本数据集中的图片输入冻结 ViT，得到形状为 `N x C` 的 patch 特征并缓存为特征数据集。
3. 在缓存特征上训练轻量 MAP head 和分类层，训练时对冻结特征应用 FroFA。

其中 `N` 是 patch 数量，`C` 是每个 patch 的通道数。论文使用 MAP（multi-head attention pooling）作为主要轻量头，另以 closed-form L2 linear probe 作为辅助 baseline。

## FroFA 的核心思想

传统图像增强一般定义在像素空间，例如 brightness、contrast、crop、rotate 等。FroFA 将这类增强搬到特征空间，关键是处理两个差异：

- 图像通常是 3 通道，特征有任意通道数 `C`。
- 图像值通常在 `[0, 1]` 或 `[0, 255]`，冻结特征没有固定值域。

论文的做法是：

1. 将 ViT 的 `N x C` patch 特征 reshape 成近似二维网格，即 `sqrt(N) x sqrt(N) x C`。
2. 用 min-max 映射把特征缩放到图像增强可处理的值域 `[0, 1]`。
3. 在这个归一化后的特征“图像”上应用增强。
4. 再用逆变换映射回原始特征值域。

可读公式：

```text
feature -> min-max normalize -> image-style augmentation -> inverse min-max -> augmented feature
```

论文将这个组合记为：

```text
af = t_x_to_f o a_x o t_f_to_x
```

其中 `t_f_to_x` 是特征到图像值域的映射，`a_x` 是传统图像增强，`t_x_to_f` 是逆映射。

## 三种 FroFA 变体

### 1. Default FroFA

对整个特征整体做一次增强。min/max 在所有元素上统计，增强参数也对整个特征共享。

例子：contrast FroFA 会采样一个随机 contrast factor，然后对整个归一化特征使用同一个 factor。

### 2. Channel FroFA（cFroFA）

增强参数按通道独立采样。也就是说，每个通道都有自己的随机增强强度，但 min/max 映射仍然可以基于整体特征。

例子：contrast cFroFA 会为 `C` 个通道分别采样 `C` 个 contrast factor。

### 3. Channel2 FroFA（c2FroFA）

进一步把 min-max 映射也改成按通道计算。也就是说，增强参数按通道采样，特征到 `[0, 1]` 的缩放和逆缩放也按通道进行。

论文发现 brightness c2FroFA 更稳定，对增强强度不那么敏感，是后续实验的主要选择。

作者也尝试过 element-wise FroFA，但效果明显变差。他们推测逐元素随机增强会严重破坏特征的整体结构。

## 实验设置

### 模型

论文使用 ViT Ti/16、B/16、L/16 三种规模，并在轻量头中使用 MAP。

### 预训练数据

- JFT-3B：约 30 亿多标签图像。
- ImageNet-21k：约 1419 万图像，21841 类。
- WebLI + SigLIP：大规模图文预训练模型的 L/16 ViT 图像编码器。

### 少样本数据集

论文评估了 8 个少样本迁移数据集：

- ILSVRC-2012 / ImageNet-1k
- CIFAR10
- CIFAR100
- DMLab
- DTD
- Resisc45
- SUN397
- SVHN

ImageNet-1k 使用 1、5、10、25-shot 设置。其余数据集也构造对应少样本训练集，并使用验证集调参、测试集汇报结果。

### 增强类型

论文共考察 20 种增强，其中正文主表覆盖 18 种：

- 几何增强：rotate、shear-x、shear-y、translate-x、translate-y
- crop/drop：crop、resized crop、inception crop、patch dropout
- 风格增强：brightness、contrast、equalize、invert、posterize、sharpness、solarize
- 其他：JPEG、mixup

后续还测试了 RandAugment、TrivialAugment 风格的组合，但效果有限。

### 训练细节

MAP head 训练时 sweep：

- batch size：32、64、128、256、512
- learning rate：0.01、0.03、0.06、0.1
- training steps：1000、2000、4000、8000、16000

在 Sec. 6 和 Sec. 7 中，MAP 还 sweep weight decay：0.01、0.001、0.0001、0.0。论文用 `MAPwd` 表示带 weight decay sweep 的 MAP baseline。

## 主要实验结果

### Baseline

在 JFT-3B L/16 + ImageNet-1k few-shot 设置下：

| 方法 | 1-shot | 5-shot | 10-shot | 25-shot |
|---|---:|---:|---:|---:|
| MAP | 57.9 | 78.8 | 80.9 | 83.2 |
| Linear probe | 66.5 | 79.6 | 81.5 | 82.4 |

Linear probe 在 1-shot 明显强于 MAP，但 MAP 使用 `N x C` patch 特征，保留了二维结构，因此更适合做 FroFA。

### Default FroFA

在 ImageNet-1k 上，default FroFA 的规律非常清晰：

- 几何增强几乎全部降低性能。
- crop 和 resized crop 在 1-shot 有明显收益，但在更高 shot 下收益很小。
- 风格增强最有效，尤其是 brightness、contrast、posterize。
- JPEG 和 mixup 没有带来收益，甚至会下降。

最突出的 default FroFA 是 brightness：

| 设置 | MAP | brightness FroFA 提升 |
|---|---:|---:|
| 1-shot | 57.9 | +4.8 |
| 5-shot | 78.8 | +1.1 |
| 10-shot | 80.9 | +0.6 |
| 25-shot | 83.2 | +0.1 |

这说明 FroFA 的收益在样本越少时越明显。

### Channel FroFA

对 brightness、contrast、posterize 进一步测试通道级增强：

| 增强 | 1-shot | 5-shot | 10-shot | 25-shot |
|---|---:|---:|---:|---:|
| brightness FroFA | +4.8 | +1.1 | +0.6 | +0.1 |
| brightness cFroFA | +5.9 | +1.5 | +1.1 | +0.4 |
| brightness c2FroFA | +6.1 | +1.6 | +0.9 | +0.3 |
| posterize FroFA | +3.7 | +0.8 | +0.6 | +0.0 |
| posterize cFroFA | +5.9 | +0.8 | +0.5 | +0.0 |

亮点：

- brightness cFroFA / c2FroFA 明显优于 default brightness。
- brightness c2FroFA 对 brightness level 更稳定。
- contrast 的通道级版本没有进一步提升。

### Sequential FroFA

作者把三个较强增强组合成两步顺序增强：

- Bc2：brightness c2FroFA
- C：contrast FroFA
- Pc：posterize cFroFA

最好的 1-shot 组合是 `Bc2 -> Pc`，提升从单独 Bc2 的 `+6.1` 增加到 `+7.7`。不过在 5、10、25-shot 上，顺序组合收益不稳定，说明组合策略仍有继续研究空间。

### 更多模型与预训练数据

在 JFT-3B 和 ImageNet-21k 预训练的 Ti/16、B/16、L/16 ViT 上，brightness c2FroFA 基本都能维持或提升 MAPwd baseline。

趋势：

- 1-shot 下，JFT-3B 模型越大，FroFA 对 MAPwd 的提升越明显。
- shot 数变多后，baseline 已经较强，FroFA 的提升幅度变小。
- 相比 linear probe，FroFA 在 1-shot 有时仍落后，尤其是 ImageNet-21k；但在 5-shot 到 25-shot，多数设置下能超过或至少持平 linear probe。

### 更多少样本数据集

在 CIFAR10、CIFAR100、DMLab、DTD、Resisc45、SUN397、SVHN 七个数据集上的平均结果：

| 预训练 | 方法 | 1-shot | 5-shot | 10-shot | 25-shot |
|---|---|---:|---:|---:|---:|
| JFT-3B | MAPwd | 49.5 | 65.8 | 68.3 | 74.1 |
| JFT-3B | Linear probe | 49.1 | 62.7 | 65.7 | 68.8 |
| JFT-3B | MAPwd + FroFA | 53.4 | 67.3 | 70.9 | 74.9 |
| WebLI + SigLIP | MAPwd | 45.9 | 67.7 | 71.8 | 75.1 |
| WebLI + SigLIP | Linear probe | 49.1 | 65.0 | 69.3 | 72.6 |
| WebLI + SigLIP | MAPwd + FroFA | 51.3 | 70.4 | 73.5 | 76.0 |

JFT-3B 上，FroFA 相对 MAPwd 的平均提升为：

- 1-shot：+3.9
- 5-shot：+1.5
- 10-shot：+2.6
- 25-shot：+0.8

WebLI + SigLIP 上，FroFA 也稳定最好：

- 1-shot 比 linear probe 高 +2.2
- 5-shot 比 MAPwd 高 +2.7
- 10-shot 比 MAPwd 高 +1.7
- 25-shot 比 MAPwd 高 +0.9

这说明 FroFA 不只适用于纯图像分类预训练，也能迁移到视觉-语言预训练的图像编码器。

## 关键结论

1. 冻结特征并不是只能原样使用，特征空间数据增强可以有效改善少样本分类。
2. 不是所有图像增强都适合搬到特征空间。几何增强会破坏 patch 特征结构，基本都降低性能。
3. 简单的逐点风格增强最有效，brightness 是最稳定的选择。
4. 通道级随机性很重要。brightness cFroFA / c2FroFA 比 default brightness 更强。
5. FroFA 在极低样本下收益最大，但在 5-shot 到 25-shot 也常能超过 linear probe。
6. FroFA 对模型规模、预训练数据、迁移数据集都有较好泛化性。

## 复现与实现要点

如果在本项目中复现，可以从最小可行版本开始：

1. 使用预训练 ViT 提取最后一层 patch token 特征，缓存为 `N x C`。
2. 不要先做复杂增强，先实现 brightness c2FroFA。
3. 对每个样本、每个通道分别计算 min/max，把该通道缩放到 `[0, 1]`。
4. 对每个通道采样一个 brightness 偏移或强度参数，作用到归一化特征后 clip 到合法范围。
5. 用对应通道的 min/max 逆变换回原特征值域。
6. 在增强后的特征上训练 MAP head 或一个较轻量的 attention pooling classifier。
7. 与两个 baseline 比较：不增强的 MAP head、linear probe。

实现时需要注意的要点：

- 如果 patch 数 `N` 不是平方数，论文中的二维 reshape 假设会变麻烦；常见 ViT 输入下 `N` 通常可 reshape 为 `sqrt(N) x sqrt(N)`。
- 对 brightness 这类逐点增强，不一定强依赖二维邻接结构；但 crop、rotate、patch dropout 等会依赖 patch 网格结构。
- 几何增强不建议作为首选，因为论文主实验显示它们稳定伤害性能。
- c2FroFA 的 min/max 是按通道统计，不是全特征统计。
- 训练和调参必须使用 validation split，最终只在 test split 汇报。
- 少样本划分最好多采样几次，论文对每个 shot 采样 5 次后报告平均结果。

## 对本项目的启发

本项目如果目标是 few-shot image classification，FroFA 是一个低成本、高性价比的增强方向。相比微调 backbone，它更适合在算力有限或需要快速实验时使用。优先级如下：

1. 先实现 frozen feature caching。
2. 训练不增强的 linear probe / MAP baseline。
3. 实现 brightness c2FroFA，并只在训练阶段启用。
4. 加入 validation-based sweep，例如学习率、训练步数、weight decay、brightness level。
5. 如果 baseline 稳定，再尝试 posterize cFroFA 或 `brightness c2FroFA -> posterize cFroFA` 的顺序组合。

## 局限性

- FroFA 的最佳增强依赖特征分布，目前主要来自经验搜索，理论解释还较弱。
- 几何增强在特征空间效果差，说明 patch 特征的空间结构不能简单等同于图像像素结构。
- 论文主要训练 MAP head，若换成不同分类头，最佳 FroFA 可能变化。
- 1-shot 下 linear probe 仍然很强，FroFA 并不总能完全弥补 MAP head 的差距。
- 顺序增强、RandAugment/TrivialAugment 式策略仍未充分开发，可能是后续研究空间。

