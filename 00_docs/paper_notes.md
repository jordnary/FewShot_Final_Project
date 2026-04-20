# 论文笔记

## 论文筛选原则

优先选择满足以下条件的论文：

- 2023-2026 年 ICLR/CVPR/ICCV/ECCV/ICML/NeurIPS。
- 任务是 few-shot image classification 或 cross-domain few-shot classification。
- 原论文有公开代码或清晰伪代码。
- 可在 miniImageNet、tieredImageNet、CUB、CIFAR-FS 或 Meta-Dataset 上复现。
- 方法不在 LibFewShot 当前经典 18 方法范围内。
- 能拆出清晰模块，方便做消融和贡献。

## 候选 1: Frequency Guidance Matters in Few-Shot Learning

- 会议：ICCV 2023。
- 任务：few-shot classification。
- 方法简称：FGFL。
- 核心思想：利用任务相关频率成分，生成 masked/unmasked 图像信息，并通过多级 metric learning 提升泛化。
- 可能实现位置：LibFewShot classifier + transform/frequency module。
- 主要实验：standard、cross-dataset、cross-domain、coarse-to-fine。
- 推荐复现设置：先做 miniImageNet 或 CUB 的 5-way 1-shot/5-shot。
- 链接：https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_Frequency_Guidance_Matters_in_Few-Shot_Learning_ICCV_2023_paper.html

待读重点：

- 频域 mask 如何生成。
- masked/unmasked/original 三类特征如何参与损失。
- 训练阶段和测试阶段是否都需要频域模块。
- 与 ProtoNet/RFS 类 baseline 的接口差异。

## 候选 2: Frozen Feature Augmentation for Few-Shot Image Classification

- 会议：CVPR 2024。
- 任务：few-shot image classification。
- 方法简称：FroFA。
- 核心思想：冻结预训练视觉模型特征，在 feature space 中做 augmentation，再训练轻量分类器。
- 可能实现位置：feature extractor wrapper + feature augmentation module + classifier。
- 推荐复现设置：先提取 frozen features，再比较 no augmentation / brightness / pointwise FroFA。
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.html

待读重点：

- 论文中 20 种 feature augmentation 的定义。
- 哪种 augmentation 是主结果使用的默认方法。
- backbone 和预训练数据集对结果的影响。
- 如何把 frozen feature pipeline 接到 LibFewShot。

## 候选 3: Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation

- 会议：CVPR 2024。
- 任务：cross-domain few-shot learning。
- 方法简称：DIPA。
- 核心思想：用轻量线性变换进行参数高效特征适配，并替换传统 nearest centroid classifier 为 sample-aware discriminative loss。
- 可能实现位置：feature adaptation module + discriminative classifier。
- 推荐复现设置：如果 Meta-Dataset 配置太重，先在 miniImageNet -> CUB 或 miniImageNet -> Cars/Aircraft 的跨域设置上做简化验证。
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Perera_Discriminative_Sample-Guided_and_Parameter-Efficient_Feature_Space_Adaptation_for_Cross-Domain_Few-Shot_CVPR_2024_paper.html

待读重点：

- 参数高效适配层的数学形式。
- sample-aware loss 的输入输出。
- seen/unseen domain 的划分方式。
- 原论文和 LibFewShot 数据协议的差异。

## 候选 4: Simple Semantic-Aided Few-Shot Learning

- 会议：CVPR 2024。
- 任务：few-shot classification。
- 方法简称：SemFew。
- 核心思想：自动生成高质量类别语义，并用简单的 Semantic Alignment Network 将语义和视觉特征转成鲁棒类别原型。
- 可能实现位置：semantic prototype loader + semantic alignment classifier。
- 推荐复现设置：先固定类别语义文本，复现视觉特征 + 语义原型的核心分类逻辑。
- 链接：https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Simple_Semantic-Aided_Few-Shot_Learning_CVPR_2024_paper.html

待读重点：

- Semantic Evolution 如何生成类别语义。
- Semantic Alignment Network 的结构。
- 类别语义是否需要外部模型生成。
- 无语义、短类名语义、高质量语义三组消融。

## 当前建议

主复现优先选 FGFL；备选或第二复现选 FroFA。

原因：

- FGFL 更接近传统 few-shot 分类论文，适合展示“算法复现”能力。
- FroFA 轻量、可控、消融清楚，适合做第二方法或改进实验。
- 两者都不像 ProtoNet/MAML/DN4 那样属于 LibFewShot 经典已覆盖内容，更符合“复现 LibFewShot 中没有的方法”的要求。
