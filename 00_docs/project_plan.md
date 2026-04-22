# 小样本学习项目计划

## 当前路线确认

- 项目方向：小样本学习。
- 框架：LibFewShot。
- 数据集：CUB。
- baseline：ProtoNet。
- 复现目标：复现一篇 2023-2026 年顶会小样本学习论文。
- 阶段 1 完成标准：`ml` 环境能运行 PyTorch。
- 当前环境结论：`ml` 环境 Python 3.12.13，PyTorch 2.11.0+cu130，`torch.cuda.is_available()` 为 `True`。
- 阶段 2-4 当前状态：LibFewShot 已下载，CUB 数据集已整理，ProtoNet baseline 已完成正式 CUB 运行；早期 synthetic sanity 验证产物已清理。

## 1. 项目目标

复现 1-2 种 LibFewShot 中没有的小样本学习算法，并在统一实验设置下提交：

- 可运行代码和配置文件。
- 实验思路与复现结果 PDF。
- 展示材料。
- 若代码质量和接口匹配度足够，尝试向 LibFewShot 提交贡献。

顶会范围：ICLR、CVPR、ICCV、ECCV、ICML、NeurIPS，年份范围：2023-2026。

## 2. 推荐技术路线

优先选择“标准 few-shot image classification / cross-domain few-shot classification”方向，因为它与 LibFewShot 的任务形态最接近，复现、比较和贡献成本最低。

建议主线：

1. 环境与框架跑通：安装 LibFewShot，完成 miniImageNet 或 CUB 的一次 baseline 训练/测试。
2. 基线复现实验：至少运行 ProtoNet、RelationNet、RFS 或 Baseline++ 中的 2-3 个，确认数据、日志和评估流程可靠。
3. 论文选择：从 2023-2024 的 CVPR/ICCV 方法中选 1 个主复现、1 个备选复现。
4. 方法接入：优先实现为 LibFewShot classifier 或 wrapper，保持配置文件、日志和评估接口统一。
5. 结果对齐：复现原论文主要表格中的 5-way 1-shot / 5-way 5-shot，记录均值和置信区间。
6. 消融分析：至少做 2-3 组关键模块消融，例如是否使用频域增强、语义辅助、特征增强、跨域适配等。
7. 报告整理：说明任务设定、论文方法、实现差异、实验设置、结果表格、失败原因和改进方向。

## 3. 候选论文优先级

### A. Frequency Guidance Matters in Few-Shot Learning, ICCV 2023

推荐程度：高。

优点：

- 属于标准 few-shot classification，和 LibFewShot 任务匹配。
- 方法核心是频域引导、mask、metric learning，便于做模块化实现。
- 可在 miniImageNet、CUB 等常见数据集上比较。

风险：

- 频域处理和多级损失需要仔细对齐原论文实现。
- 训练成本中等，可能需要先做简化版本。

### B. Frozen Feature Augmentation for Few-Shot Image Classification, CVPR 2024

推荐程度：高，尤其适合作为第二个方法或轻量复现。

优点：

- 思路简单：在 frozen feature space 做 augmentation。
- 对算力友好，适合先复现小规模结果。
- 很适合做消融：不同 feature augmentation、不同 backbone、不同 shot。

风险：

- 如果完全依赖大规模预训练特征，和 LibFewShot 原有训练流程可能需要 adapter。

### C. Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation, CVPR 2024

推荐程度：中高。

优点：

- 聚焦 cross-domain few-shot learning，方向较新。
- 参数高效适配容易形成清晰实验故事。

风险：

- Meta-Dataset 相关设置可能比 miniImageNet/CUB 更重。
- 与 LibFewShot 标准数据格式可能有额外适配成本。

### D. Simple Semantic-Aided Few-Shot Learning, CVPR 2024

推荐程度：中。

优点：

- 视觉特征 + 语义原型，方法故事清楚。
- 可作为多模态/语义辅助方向的亮点。

风险：

- 需要生成或准备高质量类别语义。
- 依赖外部语义资源时，复现可控性下降。

## 4. 最小可交付版本

如果时间紧，建议交付：

- 1 个主方法完整复现：FGFL 或 FroFA。
- 2-3 个 LibFewShot 原生基线。
- miniImageNet 和 CUB 中至少 1 个数据集的完整结果。
- 5-way 1-shot / 5-way 5-shot 两个设置。
- 1 页方法图或流程图。
- 1 张主结果表 + 1 张消融表。

## 5. 周期安排

第 1 周：

- 补齐 PyTorch、深度学习、few-shot 基础概念。
- 配置 LibFewShot 环境，下载 miniImageNet/CUB。
- 跑通一个官方 baseline。

第 2 周：

- 阅读 3-5 篇候选论文，确定主复现论文。
- 复现实验协议：way、shot、query、episode 数、backbone、输入尺寸、优化器。
- 建立实验记录模板。

第 3-4 周：

- 接入主方法代码。
- 跑小规模 sanity check。
- 跑正式 1-shot / 5-shot 实验。

第 5 周：

- 做消融实验。
- 与原论文和 LibFewShot baseline 比较。
- 分析误差来源。

第 6 周：

- 整理代码、报告 PDF 和展示材料。
- 按 LibFewShot 风格清理接口、配置和 README。

## 6. 实验记录规范

每次实验至少记录：

- 论文/方法名。
- commit hash 或代码版本。
- 数据集和划分。
- backbone、输入尺寸、预训练设置。
- way/shot/query/episode。
- batch size、optimizer、learning rate、scheduler。
- seed。
- 平均准确率和 95% confidence interval。
- 日志路径和模型权重路径。
- 与原论文设置的差异。

## 7. 参考资料

- LibFewShot documentation: https://libfewshot-en.readthedocs.io/
- LibFewShot paper: https://arxiv.org/abs/2109.04898
- FGFL, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_Frequency_Guidance_Matters_in_Few-Shot_Learning_ICCV_2023_paper.html
- FroFA, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Bar_Frozen_Feature_Augmentation_for_Few-Shot_Image_Classification_CVPR_2024_paper.html
- DIPA, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Perera_Discriminative_Sample-Guided_and_Parameter-Efficient_Feature_Space_Adaptation_for_Cross-Domain_Few-Shot_CVPR_2024_paper.html
- SemFew, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Simple_Semantic-Aided_Few-Shot_Learning_CVPR_2024_paper.html
