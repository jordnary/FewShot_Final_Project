# Few-Shot Learning Reproduction Project

本项目围绕细粒度图像小样本分类展开，使用 LibFewShot 作为基础框架，在 CUB 数据集上建立 ProtoNet baseline，复现 Frozen Feature Augmentation for Few-Shot Image Classification (FroFA, CVPR 2024) 的核心思想，并进一步探索基于 CLIP frozen features 与 patch-token augmentation 的改进路线。

项目目标不是只保存一次实验结果，而是形成一套可复查、可继续扩展的复现实验结构：从环境配置、数据准备、基线实验、论文复现、改进实验到最终报告，各阶段产物按目录隔离，便于他人阅读和复跑。

## Project Goals

1. 在统一的 CUB few-shot classification 设置下跑通 LibFewShot 实验流程。
2. 建立 ProtoNet baseline，用于后续方法复现和消融对比。
3. 将 FroFA-style feature augmentation 接入 episodic few-shot 流程，比较 5-way 1-shot 与 5-way 5-shot 表现。
4. 探索更强 frozen representation，重点关注 CLIP ViT-B/16 global features 与 patch tokens。
5. 整理实验脚本、结果汇总和报告材料，使项目可以被他人快速理解、检查和扩展。

## Repository Structure

```text
.
├── 00_docs/                  # 项目计划、论文笔记、阅读记录
├── 01_environment/           # Python/Conda 依赖、环境配置和运行说明
├── 02_libfewshot/            # LibFewShot 代码、配置改动和项目内说明
├── 03_datasets/              # 数据集本地挂载目录；Git 仅保留空目录占位
├── 04_baseline_experiments/  # ProtoNet baseline 运行脚本、配置和结果汇总
├── 05_paper_reproduction/    # FroFA 论文复现相关代码、脚本和结果
├── 06_improvement/           # CLIP/FroFA 改进实验、消融脚本和阶段总结
└── 07_report/                # 本地报告、图表和展示材料；默认不纳入 Git
```

## Main Workflow

1. 阅读 `00_docs/project_plan.md` 了解项目路线和候选论文。
2. 按 `01_environment/setup_guide.md` 配置运行环境，并安装 `01_environment/requirements.txt` 中的依赖。
3. 将 CUB 等本地数据放入 `03_datasets/`。该目录用于体现项目结构，但真实数据文件不提交到 Git。
4. 运行 `04_baseline_experiments/` 中的 ProtoNet baseline 脚本，确认 LibFewShot、数据划分和评估流程正常。
5. 运行 `05_paper_reproduction/` 中的 FroFA 复现实验，并与 baseline 结果对比。
6. 运行 `06_improvement/` 中的 CLIP frozen feature 与 patch-token FroFA 实验，分析改进是否稳定。
7. 最终报告和图表可在本地放入 `07_report/`，该目录内容默认被 `.gitignore` 忽略。

## Data And Outputs

`03_datasets/` 只提交 `.gitkeep` 占位文件。克隆仓库后，使用者需要自行下载或挂载 CUB 数据，并保持脚本中约定的数据路径。

训练日志、模型权重、原始数据、报告导出文件和大体积二进制文件不纳入版本管理。可复查的小型结果表、阶段总结和运行脚本保留在对应实验目录中。

## Current Experiment Line

当前主线由三部分组成：

- Baseline: CUB 上的 ProtoNet few-shot classification，对比 Conv64F 与 ResNet12。
- Reproduction: FroFA-style support feature augmentation，接入 LibFewShot episodic ProtoNet 流程。
- Improvement: 使用 CLIP ViT-B/16 frozen global features 与 patch tokens，比较 no-FroFA、vector-level FroFA 和 patch-token FroFA。

更详细的实验过程、参数和结果说明见各阶段目录下的 `docs/` 文件。
