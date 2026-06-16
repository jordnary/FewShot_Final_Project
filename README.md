# Few-Shot Learning Reproduction Project

本项目围绕细粒度图像小样本分类展开，使用 LibFewShot 作为基础框架，在 CUB 数据集上建立 ProtoNet baseline，复现 Frozen Feature Augmentation for Few-Shot Image Classification (FroFA, CVPR 2024) 的核心思想，并进一步探索基于 CLIP frozen features 与 patch-token augmentation 的改进路线。

## Project Goals

1. 在统一的 CUB few-shot classification 设置下跑通 LibFewShot 实验流程。
2. 建立 ProtoNet baseline，用于后续方法复现和消融对比。
3. 将 FroFA-style feature augmentation 接入 episodic few-shot 流程，比较 5-way 1-shot 与 5-way 5-shot 表现。
4. 探索更强 frozen representation，重点关注 CLIP ViT-B/16 global features 与 patch tokens。
5. 按论文思路组织 post-LN patch tokens、validation sweep 和 paired episode evaluation 的实验流程。
6. 整理实验脚本、结果汇总和报告材料，使项目可以被快速理解、检查和演示。

## Repository Structure

```text
.
├── environment/       # Python/Conda 依赖与环境说明
├── data/              # 本地数据集挂载目录；Git 仅保留占位文件
├── third_party/       # LibFewShot 框架代码和项目内改动说明
├── configs/           # Baseline 与 FroFA 的 LibFewShot YAML 配置
├── scripts/           # Baseline、FroFA、CLIP-FroFA 的运行入口
├── experiments/       # 每条实验线的说明文档和可提交 CSV 结果
├── docs/              # 项目计划、论文笔记和论文 PDF
├── report/            # 最终报告、图表、表格和展示材料
├── demo/              # 项目演示检查清单和辅助入口
└── artifacts/         # 日志、checkpoint、feature cache、预训练权重等本地产物
```

## Main Workflow

1. 按 `environment/setup_guide.md` 配置环境，并安装 `environment/requirements.txt`。
2. 将 CUB 数据放入 `data/CUB_200_2011/`，至少包含 `images/`、`train.csv`、`val.csv` 和 `test.csv`。
3. 运行 `scripts/baseline/` 中的 ProtoNet baseline 脚本，生成 baseline 对照。
4. 运行 `scripts/frofa/` 中的 FroFA 复现实验，并与 baseline 对比。
5. 运行 `scripts/clip_frofa/` 中的 CLIP frozen feature、patch-token MAP 和 paperlike sweep 实验。
6. 在 `experiments/*/results/` 查看可提交的 CSV 汇总，在 `report/` 整理最终报告和展示材料。

## Experiment Lines

- Baseline: CUB 上的 ProtoNet few-shot classification，对比 Conv64F 与 ResNet12。
- FroFA reproduction: FroFA-style support feature augmentation，接入 LibFewShot episodic ProtoNet 流程。
- CLIP-FroFA improvement: 使用 CLIP ViT-B/16 frozen global features、projected patch tokens、post-LN patch tokens 与 MAP head，比较 no-FroFA、vector-level FroFA、patch-token FroFA 和 validation-selected paperlike FroFA。

## Data And Outputs

`data/` 只提交 `.gitkeep` 占位文件，真实图像数据不纳入 Git。

`artifacts/` 保存日志、模型权重、特征缓存和预训练权重等大体积或可再生成产物，也不纳入 Git。可复查的小型结果表保留在 `experiments/*/results/`。

## Useful Entry Points

```bash
pip install -r environment/requirements.txt
bash scripts/baseline/run_proto_cub_boost_cloud.sh
bash scripts/frofa/run_frofa_cub_cloud.sh all
bash scripts/clip_frofa/run_patch_frofa_paperlike_cloud.sh
```

更详细的运行顺序见 `docs/runbook.md` 和各实验目录下的 `run_guide.md`。
