# 05_paper_reproduction 项目结构

本目录用于复现 FroFA-style feature augmentation + ProtoNet + ResNet12 在 CUB few-shot 分类任务上的实验，并与 `04_baseline_experiments` 中的 ProtoNet-ResNet12 baseline 对比。

## 目录概览

```text
05_paper_reproduction/
├─ configs/
│  ├─ frofa_proto_cub_resnet12_5way_1shot_cloud.yaml
│  ├─ frofa_proto_cub_resnet12_5way_5shot_cloud.yaml
│  ├─ frofa_proto_cub_resnet12_frozen_5way_1shot_cloud.yaml
│  └─ frofa_proto_cub_resnet12_frozen_5way_5shot_cloud.yaml
├─ docs/
│  ├─ project_structure.md
│  ├─ run_guide.md
│  └─ experiment_summary.md
├─ logs/
├─ results/
├─ run_frofa_cub_cloud.sh
└─ summarize_frofa_results.py
```

## 文件职责

`configs/` 保存 FroFA 复现实验的 LibFewShot YAML 配置。四个配置对应 5-way 1-shot、5-way 5-shot，以及 joint training 和 frozen backbone 两种设置。

`docs/` 保存本阶段整理后的说明文档。`project_structure.md` 说明项目结构，`run_guide.md` 面向意图复现者说明如何运行，`experiment_summary.md` 归纳实验过程、数据和结论。

`logs/` 保存控制台日志，例如 `frofa_proto_cub_resnet12_5way_1shot_cloud_console.log`。这些日志用于追溯训练状态，并被 `summarize_frofa_results.py` 读取。

`results/` 保存 LibFewShot 输出目录、checkpoint、每次实验的 `.csv` 结果文件，以及供 07 报告读取的 baseline 对比总结。当前核心总结是 `frofa_vs_baseline_summary.csv`。

`run_frofa_cub_cloud.sh` 是云端运行入口。它支持 `joint`、`frozen` 和 `all` 三种模式，内部调用 `04_baseline_experiments/run_libfewshot_config_cloud.py` 渲染 `${PROJECT_ROOT}` 并启动 LibFewShot 训练。

`summarize_frofa_results.py` 从 05 阶段 FroFA 日志中提取最终测试精度，为每次实验生成独立的 `.csv`。同时它会读取 04 阶段 baseline 日志，生成 `frofa_vs_baseline_summary.csv`，作为本阶段唯一总结。

## 外部依赖

FroFA 复现依赖 `02_libfewshot/LibFewShot` 中新增或改造的 FroFAProtoNet 实现：

```text
02_libfewshot/LibFewShot/config/classifiers/FroFAProto.yaml
02_libfewshot/LibFewShot/core/model/metric/frofa_proto_net.py
```

frozen backbone 设置还依赖 04 阶段 ResNet12 baseline 的最优权重：

```text
04_baseline_experiments/results/proto_cub_resnet12_boost_5way_1shot_cloud/checkpoints/emb_func_best.pth
04_baseline_experiments/results/proto_cub_resnet12_boost_5way_5shot_cloud/checkpoints/emb_func_best.pth
```

汇总脚本还会读取 04 阶段 baseline 日志：

```text
04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
```

## 实验分层

| 层级 | 配置文件 | 作用 |
| :--- | :--- | :--- |
| joint FroFA | `frofa_proto_cub_resnet12_5way_*_cloud.yaml` | 端到端训练 ResNet12 + FroFAProtoNet |
| frozen FroFA | `frofa_proto_cub_resnet12_frozen_5way_*_cloud.yaml` | 加载 04 阶段 baseline backbone，冻结特征提取器后训练 FroFAProtoNet |
| summary | `summarize_frofa_results.py` | 汇总 joint FroFA、frozen FroFA 与 baseline 的对比结果 |

## 数据流

1. 准备 `03_datasets/CUB_200_2011` 数据目录。
2. 先完成 04 阶段 ResNet12 baseline，获得日志和 backbone 权重。
3. 选择 `05_paper_reproduction/configs/` 中的 FroFA 配置。
4. 通过 `run_frofa_cub_cloud.sh` 启动 joint、frozen 或 all 模式。
5. 控制台输出写入 `05_paper_reproduction/logs/`。
6. LibFewShot 结果和 checkpoint 写入 `05_paper_reproduction/results/`。
7. `summarize_frofa_results.py` 汇总 05 阶段日志结果，写入每次实验结果文件和 `frofa_vs_baseline_summary` 总结。
