# FroFA 复现实验结构

本实验线用于复现 FroFA-style feature augmentation + ProtoNet + ResNet12 在 CUB few-shot 分类任务上的实验，并与 ProtoNet-ResNet12 baseline 对比。

## 相关目录

```text
configs/frofa/                         # FroFA 复现实验 YAML 配置
scripts/frofa/                         # FroFA 运行和汇总入口
experiments/frofa_reproduction/        # 实验说明与可提交 CSV 结果
artifacts/logs/frofa/                  # 控制台日志，不纳入 Git
artifacts/runs/frofa/                  # LibFewShot 输出、checkpoint，不纳入 Git
artifacts/runs/baseline/               # frozen FroFA 依赖的 baseline backbone
```

## 文件职责

`configs/frofa/` 保存 FroFA 复现实验的 LibFewShot YAML 配置。四个配置对应 5-way 1-shot、5-way 5-shot，以及 joint training 和 frozen backbone 两种设置。

`scripts/frofa/run_frofa_cub_cloud.sh` 是成组运行入口，支持 `joint`、`frozen` 和 `all` 三种模式。它内部调用 `scripts/baseline/run_libfewshot_config_cloud.py` 渲染 `${PROJECT_ROOT}` 并启动 LibFewShot 训练。

`scripts/frofa/summarize_frofa_results.py` 从 FroFA 日志中提取最终测试精度，为每次实验生成独立 CSV，并生成 `frofa_vs_baseline_summary.csv`。

`experiments/frofa_reproduction/` 保存本实验线文档和小型 CSV 结果。大体积训练输出、checkpoint 和日志保存在 `artifacts/`。

## 外部依赖

FroFA 复现依赖 `third_party/LibFewShot` 中新增或改造的 FroFAProtoNet 实现：

```text
third_party/LibFewShot/config/classifiers/FroFAProto.yaml
third_party/LibFewShot/core/model/metric/frofa_proto_net.py
```

frozen backbone 设置还依赖 ResNet12 baseline 的最优权重：

```text
artifacts/runs/baseline/proto_cub_resnet12_boost_5way_1shot_cloud/checkpoints/emb_func_best.pth
artifacts/runs/baseline/proto_cub_resnet12_boost_5way_5shot_cloud/checkpoints/emb_func_best.pth
```

## 实验分层

| 层级 | 配置文件 | 作用 |
| :--- | :--- | :--- |
| joint FroFA | `frofa_proto_cub_resnet12_5way_*_cloud.yaml` | 端到端训练 ResNet12 + FroFAProtoNet |
| frozen FroFA | `frofa_proto_cub_resnet12_frozen_5way_*_cloud.yaml` | 加载 baseline backbone，冻结特征提取器后训练 FroFAProtoNet |
| summary | `summarize_frofa_results.py` | 汇总 joint FroFA、frozen FroFA 与 baseline 的对比结果 |

## 数据流

1. 准备 `data/CUB_200_2011` 数据目录。
2. 先完成 ResNet12 baseline，获得日志和 backbone 权重。
3. 选择 `configs/frofa/` 中的 FroFA 配置。
4. 通过 `scripts/frofa/run_frofa_cub_cloud.sh` 启动 joint、frozen 或 all 模式。
5. 控制台输出写入 `artifacts/logs/frofa/`。
6. LibFewShot 结果和 checkpoint 写入 `artifacts/runs/frofa/`。
7. 汇总 CSV 写入 `experiments/frofa_reproduction/results/`。
