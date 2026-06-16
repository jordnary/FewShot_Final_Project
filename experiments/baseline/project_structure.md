# Baseline 实验结构

本实验线用于管理 CUB 数据集上的 ProtoNet baseline，包括快速链路验证、Conv64F 正式 baseline 和 ResNet12 增强 baseline。

## 相关目录

```text
configs/baseline/                 # LibFewShot YAML 配置
scripts/baseline/                 # Baseline 运行和汇总入口
experiments/baseline/             # 实验说明与可提交 CSV 结果
artifacts/logs/baseline/          # 控制台日志，不纳入 Git
artifacts/runs/baseline/          # LibFewShot 输出、checkpoint，不纳入 Git
```

## 文件职责

`configs/baseline/` 保存 ProtoNet baseline 的 YAML 配置。所有项目配置使用 `${PROJECT_ROOT}` 占位符，运行时由 `run_libfewshot_config.py` 或 `run_libfewshot_config_cloud.py` 渲染为当前项目路径。

`scripts/baseline/` 保存运行入口。`run_libfewshot_config.py` 和 `run_libfewshot_config_cloud.py` 负责调用 LibFewShot；`run_proto_cub_conv64f_formal_cloud.sh` 与 `run_proto_cub_boost_cloud.sh` 是成组实验入口；`summarize_boost_results.py` 从日志中提取最终测试精度。

`experiments/baseline/` 保存本实验线文档和小型 CSV 结果。`results/summary_all.csv` 是报告中最常引用的总表。

`artifacts/logs/baseline/` 保存控制台日志，`artifacts/runs/baseline/` 保存 LibFewShot 训练输出和 checkpoint。

## 实验分层

| 层级 | 配置文件 | 作用 |
| :--- | :--- | :--- |
| 快速 baseline | `proto_cub_5way_*.yaml` | 验证训练、测试和日志链路可运行 |
| Conv64F 正式 baseline | `proto_cub_conv64f_formal_*_cloud.yaml` | 提供轻量 backbone 的正式对照 |
| ResNet12 增强 baseline | `proto_cub_resnet12_boost_*_cloud.yaml` | 提供主对照和最强 baseline |

## 数据流

1. 准备 `data/CUB_200_2011` 数据目录。
2. 选择 `configs/baseline/` 中的 YAML 配置。
3. 通过 `scripts/baseline/` 中的 runner 调用 LibFewShot。
4. 控制台输出写入 `artifacts/logs/baseline/`。
5. 训练结果和 checkpoint 写入 `artifacts/runs/baseline/`。
6. 汇总表写入 `experiments/baseline/results/`。
