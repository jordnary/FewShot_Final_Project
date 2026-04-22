# 04_baseline_experiments 项目结构

本目录用于管理 CUB 数据集上的 ProtoNet baseline 实验，包括快速链路验证、Conv64F 正式 baseline 和 ResNet12 增强 baseline。

## 目录概览

```text
04_baseline_experiments/
├─ configs/
│  ├─ proto_cub_5way_1shot.yaml
│  ├─ proto_cub_5way_5shot.yaml
│  ├─ proto_cub_conv64f_formal_5way_1shot_cloud.yaml
│  ├─ proto_cub_conv64f_formal_5way_5shot_cloud.yaml
│  ├─ proto_cub_resnet12_boost_5way_1shot_cloud.yaml
│  └─ proto_cub_resnet12_boost_5way_5shot_cloud.yaml
├─ docs/
│  ├─ project_structure.md
│  ├─ run_guide.md
│  └─ experiment_summary.md
├─ logs/
├─ results/
├─ run_libfewshot_config.py
├─ run_libfewshot_config_cloud.py
├─ run_proto_cub_conv64f_formal_cloud.sh
├─ run_proto_cub_boost_cloud.sh
└─ summarize_boost_results.py
```

## 文件职责

`configs/` 保存 LibFewShot 的 YAML 配置。快速 baseline 使用本地绝对路径配置；云端配置使用 `${PROJECT_ROOT}` 占位符，运行时由 `run_libfewshot_config_cloud.py` 渲染为当前项目路径。

`docs/` 保存该阶段的说明文档。`project_structure.md` 说明目录结构，`run_guide.md` 面向他人说明如何复现实验，`experiment_summary.md` 归纳实验过程、数据和结论。

`logs/` 保存控制台日志，例如 `*_console.log`。这些文件主要用于追溯训练和测试过程，一般属于本地或云端运行产物。

`results/` 保存 LibFewShot 输出目录、checkpoint 和人工整理后的 CSV 汇总表。`summary_all.csv` 是当前最核心的结果总表。

`run_libfewshot_config.py` 是本地运行入口，直接读取配置文件。它适合快速 baseline 或本机调试。

`run_libfewshot_config_cloud.py` 是云端运行入口，会先把配置中的 `${PROJECT_ROOT}` 替换为当前项目根目录，再调用 LibFewShot 的 `Trainer`。

`run_proto_cub_conv64f_formal_cloud.sh` 顺序运行 Conv64F 正式 baseline 的 1-shot 和 5-shot 实验。

`run_proto_cub_boost_cloud.sh` 默认运行 ResNet12 增强 baseline 的 1-shot 和 5-shot 实验，并在结束后调用 `summarize_boost_results.py` 汇总结果。

`summarize_boost_results.py` 从增强 baseline 的日志中提取最终测试精度，生成 `baseline_boost_summary.csv`。

## 实验分层

| 层级 | 配置文件 | 作用 |
| :--- | :--- | :--- |
| 快速 baseline | `proto_cub_5way_*.yaml` | 验证训练、测试和日志链路可运行 |
| Conv64F 正式 baseline | `proto_cub_conv64f_formal_*_cloud.yaml` | 提供轻量 backbone 的正式对照 |
| ResNet12 增强 baseline | `proto_cub_resnet12_boost_*_cloud.yaml` | 提供当前阶段主对照和最强 baseline |

## 数据流

1. 准备 `03_datasets/CUB_200_2011` 数据目录。
2. 选择 `configs/` 中的 YAML 配置。
3. 通过本地或云端 runner 调用 LibFewShot。
4. 控制台输出写入 `logs/`。
5. 训练结果、checkpoint 和 LibFewShot 日志写入 `results/`。
6. 汇总表记录到 `results/summary_*.csv`。
