# Baseline 实验过程与结论

本阶段目标是在 CUB few-shot 分类任务上建立可复现的 ProtoNet baseline，为后续论文复现和改进方法提供对照。

## 实验设置

所有实验均采用 5-way few-shot 分类设置，分别评估 1-shot 和 5-shot。数据集为 CUB，方法为 ProtoNet，主要变化来自训练规模和 backbone。

| 实验层级 | 方法 | 数据集 | Backbone | 训练设置 | 测试 episodes |
| :--- | :--- | :--- | :--- | :--- | ---: |
| 快速 baseline | ProtoNet | CUB | Conv64F | 10 epochs; 100 train episodes/epoch | 100 |
| Conv64F 正式 baseline | ProtoNet | CUB | Conv64F | 100 epochs; 1000 train episodes/epoch | 600 |
| ResNet12 增强 baseline | ProtoNet | CUB | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 |

## 实验过程

第一步先运行快速 baseline，确认 CUB 数据划分、LibFewShot 配置、训练循环、测试流程和结果目录均能正常工作。该组训练规模较小，只承担链路验证作用。

第二步运行 Conv64F 正式 baseline。该组在同一轻量 backbone 下增加训练 epoch、train episodes 和 test episodes，用来观察更充分训练后 baseline 的提升幅度。

第三步运行 ResNet12 增强 baseline。该组更换为表达能力更强的 ResNet12 backbone，并进一步增加训练 episode 数，作为后续论文复现和方法改进的主对照。

## 实验结果

| 实验层级 | 方法 | Backbone | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | ---: | ---: |
| 快速 baseline | ProtoNet | Conv64F | 24.133% | 37.413% |
| Conv64F 正式 baseline | ProtoNet | Conv64F | 36.249% | 69.584% |
| ResNet12 增强 baseline | ProtoNet | ResNet12 | 73.376% | 85.945% |

## 数据来源

| 实验层级 | 汇总文件 | 日志文件 |
| :--- | :--- | :--- |
| 快速 baseline | `results/summary_quick_baseline.csv` | `logs/proto_cub_5way_1shot_console.log`; `logs/proto_cub_5way_5shot_console.log` |
| Conv64F 正式 baseline | `results/summary_conv64f_formal_cloud.csv` | `logs/proto_cub_conv64f_formal_5way_1shot_cloud_console.log`; `logs/proto_cub_conv64f_formal_5way_5shot_cloud_console.log` |
| ResNet12 增强 baseline | `results/summary_resnet12_boost_cloud.csv` | `logs/proto_cub_resnet12_boost_5way_1shot_cloud_console.log`; `logs/proto_cub_resnet12_boost_5way_5shot_cloud_console.log` |

完整总表见：

```text
04_baseline_experiments/results/summary_all.csv
```

## 结论

快速 baseline 成功验证了训练和测试链路，但由于训练规模较小，结果明显偏低，不适合作为正式对比。

Conv64F 正式 baseline 相比快速 baseline 有明显提升。1-shot 从 24.133% 提升到 36.249%，提升 12.116 个百分点；5-shot 从 37.413% 提升到 69.584%，提升 32.171 个百分点。这说明训练轮数、训练 episode 数和测试 episode 数对结果稳定性和最终精度有重要影响。

ResNet12 增强 baseline 是当前阶段最强结果。相对 Conv64F 正式 baseline，1-shot 从 36.249% 提升到 73.376%，提升 37.127 个百分点；5-shot 从 69.584% 提升到 85.945%，提升 16.361 个百分点。这说明更强 backbone 对 CUB few-shot 分类任务的贡献非常明显。

后续实验以 ResNet12 增强 baseline 作为主对照，以 Conv64F 正式 baseline 作为轻量 backbone 补充对照。快速 baseline 只用于调试和链路验证。
