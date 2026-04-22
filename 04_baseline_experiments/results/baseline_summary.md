# 阶段 4：ProtoNet Baseline 结果

日期：2026-04-20

## 实验设置

- 方法：ProtoNet
- 数据集：CUB
- Backbone：Conv64F
- 图像尺寸：84
- 训练设置：快速 baseline，10 epoch，每个 epoch 100 train episodes
- 测试设置：100 test episodes
- 数据路径：`03_datasets/CUB_200_2011`

说明：该结果用于阶段性可运行 baseline，不是完整论文级复现实验。后续正式对比建议增加 train/test episode 数和训练 epoch。

## 结果表

| 方法 | 数据集 | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- |
| ProtoNet | CUB | 24.13% | 37.41% |

## 文件归档

配置：

```text
04_baseline_experiments/configs/proto_cub_5way_1shot.yaml
04_baseline_experiments/configs/proto_cub_5way_5shot.yaml
```

控制台日志：

```text
04_baseline_experiments/logs/proto_cub_5way_1shot_console.log
04_baseline_experiments/logs/proto_cub_5way_5shot_console.log
```

LibFewShot 输出：

```text
04_baseline_experiments/results/proto_cub_5way_1shot
04_baseline_experiments/results/proto_cub_5way_5shot
```

关键测试输出：

```text
5-way 1-shot: * Acc@1 24.133
5-way 5-shot: * Acc@1 37.413
```
