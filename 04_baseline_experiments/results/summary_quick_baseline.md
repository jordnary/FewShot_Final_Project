# Quick Baseline Summary

| 类别 | 方法 | 数据集 | Backbone | 训练设置 | 测试 episodes | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| 快速可运行 baseline | ProtoNet | CUB | Conv64F | 10 epochs; 100 train episodes/epoch | 100 | 24.133% | 37.413% |

## 文件对应

| Shot | Config | Log | Result |
| :--- | :--- | :--- | :--- |
| 1-shot | `proto_cub_5way_1shot.yaml` | `proto_cub_5way_1shot_console.log` | `proto_cub_5way_1shot` |
| 5-shot | `proto_cub_5way_5shot.yaml` | `proto_cub_5way_5shot_console.log` | `proto_cub_5way_5shot` |

说明：该组主要用于确认 baseline 链路可运行，不作为正式论文级结果。
