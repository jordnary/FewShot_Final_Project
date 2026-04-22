# Baseline Summary

| 类别 | 方法 | 数据集 | Backbone | 训练设置 | 测试 episodes | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| 快速可运行 baseline | ProtoNet | CUB | Conv64F | 10 epochs; 100 train episodes/epoch | 100 | 24.133% | 37.413% |
| 云端正式 baseline | ProtoNet | CUB | Conv64F | 100 epochs; 1000 train episodes/epoch | 600 | 36.249% | 69.584% |
| 云端增强 baseline | ProtoNet | CUB | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 | 73.376% | 85.945% |

## 简要结论

- 当前最佳结果来自云端增强 baseline：ProtoNet + ResNet12，5-way 1-shot 为 73.376%，5-way 5-shot 为 85.945%。
- 云端正式 Conv64F baseline 相对快速 baseline 提升明显，说明正式训练设置是必要的。
- 后续方法对比建议以 ResNet12 boost 为主，以 Conv64F formal cloud 为轻量 backbone 对照。
