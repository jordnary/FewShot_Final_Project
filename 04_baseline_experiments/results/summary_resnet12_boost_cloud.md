# ResNet12 Boost Cloud Summary

| 类别 | 方法 | 数据集 | Backbone | 训练设置 | 测试 episodes | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| 云端增强 baseline | ProtoNet | CUB | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 | 73.376% | 85.945% |

## 文件对应

| Shot | Config | Log | Result |
| :--- | :--- | :--- | :--- |
| 1-shot | `proto_cub_resnet12_boost_5way_1shot_cloud.yaml` | `proto_cub_resnet12_boost_5way_1shot_cloud_console.log` | `proto_cub_resnet12_boost_5way_1shot_cloud` |
| 5-shot | `proto_cub_resnet12_boost_5way_5shot_cloud.yaml` | `proto_cub_resnet12_boost_5way_5shot_cloud_console.log` | `proto_cub_resnet12_boost_5way_5shot_cloud` |

说明：该组是当前阶段最强 baseline，建议作为后续复现和改进方法的主对照。
