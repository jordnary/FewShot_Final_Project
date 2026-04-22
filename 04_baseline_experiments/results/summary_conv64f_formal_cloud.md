# Conv64F Formal Cloud Summary

| 类别 | 方法 | 数据集 | Backbone | 训练设置 | 测试 episodes | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| 云端正式 baseline | ProtoNet | CUB | Conv64F | 100 epochs; 1000 train episodes/epoch | 600 | 36.249% | 69.584% |

## 文件对应

| Shot | Config | Log | Result |
| :--- | :--- | :--- | :--- |
| 1-shot | `proto_cub_conv64f_formal_5way_1shot_cloud.yaml` | `proto_cub_conv64f_formal_5way_1shot_cloud_console.log` | `proto_cub_conv64f_formal_5way_1shot_cloud` |
| 5-shot | `proto_cub_conv64f_formal_5way_5shot_cloud.yaml` | `proto_cub_conv64f_formal_5way_5shot_cloud_console.log` | `proto_cub_conv64f_formal_5way_5shot_cloud` |

说明：该组是 Conv64F backbone 下更充分训练的正式 baseline，可作为轻量 backbone 对照。
