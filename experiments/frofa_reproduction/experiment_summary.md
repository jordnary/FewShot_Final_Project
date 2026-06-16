# FroFA 复现实验过程与结论

本实验线目标是在 CUB few-shot 分类任务上复现 FroFA-style feature augmentation 思路，并将其接入 LibFewShot 的 episodic ProtoNet 流程中，与 ProtoNet-ResNet12 baseline 对比。

## 实验设置

所有实验均采用 5-way few-shot 分类设置，分别评估 1-shot 和 5-shot。数据集为 CUB，backbone 为 ResNet12，对比对象为 训练充分的 ProtoNet-ResNet12 baseline。

| 实验 | 方法 | Backbone | 训练设置 | 测试 episodes | 说明 |
| :--- | :--- | :--- | :--- | ---: | :--- |
| baseline | ProtoNet | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 | 来自 baseline 实验线 |
| joint FroFA | FroFAProtoNet | ResNet12 | 120 epochs; 2000 train episodes/epoch | 1000 | backbone 与 FroFAProtoNet 端到端训练 |
| frozen FroFA | FroFAProtoNet | frozen ResNet12 | 30 epochs; 2000 train episodes/epoch | 1000 | 加载 baseline backbone 并冻结 |

FroFAProtoNet 使用 brightness 和 contrast 两类增强，`alpha=0.2`，每个 support 样本生成 2 个增强样本，距离度量为 euclidean，并启用 learnable scale。

## 实验过程

第一步先完成 ProtoNet-ResNet12 baseline，得到对比用日志和 frozen 设置需要的 `emb_func_best.pth`。

第二步运行 joint FroFA。该设置让 ResNet12 backbone 与 FroFAProtoNet 一起训练，用于验证 FroFA-style augmentation 是否能在当前 episodic pipeline 中直接带来收益。

第三步运行 frozen FroFA。该设置加载 baseline 的 backbone 权重并冻结特征提取器，只训练 FroFAProtoNet 相关部分，用于观察在固定 ResNet12 特征空间中加入 FroFA-style support augmentation 的效果。

第四步运行 `summarize_frofa_results.py`，从 FroFA 日志中提取最终测试 Acc@1，为每次实验生成独立的 `.csv`，并生成统一总表。

## 实验结果

| 方法 | 数据集 | 5-way 1-shot | 5-way 5-shot | 相对 baseline |
| :--- | :--- | ---: | ---: | :--- |
| ProtoNet-ResNet12 baseline | CUB | 73.376% | 85.945% | baseline |
| FroFAProtoNet-ResNet12 | CUB | 74.004% | 85.997% | 1-shot +0.628; 5-shot +0.052 |
| FroFAProtoNet-ResNet12 frozen | CUB | 73.324% | 85.805% | 1-shot -0.052; 5-shot -0.140 |

## 数据来源

每次实验 CSV 结果文件和最终总结：

```text
experiments/frofa_reproduction/results/frofa_proto_cub_resnet12_5way_1shot_cloud.csv
experiments/frofa_reproduction/results/frofa_proto_cub_resnet12_5way_5shot_cloud.csv
experiments/frofa_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_1shot_cloud.csv
experiments/frofa_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_5shot_cloud.csv
experiments/frofa_reproduction/results/frofa_vs_baseline_summary.csv
```

日志来源：

```text
artifacts/logs/baseline/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
artifacts/logs/baseline/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
artifacts/logs/frofa/frofa_proto_cub_resnet12_5way_1shot_cloud_console.log
artifacts/logs/frofa/frofa_proto_cub_resnet12_5way_5shot_cloud_console.log
artifacts/logs/frofa/frofa_proto_cub_resnet12_frozen_5way_1shot_cloud_console.log
artifacts/logs/frofa/frofa_proto_cub_resnet12_frozen_5way_5shot_cloud_console.log
```

## 结论

joint FroFA 相比 ProtoNet-ResNet12 baseline 有轻微提升：1-shot 提升约 0.628 个百分点，5-shot 提升约 0.052 个百分点。这说明 FroFA-style feature augmentation 已经可以接入 LibFewShot 的 ProtoNet episodic 训练流程，并在当前设置下带来小幅正向效果。

frozen FroFA 相比 baseline 略低：1-shot 下降约 0.052 个百分点，5-shot 下降约 0.140 个百分点。这说明仅冻结由 CUB episodic baseline 训练得到的 ResNet12 backbone，并不足以稳定放大 FroFA-style augmentation 的收益。

总体来看，本实验线完成了论文复现链路的接入验证，但提升幅度有限。一个重要原因是当前 frozen 设置使用的是任务内训练得到的 ResNet12 特征，而不是 FroFA 原思路中更强调的强预训练 frozen feature space。后续改进应优先考虑更强的 frozen representation，尤其是 ViT/CLIP patch-token 特征，并在 patch-token 层面做 FroFA-style augmentation。
