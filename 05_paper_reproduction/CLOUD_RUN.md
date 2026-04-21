# 云端运行 FroFA 复现

## 目标

运行 FroFA-style feature augmentation + ProtoNet + ResNet12 在 CUB 上的：

- 5-way 1-shot
- 5-way 5-shot

并和 `04_baseline_experiments` 中的 ProtoNet-ResNet12 baseline 日志汇总到同一张表。

## 运行

```bash
cd FewShot_Final_Project
chmod +x 05_paper_reproduction/run_frofa_cub_cloud.sh
bash 05_paper_reproduction/run_frofa_cub_cloud.sh
```

只运行 frozen backbone 版本：

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh frozen
```

一次运行 joint training 和 frozen backbone 两组：

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh all
```

如果 baseline 还没有跑，请先运行：

```bash
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh resnet12
```

## 输出

控制台日志：

```text
05_paper_reproduction/logs/frofa_proto_cub_resnet12_5way_1shot_cloud_console.log
05_paper_reproduction/logs/frofa_proto_cub_resnet12_5way_5shot_cloud_console.log
05_paper_reproduction/logs/frofa_proto_cub_resnet12_frozen_5way_1shot_cloud_console.log
05_paper_reproduction/logs/frofa_proto_cub_resnet12_frozen_5way_5shot_cloud_console.log
```

LibFewShot 结果目录：

```text
05_paper_reproduction/results/frofa_proto_cub_resnet12_5way_1shot_cloud
05_paper_reproduction/results/frofa_proto_cub_resnet12_5way_5shot_cloud
05_paper_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_1shot_cloud
05_paper_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_5shot_cloud
```

汇总表：

```text
05_paper_reproduction/results/frofa_vs_baseline_summary.md
05_paper_reproduction/results/frofa_vs_baseline_summary.csv
```
