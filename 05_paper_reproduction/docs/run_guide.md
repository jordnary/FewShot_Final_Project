# FroFA 复现实验运行指南

本文档面向需要复现 05 阶段实验的人员，默认从项目根目录 `FewShot_Final_Project/` 开始执行命令。

## 运行前检查

确认项目结构完整：

```text
FewShot_Final_Project/
├─ 01_environment/requirements.txt
├─ 02_libfewshot/LibFewShot/
├─ 03_datasets/CUB_200_2011/
├─ 04_baseline_experiments/
└─ 05_paper_reproduction/
```

确认 CUB 数据目录至少包含：

```text
03_datasets/CUB_200_2011/
├─ images/
├─ train.csv
├─ val.csv
└─ test.csv
```

安装依赖：

```bash
pip install -r 01_environment/requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 前置 baseline

建议先完成 04 阶段 ResNet12 baseline：

```bash
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh
```

这一步会提供两类输入：

```text
04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
04_baseline_experiments/results/proto_cub_resnet12_boost_5way_1shot_cloud/checkpoints/emb_func_best.pth
04_baseline_experiments/results/proto_cub_resnet12_boost_5way_5shot_cloud/checkpoints/emb_func_best.pth
```

其中日志用于最终汇总对比，`emb_func_best.pth` 用于 frozen FroFA 配置加载预训练 backbone。

## 运行 joint FroFA

joint 模式会端到端训练 ResNet12 + FroFAProtoNet，是脚本默认模式。

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh
```

等价于：

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh joint
```

脚本会顺序运行：

```text
FroFAProtoNet + ResNet12 + CUB 5-way 1-shot
FroFAProtoNet + ResNet12 + CUB 5-way 5-shot
```

## 运行 frozen FroFA

frozen 模式会加载 04 阶段 ResNet12 baseline 的 backbone 权重，并冻结特征提取器。

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh frozen
```

脚本会顺序运行：

```text
FroFAProtoNet + frozen ResNet12 + CUB 5-way 1-shot
FroFAProtoNet + frozen ResNet12 + CUB 5-way 5-shot
```

如果缺少 04 阶段 checkpoint，该模式会因为 `pretrain_path` 找不到而失败。

## 一次运行全部 FroFA 设置

```bash
bash 05_paper_reproduction/run_frofa_cub_cloud.sh all
```

该模式会先运行 joint FroFA 的 1-shot 和 5-shot，再运行 frozen FroFA 的 1-shot 和 5-shot。

## 输出文件

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

每次实验 CSV 结果文件和最终总结：

```text
05_paper_reproduction/results/frofa_proto_cub_resnet12_5way_1shot_cloud.csv
05_paper_reproduction/results/frofa_proto_cub_resnet12_5way_5shot_cloud.csv
05_paper_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_1shot_cloud.csv
05_paper_reproduction/results/frofa_proto_cub_resnet12_frozen_5way_5shot_cloud.csv
05_paper_reproduction/results/frofa_vs_baseline_summary.csv
```

## 单独更新汇总

如果训练日志已经存在，只需要重新生成每次实验结果和 `frofa_vs_baseline_summary` 总结：

```bash
python 05_paper_reproduction/summarize_frofa_results.py
```

如果某个日志缺失，汇总表中对应结果会显示为 `missing`；如果日志没有完整结束，会显示为 `unfinished`。

## 后台运行建议

云服务器上建议使用 `tmux`：

```bash
tmux new -s frofa-repro
bash 05_paper_reproduction/run_frofa_cub_cloud.sh all
```

训练中断开终端但保留任务：按 `Ctrl+B`，再按 `D`。

恢复会话：

```bash
tmux attach -t frofa-repro
```

## 注意事项

重新运行脚本会覆盖同名控制台日志，并可能更新同名结果目录中的 checkpoint。需要保留旧实验时，应先备份对应的 `logs/` 和 `results/` 子目录。

正式复现建议使用 GPU。CPU 可以启动部分流程，但完整训练会非常慢。
