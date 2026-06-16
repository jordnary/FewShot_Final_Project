# Baseline 实验运行指南

本文档面向需要复现实验的人员，默认从项目根目录 `FewShot_Final_Project/` 开始执行命令。

## 运行前检查

确认项目结构完整：

```text
FewShot_Final_Project/
├─ environment/requirements.txt
├─ third_party/LibFewShot/
├─ data/CUB_200_2011/
└─ experiments/baseline/
```

确认 CUB 数据目录至少包含：

```text
data/CUB_200_2011/
├─ images/
├─ train.csv
├─ val.csv
└─ test.csv
```

安装依赖：

```bash
pip install -r environment/requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 快速 baseline

快速 baseline 用于验证数据、配置和 LibFewShot 训练链路是否可运行，训练规模较小，不建议作为正式论文结果。

运行 5-way 1-shot：

```bash
python scripts/baseline/run_libfewshot_config.py configs/baseline/proto_cub_5way_1shot.yaml
```

运行 5-way 5-shot：

```bash
python scripts/baseline/run_libfewshot_config.py configs/baseline/proto_cub_5way_5shot.yaml
```

输出目录：

```text
artifacts/runs/baseline/proto_cub_5way_1shot
artifacts/runs/baseline/proto_cub_5way_5shot
```

## Conv64F 正式 baseline

该组实验使用 Conv64F backbone，训练轮数和 episode 数高于快速 baseline，适合作为轻量 backbone 对照。

```bash
bash scripts/baseline/run_proto_cub_conv64f_formal_cloud.sh
```

脚本会顺序运行：

```text
ProtoNet + Conv64F + CUB 5-way 1-shot
ProtoNet + Conv64F + CUB 5-way 5-shot
```

日志文件：

```text
artifacts/logs/baseline/proto_cub_conv64f_formal_5way_1shot_cloud_console.log
artifacts/logs/baseline/proto_cub_conv64f_formal_5way_5shot_cloud_console.log
```

结果目录：

```text
artifacts/runs/baseline/proto_cub_conv64f_formal_5way_1shot_cloud
artifacts/runs/baseline/proto_cub_conv64f_formal_5way_5shot_cloud
```

## ResNet12 增强 baseline

该组实验使用 ResNet12 backbone，是当前实验线建议优先引用的主 baseline。

```bash
bash scripts/baseline/run_proto_cub_boost_cloud.sh
```

等价于：

```bash
bash scripts/baseline/run_proto_cub_boost_cloud.sh resnet12
```

脚本会顺序运行：

```text
ProtoNet + ResNet12 + CUB 5-way 1-shot
ProtoNet + ResNet12 + CUB 5-way 5-shot
```

日志文件：

```text
artifacts/logs/baseline/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
artifacts/logs/baseline/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
```

结果目录：

```text
artifacts/runs/baseline/proto_cub_resnet12_boost_5way_1shot_cloud
artifacts/runs/baseline/proto_cub_resnet12_boost_5way_5shot_cloud
```

运行结束后，脚本会调用：

```bash
python scripts/baseline/summarize_boost_results.py
```

该命令会根据日志更新：

```text
experiments/baseline/results/baseline_boost_summary.csv
```

## 后台运行建议

云服务器上建议使用 `tmux`：

```bash
tmux new -s protonet-baseline
bash scripts/baseline/run_proto_cub_boost_cloud.sh
```

训练中断开终端但保留任务：按 `Ctrl+B`，再按 `D`。

恢复会话：

```bash
tmux attach -t protonet-baseline
```

## 注意事项

`run_proto_cub_boost_cloud.sh` 当前只支持 ResNet12 增强 baseline。若后续需要新增其他 backbone，应先补齐对应配置文件，再扩展脚本分支。

重新运行脚本会覆盖同名控制台日志，并可能更新同名结果目录中的 checkpoint。需要保留旧实验时，应先备份对应的 `artifacts/logs/baseline/` 和 `artifacts/runs/baseline/` 子目录。

如果 `torch.cuda.is_available()` 返回 `False`，正式实验仍可在 CPU 上启动，但速度会很慢，不建议用于完整复现。
