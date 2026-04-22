# 云端运行 ProtoNet + Conv64F + CUB

## 目录要求

上传整个项目到云服务器后，保持结构：

```text
FewShot_Final_Project/
  02_libfewshot/LibFewShot/
  03_datasets/CUB_200_2011/
    images/
    train.csv
    val.csv
    test.csv
  04_baseline_experiments/
```

## 安装依赖

```bash
cd FewShot_Final_Project
pip install -r 01_environment/requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 运行正式 baseline

```bash
cd FewShot_Final_Project
chmod +x 04_baseline_experiments/run_proto_cub_conv64f_formal_cloud.sh
bash 04_baseline_experiments/run_proto_cub_conv64f_formal_cloud.sh
```

脚本会顺序运行：

```text
ProtoNet + Conv64F + CUB 5-way 1-shot
ProtoNet + Conv64F + CUB 5-way 5-shot
```

## 日志和结果

控制台日志：

```text
04_baseline_experiments/logs/proto_cub_conv64f_formal_5way_1shot_cloud_console.log
04_baseline_experiments/logs/proto_cub_conv64f_formal_5way_5shot_cloud_console.log
```

LibFewShot 结果目录：

```text
04_baseline_experiments/results/proto_cub_conv64f_formal_5way_1shot_cloud
04_baseline_experiments/results/proto_cub_conv64f_formal_5way_5shot_cloud
```

## 后台运行

如果服务器支持 `tmux`：

```bash
tmux new -s protonet-cub
bash 04_baseline_experiments/run_proto_cub_conv64f_formal_cloud.sh
```

退出但不中断任务：按 `Ctrl+B`，再按 `D`。

恢复：

```bash
tmux attach -t protonet-cub
```
