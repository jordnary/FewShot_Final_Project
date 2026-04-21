## 推荐命令

```bash
cd FewShot_Final_Project
chmod +x 04_baseline_experiments/run_proto_cub_boost_cloud.sh
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh resnet12
```

如果还想和原 Conv64F baseline 公平比较：

```bash
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh conv64f
```

一次全跑：

```bash
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh all
```

## 后台运行

```bash
tmux new -s protonet-boost
bash 04_baseline_experiments/run_proto_cub_boost_cloud.sh resnet12
```

退出但不中断：按 `Ctrl+B`，再按 `D`。

恢复：

```bash
tmux attach -t protonet-boost
```

## 判断是否跑完

```bash
tail -n 60 04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
tail -n 60 04_baseline_experiments/logs/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
```

看到 `End of experiment`，并且最后一次 `Testing on the test set` 后有 `* Acc@1 xx.xxx`，就是完成。

汇总表会写到：

```text
04_baseline_experiments/results/baseline_boost_summary.md
04_baseline_experiments/results/baseline_boost_summary.csv
```

也可以手动重新生成：

```bash
python 04_baseline_experiments/summarize_boost_results.py
```

## GPU 仍未跑满时

当前配置已把 `episode_size` 提到 4、`workers` 提到 16。少样本学习的 episode 训练本身计算块较碎，GPU 利用率不一定长期 100%。如果显存占用很低，可以先把两个 ResNet12 配置里的：

```yaml
episode_size: 4
train_episode: 2000
test_episode: 1000
```

改成：

```yaml
episode_size: 8
train_episode: 4000
test_episode: 1200
```

注意 `train_episode` 和 `test_episode` 必须能被 `episode_size` 整除。若出现 CUDA OOM，再改回 4。
