# 阶段 04：Baseline 实验整理与结果归纳

整理日期：2026-04-21

## 目录归档

```text
04_baseline_experiments/
├─ configs/   实验配置文件
├─ logs/      云端控制台日志，本地保留但不纳入 Git
├─ results/   结果汇总纳入 Git；逐次 run 输出和 checkpoint 本地保留
├─ *_RUN.md   云端运行说明
└─ run_*.py / run_*.sh / run_*.ps1 运行脚本
```

核心可引用文件如下：

- 结果总表：`results/summary_all.md`
- 结果总表 CSV：`results/summary_all.csv`
- 快速 baseline 汇总：`results/summary_quick_baseline.md`
- Conv64F 云端正式 baseline 汇总：`results/summary_conv64f_formal_cloud.md`
- ResNet12 云端增强 baseline 汇总：`results/summary_resnet12_boost_cloud.md`
- 云端控制台日志：`logs/*_cloud_console.log`（本地训练产物）
- 最优模型权重：各实验目录下的 `checkpoints/model_best.pth` 与 `checkpoints/emb_func_best.pth`（本地训练产物）

## 实验结果

| 实验层级 | 方法 | Backbone | 训练设置 | 测试 episodes | 5-way 1-shot | 5-way 5-shot |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| 快速可运行 baseline | ProtoNet | Conv64F | 10 epoch, 100 train episodes/epoch | 100 | 24.133% | 37.413% |
| 云端正式 Conv64F baseline | ProtoNet | Conv64F | 100 epoch, 1000 train episodes/epoch | 600 | 36.249% | 69.584% |
| 云端增强 baseline | ProtoNet | ResNet12 | 120 epoch, 2000 train episodes/epoch | 1000 | 73.376% | 85.945% |

## 结论

1. 快速 baseline 只用于验证训练链路可运行，结果偏低，不建议作为论文级对比结果。
2. 云端正式 Conv64F baseline 在训练轮数和 episode 数增加后明显提升：相对快速 baseline，1-shot 提升 12.116 个百分点，5-shot 提升 32.171 个百分点。
3. ResNet12 boost 是当前阶段最强 baseline：相对云端正式 Conv64F，1-shot 提升 37.127 个百分点，5-shot 提升 16.361 个百分点。
4. 后续论文复现或改进方法应优先对比 ResNet12 boost 这一组结果；Conv64F formal 可作为轻量 backbone 的补充对照。

## 日志来源

```text
快速 1-shot: logs/proto_cub_5way_1shot_console.log
快速 5-shot: logs/proto_cub_5way_5shot_console.log
Conv64F formal 1-shot: logs/proto_cub_conv64f_formal_5way_1shot_cloud_console.log
Conv64F formal 5-shot: logs/proto_cub_conv64f_formal_5way_5shot_cloud_console.log
ResNet12 boost 1-shot: logs/proto_cub_resnet12_boost_5way_1shot_cloud_console.log
ResNet12 boost 5-shot: logs/proto_cub_resnet12_boost_5way_5shot_cloud_console.log
```
