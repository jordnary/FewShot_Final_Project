# 项目运行手册

本文档给出面向报告复查和项目演示的推荐运行顺序。默认所有命令都从项目根目录执行。

## 准备环境

```bash
pip install -r environment/requirements.txt
```

CUB 数据应放在：

```text
data/CUB_200_2011/
├─ images/
├─ train.csv
├─ val.csv
└─ test.csv
```

## 推荐复现实验顺序

先运行 ProtoNet baseline：

```bash
bash scripts/baseline/run_proto_cub_boost_cloud.sh
```

再运行 FroFA 复现实验：

```bash
bash scripts/frofa/run_frofa_cub_cloud.sh all
```

最后运行 CLIP-FroFA 改进实验：

```bash
bash scripts/clip_frofa/run_patch_frofa_paperlike_cloud.sh
```

## 输出位置

可提交的小型 CSV 汇总位于：

```text
experiments/baseline/results/
experiments/frofa_reproduction/results/
experiments/clip_frofa_improvement/results/
```

本地运行产物位于：

```text
artifacts/logs/
artifacts/runs/
artifacts/features/
artifacts/pretrained/
```

`artifacts/` 下内容默认不纳入 Git。

## 报告撰写建议

报告正文建议放在 `report/final_report/`，图表放在 `report/figures/` 和 `report/tables/`。实验解释优先引用 `experiments/*/experiment_summary.md`，结果数值优先引用 `experiments/*/results/*.csv`。
