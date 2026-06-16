# Demo Checklist

演示前建议确认：

- `README.md` 能清楚解释项目目标和目录结构。
- `docs/runbook.md` 中的运行顺序与实际脚本一致。
- `experiments/*/results/` 中的核心 CSV 可以打开。
- `report/final_report/report.md` 已整理主要结论。
- `artifacts/pretrained/` 中已放置需要的本地 CLIP 权重，或环境可以联网下载。

轻量检查命令：

```powershell
.\demo\demo_run.ps1
```
