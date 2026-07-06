# Demo Checklist

演示前请先过一遍：

- `README.md` 已能清楚解释项目目标和目录结构。
- `docs/runbook.md` 中的运行顺序与实际脚本一致。
- `experiments/*/results/` 中的核心 CSV 能正常打开。
- `report/final_report/report.md` 已整理主要结论。
- `artifacts/pretrained/` 中已放置需要的本地 CLIP 权重，或当前环境能联网下载。

可以使用下述命令进行轻量检查：

```powershell
.\demo\demo_run.ps1
```
