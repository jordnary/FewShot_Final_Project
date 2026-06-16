# Artifacts

本目录用于保存本地或云端运行产物，包括日志、checkpoint、特征缓存和预训练权重。

这些文件通常体积较大，或可以从脚本重新生成，因此默认不纳入 Git。推荐子目录如下：

```text
artifacts/
├─ logs/
├─ runs/
├─ features/
└─ pretrained/
```

可提交的小型结果表应放在 `experiments/*/results/`。
