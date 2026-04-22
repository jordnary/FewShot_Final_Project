# 阶段 6：个人思考与改进分析

## 改进目标

前 4-5 阶段已经完成了 CUB 上的 LibFewShot baseline 和 FroFA-style ProtoNet 接入。第 6 阶段进一步采用更贴近 FroFA 原论文的设置：

```text
CUB images
-> pretrained CLIP ViT-B/16 frozen encoder
-> offline patch-token cache
-> no-FroFA MAP head
-> FroFA augmented patch tokens + MAP head
-> compare 5-way 1-shot / 5-shot
```

这样做的原因是 FroFA 的核心假设不是“重新训练一个小 backbone”，而是“大规模预训练视觉模型已经提供强冻结特征，少样本阶段只训练轻量头”。因此用 CLIP ViT-B/16 比继续使用 Conv64F 或 ResNet12 更能体现论文思想。

## 文件说明

- `extract_clip_features.py`：冻结 CLIP ViT-B/16，读取 `03_datasets/CUB_200_2011/{train,val,test}.csv`，离线提取并保存特征。
- `run_frofa_linear_eval.py`：在缓存特征上采样 episode，比较 no-FroFA 与 FroFA 的闭式 L2 linear probe。
- `extract_clip_patch_tokens.py`：冻结 CLIP ViT-B/16，离线提取最终层 patch tokens，默认输出 test split。
- `run_frofa_map_eval.py`：在 patch tokens 上按 episode 训练 MAP head，对比 no-FroFA MAP 与 FroFA + MAP。
- `run_clip_frofa_cub_cloud.sh`：云端一键运行特征提取和 5-way 1-shot / 5-shot 对比。
- `ablation_study.md`：个人思考、改进动机、预期结果和报告可用分析。

## 依赖

统一依赖放在 `01_environment/requirements.txt`，其中已经包含 `open_clip_torch`、`torch` 和 `torchvision`：

```bash
pip install -r 01_environment/requirements.txt
```

云端脚本默认使用 `open_clip` 后端，并设置 HuggingFace 镜像，避免直接访问 HuggingFace 失败：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果镜像仍不可用，可以改用非 CLIP 的兜底 frozen ViT-B/16：

```bash
python 06_improvement/extract_clip_features.py \
  --data-root 03_datasets/CUB_200_2011 \
  --output-dir 06_improvement/results/features \
  --output-prefix torchvision_vit_b16 \
  --backend torchvision_vit
```

这个兜底不等同于 CLIP，但仍满足“强 frozen ViT features + FroFA linear classifier”的改进路线。

也可以本地下载 CLIP 权重后上传云端。推荐把权重放在：

```text
06_improvement/pretrained/open_clip_model.safetensors
```

然后云端运行：

```bash
export CLIP_PRETRAINED=06_improvement/pretrained/open_clip_model.safetensors
bash 06_improvement/run_clip_frofa_cub_cloud.sh
```

`CLIP_PRETRAINED` 不设置时，脚本默认仍会尝试 `open_clip` 的 `openai` 权重名。

## 运行

```bash
cd FewShot_Final_Project
chmod +x 06_improvement/run_clip_frofa_cub_cloud.sh
bash 06_improvement/run_clip_frofa_cub_cloud.sh
```

输出文件：

```text
06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz
06_improvement/results/clip_vit_b16_patch_frofa_map_cub.md
06_improvement/results/clip_vit_b16_patch_frofa_map_cub.csv
06_improvement/results/final_stage6_summary.md
```

`results/features/` 中的 `.npz` 是可复用特征缓存，属于本地训练产物，不纳入 Git；`results/*.md` 和 `results/*.csv` 是可跟踪的结果汇总。

## 报告中的结果表格

| 方法 | Frozen encoder | Head | 5-way 1-shot | 5-way 5-shot |
|---|---|---|---:|---:|
| MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 46.213 +/- 0.875 | 75.478 +/- 0.765 |
| FroFA + MAP | CLIP ViT-B/16 patch tokens | episode-trained MAP head | 45.402 +/- 0.890 | 77.056 +/- 0.819 |

当前结论：FroFA + MAP 在 5-shot 上有提升，但 1-shot 仍需调参；最终解释以 `results/final_stage6_summary.md` 为准。
