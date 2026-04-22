# 02_libfewshot 结构说明

本目录保存本项目使用和改造后的 LibFewShot 框架代码。它的定位是项目中的“少样本学习训练与评估内核”：数据集准备、baseline 实验、FroFA 复现和后续改进实验都会通过这里的 LibFewShot 入口、配置系统和模型实现来运行。

## 目录结构

```text
02_libfewshot/
├── README.md
└── LibFewShot/
    ├── run_trainer.py
    ├── run_trainer_resume.py
    ├── run_test.py
    ├── config/
    │   ├── headers/
    │   ├── backbones/
    │   ├── classifiers/
    │   └── *.yaml
    ├── core/
    │   ├── config/
    │   ├── data/
    │   ├── model/
    │   │   ├── backbone/
    │   │   ├── finetuning/
    │   │   ├── meta/
    │   │   └── metric/
    │   ├── trainer.py
    │   └── test.py
    └── reproduce/
```

### `LibFewShot/run_*.py`

训练、断点续训和测试入口。项目中的实验脚本通常不会直接重写训练循环，而是生成或指定 LibFewShot 配置文件，再调用这些入口完成训练与评估。

### `LibFewShot/config/`

LibFewShot 的 YAML 配置区。

- `headers/`：数据、设备、模型、优化器等通用配置片段。
- `backbones/`：Conv64F、ResNet12、ResNet18、ViT 等 backbone 配置。
- `classifiers/`：各类 few-shot classifier 的参数配置。
- 顶层 `*.yaml`：完整实验配置，通常通过 `includes` 组合 header、backbone 和 classifier。

### `LibFewShot/core/`

框架核心代码。

- `core/config/`：配置加载与合并逻辑。
- `core/data/`：数据集、episode sampler、dataloader 和增强 collate。
- `core/model/backbone/`：图像特征提取网络。
- `core/model/metric/`：ProtoNet、RelationNet、FEAT、FRN、FroFAProtoNet 等 metric-learning 方法。
- `core/model/meta/`：MAML、ANIL、BOIL、R2D2 等 meta-learning 方法。
- `core/model/finetuning/`：Baseline、Baseline++、RFS 等 finetuning 类方法。
- `core/trainer.py` 与 `core/test.py`：训练、验证、测试流程。

### `LibFewShot/reproduce/`

LibFewShot 官方/已有方法的复现实验配置，按方法名分目录保存。这里主要作为参考配置库，便于查看不同方法、数据集、backbone 和 shot 设置的组合方式。

## 本项目在此新增的文件

### `LibFewShot/config/frofa_proto.yaml`

FroFAProtoNet 的完整示例配置。它组合了通用 header、`classifiers/FroFAProto.yaml` 和默认 backbone，用于快速跑通 FroFA-style feature augmentation + ProtoNet 的 episodic few-shot 实验。

### `LibFewShot/config/classifiers/FroFAProto.yaml`

FroFAProtoNet 的 classifier 参数配置，包括：

- `augmentations`：启用的特征空间增强，如 `brightness`。
- `alpha`：增强强度。
- `num_aug`：每个 episode 中为 support feature 生成的增强次数。
- `distance`：query 与 prototype 的距离度量。
- `freeze_emb_func`：是否冻结 backbone。
- `learnable_scale`：是否学习 logits 缩放参数。

### `LibFewShot/core/model/metric/frofa_proto_net.py`

FroFA-style feature augmentation 的主要实现文件。它把 FroFA 的核心思想接入 LibFewShot 的 metric-learning 接口：

1. 提取 support/query features。
2. 将 support features 归一化到 `[0, 1]` 范围。
3. 在特征空间执行 `brightness`、`contrast`、`posterize` 等增强。
4. 将增强后的 support features 映射回原特征尺度。
5. 用原始和增强后的 support features 共同计算 class prototype。
6. 按 ProtoNet 风格对 query features 分类。

该文件支持冻结 backbone、欧氏距离/余弦相似度、可学习 logit scale 等选项，是本项目复现 FroFA 思路时在 LibFewShot 内部添加的核心模型代码。

## 本项目修改的已有文件

### `LibFewShot/core/model/metric/__init__.py`

注册 `FroFAProtoNet`，使配置文件中的 `classifier.name: FroFAProtoNet` 能被 LibFewShot 的模型构建流程找到。

### `LibFewShot/core/trainer.py`

做了两处运行兼容性调整：

- 非分布式训练时将模型移动到 `self.device`，避免单卡/CPU 场景中使用 rank 作为设备导致的问题。
- 在数据配置检查中使用 `effective_n_gpu = max(1, n_gpu)`，避免 `n_gpu` 为 0 或单设备配置时 episode 校验出错。

## 与项目其他目录的关系

- `03_datasets/` 提供 CUB 等数据和 split 文件。
- `04_baseline_experiments/` 使用本目录中的 LibFewShot 入口运行 ProtoNet baseline。
- `05_paper_reproduction/` 使用 `FroFAProtoNet` 相关配置复现 FroFA 风格方法。
- `06_improvement/` 在 LibFewShot 复现链路之外补充 CLIP frozen feature、patch-token MAP 和 FroFA 改进实验。

## 使用建议

如果只是运行实验，优先从 `04_baseline_experiments/`、`05_paper_reproduction/` 或 `06_improvement/` 的脚本进入；如果需要修改 few-shot 方法本身，再进入 `LibFewShot/core/model/` 和 `LibFewShot/config/`。
