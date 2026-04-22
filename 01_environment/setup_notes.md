# 环境安装说明

本文件用于说明如何为项目创建 Python 环境并安装依赖。项目依赖统一维护在 `01_environment/requirements.txt` 中，其他目录不再单独保存依赖清单。

## 基础要求

- 建议使用 Conda 或 Miniconda 管理 Python 环境。
- 建议使用 Python 3.10 及以上版本；如果没有特殊兼容性要求，可以使用 Python 3.12。
- 如需运行 GPU 训练或 CLIP 特征提取，建议使用带 NVIDIA GPU 的环境，并安装与本机 CUDA 驱动匹配的 PyTorch。

## 创建环境

下面的环境名 `fewshot` 仅作示例，可以根据需要自行修改：

```bash
conda create -n fewshot python=3.12 -y
conda activate fewshot
```

## 安装依赖

在项目根目录执行：

```bash
pip install -r 01_environment/requirements.txt
```

如果需要指定 CUDA 版本，建议先根据 PyTorch 官方安装页面安装匹配的 `torch` 和 `torchvision`，再安装本项目其余依赖。

## 依赖包说明

- `numpy`：数组计算和特征数据处理的基础库。
- `pandas`：读取、整理和导出实验结果表格。
- `scipy`：科学计算工具，部分评估和统计处理会用到。
- `scikit-learn`：传统机器学习工具库，用于指标计算或简单分类评估。
- `matplotlib`：绘制结果图表。
- `Pillow`：图像读取和预处理。
- `PyYAML`：读取 YAML 配置文件。
- `einops`：简洁地重排张量维度。
- `future`：兼容部分旧式 Python 代码依赖。
- `rich`：更友好的命令行输出。
- `tensorboard`：记录和查看训练过程指标。
- `torch`：PyTorch 深度学习框架。
- `torchvision`：PyTorch 图像模型、数据处理和视觉工具。
- `open_clip_torch`：加载 OpenCLIP 模型，用于 CLIP 相关特征提取实验。

## 验证安装

安装完成后，可以用下面的命令确认 PyTorch 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出的第二项为 `True`，表示当前环境可以使用 CUDA；如果为 `False`，表示当前 PyTorch 环境只能使用 CPU。
