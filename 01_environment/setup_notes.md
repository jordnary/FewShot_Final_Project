# 阶段 1：环境确认记录

日期：2026-04-20  
工作目录：`D:\Codes\Python\Machine_Learning\FewShot_Final_Project`

## Conda / Python

使用 PowerShell 函数 `cda` 初始化 Miniconda：

```powershell
cda
conda activate ml
python --version
```

结果：

```text
Miniconda Initialized!
Python 3.12.13
```

备注：每次启动 PowerShell 时会出现 `oh-my-posh` init script 写入失败：

```text
Failed to write init script: open C:\Users\jordn\AppData\Local\oh-my-posh\init...ps1: Access is denied.
```

该提示未阻止 `cda` 初始化 Miniconda，也未阻止 `ml` 环境运行 Python。

## PyTorch / CUDA

原始命令：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

首次运行报错：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

临时验证命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

结果：

```text
2.11.0+cu130
True
```

结论：`ml` 环境可以运行 PyTorch，且 CUDA 可用。OpenMP 重复加载问题需要后续清理依赖来源；当前可用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为临时 workaround。
