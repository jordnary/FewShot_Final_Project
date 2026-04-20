# 阶段 2：LibFewShot 跑通记录

日期：2026-04-20

## 仓库下载

目标目录：

```text
D:\Codes\Python\Machine_Learning\FewShot_Final_Project\02_libfewshot\LibFewShot
```

命令：

```powershell
git clone https://github.com/RL-VIG/LibFewShot.git LibFewShot
```

当前版本：

```text
067efb7 Merge pull request #128 from LeonaApera/pr
```

## 依赖安装

命令：

```powershell
cda
conda activate ml
pip install -r requirements.txt
```

安装的主要缺失依赖包括：

```text
PyYAML 6.0.3
einops 0.8.2
future 1.0.0
rich 15.0.0
tensorboard 2.20.0
```

## 数据说明

当前尚未下载真实 CUB 数据集。为了优先验证官方框架能跑通，临时创建了一个 LibFewShot CSV 格式的 synthetic sanity 数据集：

```text
03_datasets\synthetic_fewshot_sanity
```

结构：

```text
images/
train.csv
val.csv
test.csv
```

每个 split 有 5 类，每类 20 张 84x84 PNG，仅用于检查训练/验证/测试管线。

## 运行方法

使用官方已有方法 ProtoNet，配置基于：

```text
02_libfewshot\LibFewShot\config\test_install.yaml
```

实际运行命令：

```powershell
cda
conda activate ml
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python -c "from core.config import Config; from core import Trainer; cfg=Config('./config/test_install.yaml', {'data_root':'D:/Codes/Python/Machine_Learning/FewShot_Final_Project/03_datasets/synthetic_fewshot_sanity','train_episode':2,'test_episode':2,'epoch':1,'workers':0,'augment':False,'log_interval':1,'result_root':'./results/sanity_proto','log_name':'proto_sanity_run','n_gpu':1,'device_ids':'0'}).get_config_dict(); trainer=Trainer(0,cfg); trainer.train_loop(0)"
```

输出目录：

```text
02_libfewshot\LibFewShot\results\sanity_proto\proto_sanity_run
```

关键输出：

```text
load 100 train image with 5 label.
load 100 val image with 5 label.
load 100 test image with 5 label.
Epoch-(0): [1/2] ... Loss 0.000 ... Acc@1 100.000
Epoch-(0): [2/2] ... Loss 0.000 ... Acc@1 100.000
* Acc@1 100.000
Validation ... Acc@1 100.000
Testing ... Acc@1 96.667
End of experiment, took 0:00:01
```

完成标准：已能看到 ProtoNet 的训练 loss 和 accuracy 输出。

## 报错与解决

1. 网络受限导致首次 `git clone` 失败：

```text
Failed to connect to github.com port 443
```

解决：授权网络访问后重新执行 `git clone`。

2. `pip install -r requirements.txt` 首次因网络权限失败：

```text
WinError 10013
```

解决：授权 pip 网络访问后安装成功。

3. PyTorch 导入时 OpenMP runtime 重复：

```text
OMP: Error #15: Initializing libiomp5md.dll
```

解决：本次验证使用临时环境变量：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

4. Python 3.12 兼容问题：

```text
ImportError: cannot import name 'Iterable' from 'collections'
```

解决：将 `core\data\collates\collate_functions.py` 中的导入改为：

```python
from collections.abc import Iterable
```

5. Pandas 3 兼容问题：

```text
ValueError: assignment destination is read-only
```

解决：将 `core\utils\utils.py` 中 `AverageMeter` 的 DataFrame 更新改为 `.loc` 写法。

6. PyTorch 新版 Sampler 初始化兼容问题：

```text
TypeError: object.__init__() takes exactly one argument
```

解决：将 `core\data\samplers.py` 中 `Sampler` 父类初始化改为无参 `super(...).__init__()`。

## 后续

下一步需要下载并整理真实 CUB 数据集到 LibFewShot 期望格式，再把 `data_root` 切换到 CUB 路径，运行正式 ProtoNet baseline。
