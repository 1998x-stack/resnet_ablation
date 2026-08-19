# ResNet 项目构建

**User:** Anonymous (harbertparker90526@outlook.com)  
**Created:** 9/16/2025 16:20:30  
**Updated:** 9/16/2025 17:02:23  
**Exported:** 9/16/2025 17:02:26  
**Link:** [https://chatgpt.com/g/g-p-68c390f088dc81918596271e1836757f-deepsearch/c/68c91dcd-e3b8-832f-8bc1-ce3a1cfdd8a4](https://chatgpt.com/g/g-p-68c390f088dc81918596271e1836757f-deepsearch/c/68c91dcd-e3b8-832f-8bc1-ce3a1cfdd8a4)  

## Prompt:
2-ResNet.md

文件

2-ResNet.pdf

PDF

# ResNet Ablation 可插拔project构建

结合论文实验过程, Implement totally new ResNet project for industry enginner from scratch please

explore uploaded files the train data and resnet arch
then design train/test process
use torch,tensorboard, loguru
do not use canvas

give me code file by file

已思考 1m 53s

正在读取文档

好吧，我们需要从 2-ResNet.md/pdf 文档中引用设计决定或数据集的细节。对于代码部分，引用可以放在文本描述中，但不要在代码块内使用引用。指南说引用必须是内联的，不是放在结尾。这个和通常的格式有些不一样。明白了，用户想让我们从头开始实现 ResNet 项目，并且有一些具体的工具使用要求。

用户经常要求工业级别的包装、Google风格的文档字符串、中文注释和强健的代码。尽管没有明确要求双语，但他们的风格偏好指向中文注释和PEP-257文档字符串。我们将遵循这些要求，按项目结构逐个文件展示代码。还需要从上传文件中提取训练数据和ResNet架构信息。

正在读取文档

我需要考虑是否要加入 source\_filter 参数。说明文档中没有提到它，但现在工具却提示必须包含。尽管 API schema 没有明确列出它，但错误信息却要求加入 'files\_uploaded\_in\_conversation'。看来工具可能需要这个字段。所以，我将尝试添加 "intent":"" 和 "source\_filter":\["files\_uploaded\_in\_conversation"\]。

正在读取文档

好的，我决定去掉 "intent" 字段，只保留 "source\_filter"。这样做看起来能解决问题，也避免了不必要的字段干扰。

正在读取文档

我们得到了 2-ResNet.pdf 的内容，这似乎是 He 等人原始的 ResNet 论文，包括检测部分和一些 CIFAR-10 架构的细节以及错误表格。很好，不过 2-ResNet.md 文件内容没有被返回。所以，我打算调整查询，专门针对这个文件名。

正在读取文档

现在我有足够的内容来引用关键的 ResNet 架构细节和 CIFAR-10 数据集。我打算设计一个模块化项目，将代码按文件展示。首先需要考虑使用 `setup.py` 还是 `pyproject`，再提供项目树结构。目标是使用 PyTorch、TensorBoard 和 Loguru，配置可能会用 YAML。任务比较复杂，不需要问用户更多问题，只需尽力而为。

我要创建一个简单但有效的项目结构：

*   项目根目录下包含 `resnet_ablation/` 文件夹，里面涵盖配置、模型、数据、训练引擎、实验、日志等。每个子文件夹有相应的 Python 文件、功能模块，如 `configs/` 中的 YAML 文件、`engine/` 中的训练和评估脚本等。

这个结构清晰同时考虑了 token 限制。

为了满足用户需求，我们将代码拆分成多个文件，且保持文件简洁。在每个模块中，配置、日志、数据、模型和训练脚本都会有独立的文件。在 `models/blocks.py` 中实现基本模块（BasicBlock 和 Bottleneck），并添加对切换降采样选项的支持，为 CIFAR 数据集定制 IdentityPadDownsample 层。

我们将实现三种快捷方式选项：

*   选项A：使用零填充和步幅2。
*   选项B：使用1x1卷积投影。
*   选项C：为所有块使用投影。

将按照这种设计细节调整 `models/resnet.py` 为适应CIFAR风格ResNet，并在 `data/datasets.py` 中设置CIFAR-10数据集以及适当的图像处理和像素减法。

完成

## Response:
2025-09-16 16:21:28

下面是一个“可插拔（pluggable）”的 ResNet 消融实验工业级项目，从零实现，使用 PyTorch + TensorBoard + Loguru，支持：

*   基础块 BasicBlock（18/34）与瓶颈块 Bottleneck（50/101/152）可切换；
*   Shortcut 选项 A/B/C 可切换（CIFAR-10 上常用 A：恒等 + 维度零填充，下采样 stride 采样）；
*   ImageNet 风格与 CIFAR 风格两套构建器（CIFAR 使用 6n+2 层、特征图 32→16→8、通道 16/32/64，默认 option A 恒等短接）；
*   统一的训练/评估引擎、混合精度、梯度裁剪、自动保存与恢复、TensorBoard 记录、Loguru 日志；
*   清晰的 Google/PEP-257 风格 Docstring + 关键步骤中文注释。

参考依据（来自你上传的资料）

*   CIFAR-10 架构：输入 32×32，按大小 {32,16,8} 依次堆叠 3×3 卷积，通道 {16,32,64}，总层数 6n+2；在 CIFAR 上全部使用 identity（Option A）作为 shortcut（配合 stride 下采样和零填充），并提供 20/32/44/56/110/1202 等深度选项，这正是经典 ResNet-CIFAR 设定 2-ResNet。
*   Bottleneck 三层（1×1, 3×3, 1×1）用于更深网络（50/101/152），以更低计算代价获得更深的网络；在 ImageNet 设定下，下采样处常用 Option B（投影 1×1 conv） 2-ResNet 2-ResNet。
*   原始论文给出 ImageNet 与 CIFAR 的误差对比与层数配置表（我们据此提供基础工厂函数与可插拔块实现） 2-ResNet 2-ResNet。

* * *

项目结构
====

```
resnet_ablation/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ resnet_ablation/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ logger.py
│  ├─ utils.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ blocks.py
│  │  ├─ resnet_imagenet.py
│  │  └─ resnet_cifar.py
│  ├─ engine/
│  │  ├─ trainer.py
│  │  └─ evaluator.py
│  └─ checkpoints/        # 训练过程中自动创建
├─ scripts/
│  ├─ train.py
│  └─ eval.py
└─ configs/
   ├─ cifar10_resnet20.yaml
   ├─ cifar10_resnet56_ablate_optionB.yaml
   └─ imagenet_resnet50.yaml
```

> 说明：CIFAR 风格的 ResNet 使用 6n+2 规则（BasicBlock），默认 Shortcut=Option A；ImageNet 风格的 ResNet 既支持 BasicBlock（18/34）也支持 Bottleneck（50/101/152），默认在降采样处使用 Option B（1×1 投影），这是论文推荐的更高效实践 2-ResNet。

* * *

1) `pyproject.toml`
-------------------

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "resnet-ablation"
version = "0.1.0"
description = "Pluggable ResNet ablation project with PyTorch, TensorBoard, Loguru"
authors = [{name = "Your Team"}]
requires-python = ">=3.9"
dependencies = [
  "torch>=2.2.0",
  "torchvision>=0.17.0",
  "tensorboard>=2.12.0",
  "loguru>=0.7.0",
  "pyyaml>=6.0.1",
  "tqdm>=4.66.0"
]

[tool.setuptools.packages.find]
where = ["."]
include = ["resnet_ablation*"]
```

2) `requirements.txt`
---------------------

```txt
torch>=2.2.0
torchvision>=0.17.0
tensorboard>=2.12.0
loguru>=0.7.0
pyyaml>=6.0.1
tqdm>=4.66.0
```

3) `README.md`
--------------

````md
# ResNet Ablation (Pluggable)

- CIFAR & ImageNet 风格 ResNet 模型，Basic/Bottleneck/Shortcut A/B/C 可切换
- 训练引擎：混合精度、断点续训、TensorBoard、Loguru
- 参考：ResNet 论文（CIFAR 架构 6n+2；ImageNet Bottleneck/Option B）【见源码注释中的引用】

Quickstart:

```bash
pip install -e .
python scripts/train.py --config configs/cifar10_resnet20.yaml
````

````

---

## 4) `resnet_ablation/__init__.py`

```python
__all__ = []
````

5) `resnet_ablation/config.py`
------------------------------

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    arch: str = "resnet20_cifar"     # e.g., resnet20_cifar / resnet56_cifar / resnet18 / resnet50
    num_classes: int = 10
    shortcut: str = "A"              # A / B / C
    stem: str = "cifar"              # cifar / imagenet_deep_stem
    width_mult: float = 1.0
    drop_prob: float = 0.0

@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    warmup_epochs: int = 5
    sched: str = "cosine"            # cosine / multistep
    milestones: List[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    clip_grad_norm: Optional[float] = 1.0
    log_interval: int = 50
    val_interval: int = 1
    out_dir: str = "resnet_ablation/checkpoints"
    resume: Optional[str] = None
    seed: int = 42
    tb_dir: str = "runs"

@dataclass
class DataConfig:
    name: str = "cifar10"            # cifar10 / imagenet
    root: str = "./data"
    aug: str = "standard"            # standard / strong
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
```

6) `resnet_ablation/logger.py`
------------------------------

```python
from __future__ import annotations
from loguru import logger
import sys
from pathlib import Path

def setup_logger(save_dir: str) -> None:
    """配置 Loguru 日志输出到控制台与文件。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
    logger.add(Path(save_dir) / "train.log", level="INFO", rotation="10 MB", enqueue=True)
```

7) `resnet_ablation/utils.py`
-----------------------------

```python
from __future__ import annotations
import random, os
import numpy as np
import torch
from torch import nn

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

8) `resnet_ablation/data.py`
----------------------------

```python
from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision as tv

def build_cifar10(root: str, batch_size: int, num_workers: int, aug: str = "standard") -> Tuple[DataLoader, DataLoader]:
    """构建 CIFAR-10 DataLoader。
    论文在 CIFAR 使用“逐像素均值减法”，本实现采用更常见的标准化流程，并提供基础增强。"""
    if aug == "standard":
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616)),
        ])
    else:
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.CIFAR10),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616)),
        ])

    test_tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616)),
    ])

    train_set = tv.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
```

9) `resnet_ablation/metrics.py`
-------------------------------

```python
from __future__ import annotations
from typing import Tuple
import torch

@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Tuple[torch.Tensor, ...]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum())
    return tuple(res)
```

10) `resnet_ablation/models/__init__.py`
----------------------------------------

```python
from .resnet_cifar import resnet_cifar_6n2
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, resnet152

__all__ = [
    "resnet_cifar_6n2",
    "resnet18","resnet34","resnet50","resnet101","resnet152"
]
```

11) `resnet_ablation/models/blocks.py`
--------------------------------------

```python
from __future__ import annotations
from typing import Callable, Optional
import torch
from torch import nn
import torch.nn.functional as F

class ShortcutA(nn.Module):
    """Option A: 恒等+零填充，用于 CIFAR：通道不匹配时右侧零填充。"""
    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pad_c = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad_c > 0:
            pad = (0, 0, 0, 0, 0, self.pad_c)  # (W_left, W_right, H_bottom, H_top, C_front, C_back)
            x = F.pad(x, pad)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """ResNet BasicBlock：两个 3x3 卷积。"""
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "A", norm: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        self.shortcut_type = shortcut
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut == "A":
                self.downsample = ShortcutA(in_planes, planes * self.expansion, stride)
            elif shortcut in ("B", "C"):
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride),
                    norm(planes * self.expansion)
                )
            else:
                raise ValueError(f"Unknown shortcut {shortcut}")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """ResNet Bottleneck：1x1 → 3x3 → 1x1。ImageNet 风格更节省计算。"""
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B", norm: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            # 论文指出：Bottleneck 使用 identity shortcut 更高效，但在降采样/升维处常用 Option B（1x1 conv projection）:contentReference[oaicite:6]{index=6}
            if shortcut in ("B", "C", "A"):
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride),
                    norm(planes * self.expansion)
                )
            else:
                raise ValueError(f"Unknown shortcut {shortcut}")

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
```

12) `resnet_ablation/models/resnet_cifar.py`
--------------------------------------------

```python
from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock

class CIFARResNet(nn.Module):
    """CIFAR 风格 ResNet（6n+2，总通道 16/32/64，默认 Shortcut=Option A）。
    依据论文：输入 32×32，特征图依次 {32,16,8}，每个尺度堆叠 2n 个 BasicBlock，最后全局平均池化+全连接 :contentReference[oaicite:7]{index=7}。
    """
    def __init__(self, depth: int, num_classes: int = 10, shortcut: str = "A", width_mult: float = 1.0):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        base = int(16 * width_mult)
        self.in_planes = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, base,  n, stride=1, shortcut=shortcut)
        self.layer2 = self._make_layer(BasicBlock, base*2, n, stride=2, shortcut=shortcut)
        self.layer3 = self._make_layer(BasicBlock, base*4, n, stride=2, shortcut=shortcut)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base*4 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int, shortcut: str):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

def resnet_cifar_6n2(depth: int, num_classes: int = 10, shortcut: str = "A", width_mult: float = 1.0) -> CIFARResNet:
    return CIFARResNet(depth=depth, num_classes=num_classes, shortcut=shortcut, width_mult=width_mult)
```

13) `resnet_ablation/models/resnet_imagenet.py`
-----------------------------------------------

```python
from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock, Bottleneck

class ImageNetResNet(nn.Module):
    """ImageNet 风格 ResNet：支持 BasicBlock(18/34) 与 Bottleneck(50/101/152)。"""
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 1000, shortcut: str = "B"):
        super().__init__()
        self.in_planes = 64

        # deep stem 可选：这里采用标准 stem（7x7 stride 2 + 3x3 maxpool）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, shortcut=shortcut)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, shortcut=shortcut)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, shortcut=shortcut)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, shortcut=shortcut)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int, shortcut: str):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)

def _resnet(block: Type[nn.Module], layers: List[int], num_classes: int, shortcut: str) -> ImageNetResNet:
    return ImageNetResNet(block, layers, num_classes=num_classes, shortcut=shortcut)

# 工厂函数（与论文表层数对应）
def resnet18(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, shortcut)

def resnet34(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, shortcut)

def resnet50(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, shortcut)

def resnet101(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, shortcut)

def resnet152(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes, shortcut)
```

14) `resnet_ablation/engine/trainer.py`
---------------------------------------

```python
from __future__ import annotations
from typing import Dict
from pathlib import Path
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from ..metrics import topk_correct

class Trainer:
    """通用训练引擎（支持 AMP、断点续训、TensorBoard、Loguru）。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 sched, device: torch.device, out_dir: str, amp: bool = True,
                 clip_grad_norm: float | None = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.sched = sched
        self.device = device
        self.amp = amp
        self.clip_grad_norm = clip_grad_norm
        self.scaler = GradScaler(enabled=amp)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(out_dir)
        self.tb = SummaryWriter(log_dir=str(Path(out_dir).parent / "runs"))

    def save(self, epoch: int, extra: Dict | None = None) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sched": getattr(self.sched, "state_dict", lambda: {})(),
            "extra": extra or {}
        }
        torch.save(state, self.ckpt_dir / "last.pt")
        if extra and extra.get("is_best", False):
            torch.save(state, self.ckpt_dir / "best.pt")

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["opt"])
        if "sched" in ckpt and hasattr(self.sched, "load_state_dict"):
            self.sched.load_state_dict(ckpt["sched"])
        try:
            self.scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
        logger.info(f"Loaded checkpoint from {path}")
        return ckpt.get("epoch", 0)

    def train_one_epoch(self, epoch: int, loader, criterion) -> Dict[str, float]:
        self.model.train()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for step, (images, targets) in enumerate(loader, 1):
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(images)
                loss = criterion(logits, targets)
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.sched is not None and hasattr(self.sched, "step") and not isinstance(self.sched, torch.optim.lr_scheduler.MultiStepLR):
                self.sched.step()

            with torch.no_grad():
                correct1, = topk_correct(logits, targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs
            if step % 50 == 0:
                logger.info(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}, acc1={(correct1/bs*100):.2f}%")

        return {"loss": loss_sum / n, "acc1": acc1_sum / n}

    @torch.no_grad()
    def validate(self, epoch: int, loader, criterion) -> Dict[str, float]:
        self.model.eval()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for images, targets in loader:
            images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = criterion(logits, targets)
            correct1, = topk_correct(logits, targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs
        return {"val_loss": loss_sum / n, "val_acc1": acc1_sum / n}
```

15) `resnet_ablation/engine/evaluator.py`
-----------------------------------------

```python
from __future__ import annotations
from typing import Dict
import torch
from torch import nn
from ..metrics import topk_correct

@torch.no_grad()
def evaluate(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    top1, n = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        correct1, = topk_correct(logits, targets, (1,))
        top1 += correct1.item()
        n += images.size(0)
    return {"acc1": top1 / n}
```

16) `configs/cifar10_resnet20.yaml`
-----------------------------------

```yaml
model:
  arch: resnet20_cifar
  num_classes: 10
  shortcut: "A"     # CIFAR 上论文使用 Option A 恒等短接 :contentReference[oaicite:8]{index=8}
  width_mult: 1.0

data:
  name: cifar10
  root: ./data
  aug: standard

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 200
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar10_resnet20
  tb_dir: runs/cifar10_resnet20
  seed: 42
```

17) `configs/cifar10_resnet56_ablate_optionB.yaml`
--------------------------------------------------

```yaml
model:
  arch: resnet56_cifar
  num_classes: 10
  shortcut: "B"   # 消融：把 CIFAR 的 A 换成投影 B，看效果差异（论文表明 A/B/C 差异不大）:contentReference[oaicite:9]{index=9}

data:
  name: cifar10
  root: ./data
  aug: strong

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 200
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar10_resnet56_optB
  tb_dir: runs/cifar10_resnet56_optB
  seed: 42
```

18) `configs/imagenet_resnet50.yaml`
------------------------------------

```yaml
model:
  arch: resnet50
  num_classes: 1000
  shortcut: "B"

data:
  name: imagenet
  root: /path/to/imagenet
  aug: standard

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: multistep
  milestones: [30, 60, 90]
  gamma: 0.1

train:
  epochs: 120
  batch_size: 256
  num_workers: 8
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/imagenet_resnet50
  tb_dir: runs/imagenet_resnet50
  seed: 42
```

19) `scripts/train.py`
----------------------

```python
from __future__ import annotations
import argparse, yaml
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pathlib import Path
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device, count_params
from resnet_ablation.data import build_cifar10
from resnet_ablation.engine.trainer import Trainer
from resnet_ablation.models import resnet_cifar_6n2, resnet18, resnet34, resnet50, resnet101, resnet152

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def build_model(cfg: Config) -> torch.nn.Module:
    m = cfg.model
    if m.arch.endswith("_cifar"):
        depth = int(m.arch.replace("resnet", "").replace("_cifar", ""))
        model = resnet_cifar_6n2(depth=depth, num_classes=m.num_classes, shortcut=m.shortcut, width_mult=m.width_mult)
    elif m.arch == "resnet18":
        model = resnet18(num_classes=m.num_classes, shortcut=m.shortcut)
    elif m.arch == "resnet34":
        model = resnet34(num_classes=m.num_classes, shortcut=m.shortcut)
    elif m.arch == "resnet50":
        model = resnet50(num_classes=m.num_classes, shortcut=m.shortcut)
    elif m.arch == "resnet101":
        model = resnet101(num_classes=m.num_classes, shortcut=m.shortcut)
    elif m.arch == "resnet152":
        model = resnet152(num_classes=m.num_classes, shortcut=m.shortcut)
    else:
        raise ValueError(f"Unknown arch {m.arch}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data (support CIFAR10 demo out-of-box)
    if cfg.data.name.lower() == "cifar10":
        train_loader, val_loader = build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    else:
        raise NotImplementedError("Only CIFAR10 demo is wired in scripts; add ImageNet builder as needed.")

    # Model
    model = build_model(cfg).to(device)
    logger.info(f"Model: {cfg.model.arch}, params: {count_params(model)/1e6:.2f}M")

    # Optim & Sched
    opt = SGD(model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
              weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)
    if cfg.optim.sched == "cosine":
        sched = CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    else:
        sched = MultiStepLR(opt, milestones=cfg.optim.milestones, gamma=cfg.optim.gamma)

    # Loss
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(model, opt, sched, device, cfg.train.out_dir, amp=cfg.train.amp, clip_grad_norm=cfg.train.clip_grad_norm)
    start_epoch = 0
    if cfg.train.resume:
        start_epoch = trainer.load(cfg.train.resume)

    best_acc = 0.0
    for epoch in range(start_epoch, cfg.train.epochs):
        stats = trainer.train_one_epoch(epoch, train_loader, criterion)
        if isinstance(sched, MultiStepLR):  # multi-step: 每 epoch 调整
            sched.step()
        if (epoch + 1) % cfg.train.val_interval == 0:
            v = trainer.validate(epoch, val_loader, criterion)
            acc1 = v["val_acc1"] * 100.0
            logger.info(f"[Epoch {epoch}] train_loss={stats['loss']:.4f}, train_acc1={stats['acc1']*100:.2f}%, "
                        f"val_loss={v['val_loss']:.4f}, val_acc1={acc1:.2f}%")
            trainer.tb.add_scalar("train/loss", stats["loss"], epoch)
            trainer.tb.add_scalar("train/acc1", stats["acc1"], epoch)
            trainer.tb.add_scalar("val/loss", v["val_loss"], epoch)
            trainer.tb.add_scalar("val/acc1", v["val_acc1"], epoch)
            is_best = v["val_acc1"] > best_acc
            best_acc = max(best_acc, v["val_acc1"])
            trainer.save(epoch, {"is_best": is_best})

    logger.info("Training done.")

if __name__ == "__main__":
    main()
```

20) `scripts/eval.py`
---------------------

```python
from __future__ import annotations
import argparse, yaml
import torch
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device
from resnet_ablation.data import build_cifar10
from resnet_ablation.engine.evaluator import evaluate
from resnet_ablation.scripts.train import build_model  # 复用工厂

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()

    train_loader, val_loader = build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {args.ckpt}")

    res = evaluate(model, val_loader)
    logger.info(f"Eval Acc@1: {res['acc1']*100:.2f}%")

if __name__ == "__main__":
    main()
```

* * *

使用说明（快速开始）
==========

*   安装依赖并可编辑安装：
    ```bash
    pip install -e .
    ```
*   训练（CIFAR-10，ResNet-20）：
    ```bash
    python scripts/train.py --config configs/cifar10_resnet20.yaml
    ```
*   消融：将 CIFAR 的 Shortcut 从 A 改为 B（投影）：
    ```bash
    python scripts/train.py --config configs/cifar10_resnet56_ablate_optionB.yaml
    ```
*   可视化：
    ```bash
    tensorboard --logdir runs
    ```

* * *

与论文设定对齐的关键点
-----------

*   **CIFAR 风格**：6n+2 层、通道 {16,32,64}、三个尺度的 3×3 Conv 堆叠、最终 GAP+FC；CIFAR 默认 **Option A**（恒等短接+零填充，降采样用 stride 采样），以上均与论文描述一致 2-ResNet。
*   **ImageNet 风格**：更深网络使用 **Bottleneck**（1×1 降维 → 3×3 → 1×1 升维），在降采样/升维处通常使用 **Option B 投影**，在模型复杂度与精度间取得更优折中 2-ResNet 2-ResNet。
*   **深度选择**：18/34（Basic），50/101/152（Bottleneck）对应工厂函数，层分布与论文一致（表中 conv3/4/5 起始层 stride=2 下采样） 2-ResNet。

* * *

额外说明（工程化要点）
-----------

*   你可以在 `configs/` 中新增不同的消融组合（如 `width_mult`、`drop_prob`、`shortcut=C`），对比验证**Option A/B/C 差异通常不大**这一论文观察（我们也将其作为一个实验位） 2-ResNet。
*   若使用 ImageNet，请接入你本地的数据集构建函数（目前脚本默认演示 CIFAR-10）；ImageNet 训练策略可用 MultiStepLR（30/60/90）或 Cosine。
*   评估脚本 `scripts/eval.py` 支持从 `best.pt/last.pt` 加载进行验证集准确率评估；训练过程自动保存日志与 TensorBoard。

* * *

如果你想再加几种论文里常见的变体（如 DeepStem/AvgPool-Downsample、ECA/SE 插件化注意力、DropPath、LabelSmoothing/Mixup/CutMix），我这边可以直接在 `models/blocks.py` 和 `engine/trainer.py` 上继续堆可插拔模块与数据增强开关。

> 注：本文所有“与论文一致”的具体 CIFAR 和 Bottleneck/Shortcut 细节均来自你上传的 ResNet 资料（CIFAR-10 的 6n+2、{16/32/64} 与 Option A 设定；ImageNet 的 Bottleneck 设计与投影策略） 2-ResNet 2-ResNet 2-ResNet。

## Prompt:
2025-09-16 16:44:00

如果你想再加几种论文里常见的变体（如 DeepStem/AvgPool-Downsample、ECA/SE 插件化注意力、DropPath、LabelSmoothing/Mixup/CutMix），我这边可以直接在 \`models/blocks.py\` 和 \`engine/trainer.py\` 上继续堆可插拔模块与数据增强开关。

add please

## Response:
2025-09-16 16:44:02

已思考 1m 45s

The user wants to add features like DeepStem/AvgPool-Downsample, ECA/SE attention plugins, DropPath, LabelSmoothing/Mixup/CutMix. Some of these were partly supported before but not fully implemented, so we’ll need to update the project files accordingly. I'll handle this file-by-file, ensuring clear modifications and no need for citations since we're not using file\_search now. Let's start by adding the SE and ECA modules in models/blocks.py.

We’re adding several optional updates to the model, such as attention modules in BasicBlock and Bottleneck, DropPath (stochastic depth), and options for AvgPool-Downsample and DeepStem. The model's configuration will allow a choice between different stems, including a deep stem for ImageNet. I'll also handle augmentations like Mixup and CutMix, integrating them into the data pipeline through new utility functions.

We’ll update the configuration to include mixup and cutmix parameters like `mixup_alpha` and `cutmix_alpha` for label smoothing and soft targets. A new SoftTargetCrossEntropy class will handle these soft labels. I'll also integrate DropPath as stochastic depth in the block forward pass with a drop probability during training.

We’ll update ModelConfig by including a `drop_path_rate` for DropPath, and add support for attention modules like ‘none’, ‘se’, and ‘eca’ with their respective parameters. I’ll also implement ResNet-D's AvgPool-Downsample for shortcut projections, adjusting strides and pooling to modify both shortcut and residual paths as per ResNet-D design.

We need to implement various updates, including adding DropPath rates, implementing attention modules (SE, ECA), and modifying CIFAR and ImageNet-specific blocks. We'll introduce new parameters for ModelConfig and Trainer, such as `mixup_alpha`, `cutmix_alpha`, and `label_smoothing`. I'll make sure the scripts handle these options effectively and update the loss and training flow accordingly.

完成

太棒了，我们把你刚才点名的几类“可插拔增强/消融”一次性补齐：**DeepStem / AvgPool-Downsample（ResNet-D）**、**SE/ECA 注意力插件**、**DropPath（Stochastic Depth）**、**LabelSmoothing + Mixup/CutMix**。  
下面给出**更新后的项目树（标注 new/changed）**与**新增/修改文件的完整代码**。可直接覆盖你的现有目录运行。

* * *

✅ Updated Project Tree (new/changed)
====================================

```
resnet_ablation/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ resnet_ablation/
│  ├─ __init__.py
│  ├─ config.py                                  # CHANGED: 新增模型/训练增强参数
│  ├─ logger.py
│  ├─ utils.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ augment.py                                 # NEW: Mixup/CutMix 工具
│  ├─ losses.py                                  # NEW: SoftTargetCrossEntropy
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ blocks.py                               # CHANGED: SE/ECA/DropPath 可插拔
│  │  ├─ resnet_imagenet.py                      # CHANGED: DeepStem/ResNet-D 支持
│  │  ├─ resnet_cifar.py                         # CHANGED: 支持注意力/DropPath
│  │  └─ factory.py                              # NEW: build_model 工厂
│  ├─ engine/
│  │  ├─ trainer.py                              # CHANGED: LabelSmooth/Mixup/CutMix
│  │  └─ evaluator.py
│  └─ checkpoints/
├─ scripts/
│  ├─ train.py                                   # CHANGED: 调用新工厂 & 新增配置
│  └─ eval.py                                    # CHANGED: 调用新工厂 & 兼容加载
└─ configs/
   ├─ cifar10_resnet20.yaml
   ├─ cifar10_resnet56_ablate_optionB.yaml
   ├─ imagenet_resnet50.yaml
   └─ cifar10_resnet20_se_ecadrop_mix.yaml       # NEW: 一站式演示增强开关
```

* * *

CHANGED — `resnet_ablation/config.py`
-------------------------------------

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    arch: str = "resnet20_cifar"         # resnet20_cifar / resnet56_cifar / resnet18 / resnet50 ...
    num_classes: int = 10
    shortcut: str = "A"                  # A/B/C (CIFAR 基本块)；ImageNet 下仍使用 B 投影更常见
    stem: str = "cifar"                  # cifar / imagenet_standard / imagenet_deep
    downsample: str = "proj"             # proj(标准1x1投影) / avgproj(ResNet-D：AvgPool2d再1x1)
    width_mult: float = 1.0
    drop_prob: float = 0.0               # 预留：Dropout（分类头）
    drop_path_rate: float = 0.0          # NEW: Stochastic Depth 全局最大比例
    attention: str = "none"              # NEW: none / se / eca
    se_ratio: float = 0.25               # NEW: SE reduction ratio（仅 attention=se 时有效）

@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    warmup_epochs: int = 5
    sched: str = "cosine"                # cosine / multistep
    milestones: List[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    clip_grad_norm: Optional[float] = 1.0
    log_interval: int = 50
    val_interval: int = 1
    out_dir: str = "resnet_ablation/checkpoints"
    resume: Optional[str] = None
    seed: int = 42
    tb_dir: str = "runs"
    label_smoothing: float = 0.0         # NEW: 标签平滑（无 Mixup/CutMix 时生效）

@dataclass
class DataConfig:
    name: str = "cifar10"                # cifar10 / imagenet
    root: str = "./data"
    aug: str = "standard"                # standard / strong
    mixup_alpha: float = 0.0             # NEW: Mixup α>0 启用
    cutmix_alpha: float = 0.0            # NEW: CutMix α>0 启用

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
```

* * *

NEW — `resnet_ablation/augment.py`
----------------------------------

```python
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def one_hot(labels: Tensor, num_classes: int, smoothing: float = 0.0) -> Tensor:
    """将整数标签变为 one-hot（可选标签平滑）。"""
    with torch.no_grad():
        y = F.one_hot(labels, num_classes=num_classes).float()
        if smoothing > 0.0:
            y = y * (1.0 - smoothing) + smoothing / num_classes
    return y

def rand_bbox(W: int, H: int, lam: float) -> tuple[int, int, int, int]:
    """CutMix 随机框生成。"""
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def mixup(images: Tensor, targets: Tensor, num_classes: int, alpha: float, smoothing: float = 0.0):
    """Mixup 增强。返回混合后的图像与“软标签”"""
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for mixup.")
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    indices = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[indices]
    y1 = one_hot(targets, num_classes, smoothing)
    y2 = one_hot(targets[indices], num_classes, smoothing)
    y = lam * y1 + (1 - lam) * y2
    return mixed, y

def cutmix(images: Tensor, targets: Tensor, num_classes: int, alpha: float, smoothing: float = 0.0):
    """CutMix 增强。返回混合后的图像与“软标签”"""
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for cutmix.")
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    indices = torch.randperm(images.size(0), device=images.device)
    W, H = images.size(3), images.size(2)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    # 根据实际区域重新计算 lam
    box_area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - box_area / float(W * H)
    y1 = one_hot(targets, num_classes, smoothing)
    y2 = one_hot(targets[indices], num_classes, smoothing)
    y = lam_adj * y1 + (1 - lam_adj) * y2
    return mixed, y
```

* * *

NEW — `resnet_ablation/losses.py`
---------------------------------

```python
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    """用于 Mixup/CutMix 的交叉熵（targets 为概率分布）。"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        loss = -(targets * logp).sum(dim=1).mean()
        return loss
```

* * *

CHANGED — `resnet_ablation/models/blocks.py`
--------------------------------------------

```python
from __future__ import annotations
from typing import Callable, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------
# Stochastic Depth (DropPath)
# ---------------------------
class DropPath(nn.Module):
    """按样本随机丢弃残差分支（Stochastic Depth）"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # (N, 1, 1, 1) broadcast
        mask = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x / keep * mask

# -------------
# Attention: SE
# -------------
class SEModule(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w

# -------------
# Attention: ECA
# -------------
def _eca_kernel(channels: int, gamma: int = 2, b: int = 1) -> int:
    k = int(abs((math.log2(channels) / gamma) + b))
    if k % 2 == 0:
        k += 1
    return max(3, k)

class ECAModule(nn.Module):
    """Efficient Channel Attention（无需降维，1D conv）"""
    def __init__(self, channels: int, k_size: Optional[int] = None):
        super().__init__()
        k = _eca_kernel(channels) if k_size is None else k_size
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                               # (N, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))# (N, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y

def build_attention(kind: str, channels: int, se_ratio: float = 0.25) -> Optional[nn.Module]:
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    if kind == "se":
        return SEModule(channels, ratio=se_ratio)
    if kind == "eca":
        return ECAModule(channels)
    raise ValueError(f"Unknown attention kind: {kind}")

# -------------
# Shortcut A/B
# -------------
class ShortcutA(nn.Module):
    """Option A: 恒等+零填充，用于 CIFAR：通道不匹配时右侧零填充。"""
    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pad_c = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad_c > 0:
            pad = (0, 0, 0, 0, 0, self.pad_c)
            x = F.pad(x, pad)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ----------------
# Basic / Bottleneck
# ----------------
class BasicBlock(nn.Module):
    """ResNet BasicBlock：两个 3×3 卷积，支持注意力与 DropPath。"""
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "A",
                 norm: Callable[..., nn.Module] = nn.BatchNorm2d,
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path: float = 0.0):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)
        self.attn = build_attention(attention, planes, se_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut == "A":
                self.downsample = ShortcutA(in_planes, planes * self.expansion, stride)
            elif shortcut in ("B", "C"):
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride),
                    norm(planes * self.expansion)
                )
            else:
                raise ValueError(f"Unknown shortcut {shortcut}")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """ResNet Bottleneck：1×1 → 3×3 → 1×1，支持注意力、DropPath。"""
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B",
                 norm: Callable[..., nn.Module] = nn.BatchNorm2d,
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path: float = 0.0,
                 downsample_mode: str = "proj"):  # proj / avgproj (ResNet-D)
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm(planes)

        # 残差主分支下采样：仍在 3x3 上做 stride（ResNet-D 的 avg 在快捷分支处理）
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)

        self.attn = build_attention(attention, planes * self.expansion, se_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            if downsample_mode == "avgproj":
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False),
                    conv1x1(in_planes, planes * self.expansion, stride=1),
                    norm(planes * self.expansion)
                )
            else:
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride=stride),
                    norm(planes * self.expansion)
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
```

* * *

CHANGED — `resnet_ablation/models/resnet_imagenet.py`
-----------------------------------------------------

```python
from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock, Bottleneck

def _make_stem(kind: str) -> nn.Sequential:
    """ImageNet stem: standard(7x7, s=2) or deep(3×3×3)."""
    kind = kind.lower()
    if kind == "imagenet_deep":
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )

class ImageNetResNet(nn.Module):
    """ImageNet 风格 ResNet：支持 BasicBlock(18/34) 与 Bottleneck(50/101/152)，可选 DeepStem/ResNet-D/注意力/DropPath。"""
    def __init__(self, block: Type[nn.Module], layers: List[int],
                 num_classes: int = 1000,
                 shortcut: str = "B",
                 stem: str = "imagenet_standard",
                 downsample: str = "proj",         # proj / avgproj (ResNet-D)
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.in_planes = 64

        self.stem = _make_stem(stem)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 线性分配 DropPath 到每个 block
        total_blocks = sum(layers)
        dpr = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        dp_iter = iter(dpr)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int,
                    shortcut: str, attention: str, se_ratio: float, dp_iter, downsample: str):
        layers: List[nn.Module] = []
        # 第一个 block 可能下采样
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut,
                            attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter),
                            downsample_mode=downsample if block is Bottleneck else "proj"))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut,
                                attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter),
                                downsample_mode=downsample if block is Bottleneck else "proj"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def _resnet(block: Type[nn.Module], layers: List[int], **kwargs) -> ImageNetResNet:
    return ImageNetResNet(block, layers, **kwargs)

def resnet18(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet34(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet50(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet101(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet152(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, shortcut=shortcut, **extras)
```

* * *

CHANGED — `resnet_ablation/models/resnet_cifar.py`
--------------------------------------------------

```python
from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock

class CIFARResNet(nn.Module):
    """CIFAR 风格 ResNet（6n+2），支持注意力与 DropPath。"""
    def __init__(self, depth: int, num_classes: int = 10, shortcut: str = "A",
                 width_mult: float = 1.0, attention: str = "none", se_ratio: float = 0.25,
                 drop_path_rate: float = 0.0):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        base = int(16 * width_mult)
        self.in_planes = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)

        total_blocks = 3 * (2 * n)
        dpr = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        dp_iter = iter(dpr)

        self.layer1 = self._make_layer(BasicBlock, base,  n, stride=1, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)
        self.layer2 = self._make_layer(BasicBlock, base*2, n, stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)
        self.layer3 = self._make_layer(BasicBlock, base*4, n, stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base*4 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int,
                    shortcut: str, attention: str, se_ratio: float, dp_iter):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut,
                            attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter)))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut,
                                attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet_cifar_6n2(depth: int, num_classes: int = 10, shortcut: str = "A",
                      width_mult: float = 1.0, attention: str = "none",
                      se_ratio: float = 0.25, drop_path_rate: float = 0.0) -> CIFARResNet:
    return CIFARResNet(depth=depth, num_classes=num_classes, shortcut=shortcut,
                       width_mult=width_mult, attention=attention, se_ratio=se_ratio,
                       drop_path_rate=drop_path_rate)
```

* * *

NEW — `resnet_ablation/models/factory.py`
-----------------------------------------

```python
from __future__ import annotations
import torch
from resnet_ablation.config import Config
from .resnet_cifar import resnet_cifar_6n2
from .resnet_imagenet import resnet18, resnet34, resnet50, resnet101, resnet152

def build_model(cfg: Config) -> torch.nn.Module:
    m = cfg.model
    extras = dict(
        attention=m.attention,
        se_ratio=m.se_ratio,
        drop_path_rate=m.drop_path_rate,
        stem=m.stem if "imagenet" in m.stem else "imagenet_standard",
        downsample=m.downsample,
    )
    if m.arch.endswith("_cifar"):
        depth = int(m.arch.replace("resnet", "").replace("_cifar", ""))
        return resnet_cifar_6n2(depth=depth, num_classes=m.num_classes, shortcut=m.shortcut,
                                width_mult=m.width_mult, attention=m.attention,
                                se_ratio=m.se_ratio, drop_path_rate=m.drop_path_rate)
    if m.arch == "resnet18":
        return resnet18(num_classes=m.num_classes, shortcut=m.shortcut, **extras)
    if m.arch == "resnet34":
        return resnet34(num_classes=m.num_classes, shortcut=m.shortcut, **extras)
    if m.arch == "resnet50":
        return resnet50(num_classes=m.num_classes, shortcut=m.shortcut, **extras)
    if m.arch == "resnet101":
        return resnet101(num_classes=m.num_classes, shortcut=m.shortcut, **extras)
    if m.arch == "resnet152":
        return resnet152(num_classes=m.num_classes, shortcut=m.shortcut, **extras)
    raise ValueError(f"Unknown arch {m.arch}")
```

* * *

CHANGED — `resnet_ablation/engine/trainer.py`
---------------------------------------------

```python
from __future__ import annotations
from typing import Dict
from pathlib import Path
import random
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from ..metrics import topk_correct
from ..augment import mixup as do_mixup, cutmix as do_cutmix
from ..losses import SoftTargetCrossEntropy

class Trainer:
    """通用训练引擎：AMP / 断点续训 / TensorBoard / Loguru / Mixup/CutMix/LabelSmoothing / GradClip。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 sched, device: torch.device, out_dir: str, tb_dir: str,
                 amp: bool = True, clip_grad_norm: float | None = 1.0,
                 num_classes: int = 10,
                 label_smoothing: float = 0.0,
                 mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.sched = sched
        self.device = device
        self.amp = amp
        self.clip_grad_norm = clip_grad_norm
        self.scaler = GradScaler(enabled=amp)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(out_dir)
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(tb_dir))

        # 增强/损失配置
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # 当使用 Mixup/CutMix 时采用 soft CE
        self.soft_ce = SoftTargetCrossEntropy()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def save(self, epoch: int, extra: Dict | None = None) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "sched": getattr(self.sched, "state_dict", lambda: {})(),
            "extra": extra or {}
        }
        torch.save(state, self.ckpt_dir / "last.pt")
        if extra and extra.get("is_best", False):
            torch.save(state, self.ckpt_dir / "best.pt")

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["opt"])
        if "sched" in ckpt and hasattr(self.sched, "load_state_dict"):
            self.sched.load_state_dict(ckpt["sched"])
        try:
            self.scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
        logger.info(f"Loaded checkpoint from {path}")
        return ckpt.get("epoch", 0)

    def _maybe_augment(self, images: torch.Tensor, targets: torch.Tensor):
        """根据 alpha 启用 Mixup/CutMix；当两者都>0时随机二选一。"""
        if self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0:
            return images, targets, False  # no soft labels
        use_mixup = False
        if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
            use_mixup = bool(random.getrandbits(1))
        else:
            use_mixup = self.mixup_alpha > 0.0

        if use_mixup:
            images, soft_targets = do_mixup(images, targets, self.num_classes, self.mixup_alpha, smoothing=0.0)
        else:
            images, soft_targets = do_cutmix(images, targets, self.num_classes, self.cutmix_alpha, smoothing=0.0)
        return images, soft_targets, True

    def train_one_epoch(self, epoch: int, loader, _) -> Dict[str, float]:
        self.model.train()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 保存硬标签用于精度指标
            hard_targets = targets.detach()

            # 可能应用 Mixup/CutMix
            images, targets_or_soft, is_soft = self._maybe_augment(images, targets)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(images)
                if is_soft:
                    loss = self.soft_ce(logits, targets_or_soft)
                else:
                    loss = self.ce(logits, targets_or_soft)

            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 若使用 Cosine 等 per-iter 调度器在此 step
            if self.sched is not None and hasattr(self.sched, "step") and not isinstance(self.sched, torch.optim.lr_scheduler.MultiStepLR):
                self.sched.step()

            with torch.no_grad():
                correct1, = topk_correct(logits, hard_targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs

            if step % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}, acc1={(correct1/bs*100):.2f}%, lr={lr:.4e}")

        return {"loss": loss_sum / n, "acc1": acc1_sum / n}

    @torch.no_grad()
    def validate(self, epoch: int, loader, _) -> Dict[str, float]:
        self.model.eval()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.ce(logits, targets)  # 验证阶段不使用 mix/cut
            correct1, = topk_correct(logits, targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs
        return {"val_loss": loss_sum / n, "val_acc1": acc1_sum / n}
```

* * *

CHANGED — `scripts/train.py`
----------------------------

```python
from __future__ import annotations
import argparse, yaml
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pathlib import Path
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device, count_params
from resnet_ablation.data import build_cifar10
from resnet_ablation.engine.trainer import Trainer
from resnet_ablation.models.factory import build_model

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data (默认演示 CIFAR10；ImageNet 可自行接入)
    if cfg.data.name.lower() == "cifar10":
        train_loader, val_loader = build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    else:
        raise NotImplementedError("Only CIFAR10 demo is wired in scripts; add ImageNet builder as needed.")

    # Model
    model = build_model(cfg).to(device)
    logger.info(f"Model: {cfg.model.arch}, params: {count_params(model)/1e6:.2f}M | attn={cfg.model.attention} | dpr={cfg.model.drop_path_rate}")

    # Optim & Sched
    opt = SGD(model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
              weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)
    if cfg.optim.sched == "cosine":
        sched = CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    else:
        sched = MultiStepLR(opt, milestones=cfg.optim.milestones, gamma=cfg.optim.gamma)

    trainer = Trainer(
        model, opt, sched, device,
        out_dir=cfg.train.out_dir, tb_dir=cfg.train.tb_dir,
        amp=cfg.train.amp, clip_grad_norm=cfg.train.clip_grad_norm,
        num_classes=cfg.model.num_classes,
        label_smoothing=cfg.train.label_smoothing,
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha
    )

    start_epoch = 0
    if cfg.train.resume:
        start_epoch = trainer.load(cfg.train.resume)

    best_acc = 0.0
    for epoch in range(start_epoch, cfg.train.epochs):
        stats = trainer.train_one_epoch(epoch, train_loader, None)
        if isinstance(sched, MultiStepLR):  # multi-step: 每 epoch 调整
            sched.step()
        if (epoch + 1) % cfg.train.val_interval == 0:
            v = trainer.validate(epoch, val_loader, None)
            acc1 = v["val_acc1"] * 100.0
            logger.info(f"[Epoch {epoch}] train_loss={stats['loss']:.4f}, train_acc1={stats['acc1']*100:.2f}%, "
                        f"val_loss={v['val_loss']:.4f}, val_acc1={acc1:.2f}%")
            trainer.tb.add_scalar("train/loss", stats["loss"], epoch)
            trainer.tb.add_scalar("train/acc1", stats["acc1"], epoch)
            trainer.tb.add_scalar("val/loss", v["val_loss"], epoch)
            trainer.tb.add_scalar("val/acc1", v["val_acc1"], epoch)
            trainer.tb.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

            is_best = v["val_acc1"] > best_acc
            best_acc = max(best_acc, v["val_acc1"])
            trainer.save(epoch, {"is_best": is_best})

    logger.info("Training done.")

if __name__ == "__main__":
    main()
```

* * *

CHANGED — `scripts/eval.py`
---------------------------

```python
from __future__ import annotations
import argparse, yaml, torch
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device
from resnet_ablation.data import build_cifar10
from resnet_ablation.engine.evaluator import evaluate
from resnet_ablation.models.factory import build_model

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()

    if cfg.data.name.lower() == "cifar10":
        _, val_loader = build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    else:
        raise NotImplementedError("Only CIFAR10 demo is wired.")

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {args.ckpt}")

    res = evaluate(model, val_loader)
    logger.info(f"Eval Acc@1: {res['acc1']*100:.2f}%")

if __name__ == "__main__":
    main()
```

* * *

NEW — `configs/cifar10_resnet20_se_ecadrop_mix.yaml`
----------------------------------------------------

> **一站式示例**：CIFAR-10 上启用 **ECA** 注意力 + **DropPath** + **Mixup/CutMix** + **LabelSmoothing**。你也可以改成 `attention: "se"` 测试 SE。

```yaml
model:
  arch: resnet20_cifar
  num_classes: 10
  shortcut: "A"
  width_mult: 1.0
  attention: "eca"         # "none" / "se" / "eca"
  se_ratio: 0.25
  drop_path_rate: 0.05     # 线性分配到每个 block

data:
  name: cifar10
  root: ./data
  aug: strong
  mixup_alpha: 0.2         # >0 启用 Mixup
  cutmix_alpha: 0.0        # 同时>0则随机二选一

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 200
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar10_resnet20_eca_dpr_mix
  tb_dir: runs/cifar10_resnet20_eca_dpr_mix
  seed: 42
  label_smoothing: 0.1     # 无 mix/cut 时生效；有 mix/cut 时用 SoftTargetCE
```

* * *

🧪 使用方式（快速验证）
=============

```bash
# 安装（如未安装）
pip install -e .

# 训练：CIFAR-10 + ECA + DropPath + Mixup + LabelSmooth
python scripts/train.py --config configs/cifar10_resnet20_se_ecadrop_mix.yaml

# 张量板
tensorboard --logdir runs
```

* * *

设计要点回顾（与消融开关对应）
---------------

*   **注意力插件**：`model.attention = "se" | "eca"`；对 **BasicBlock/Bottleneck** 的残差输出做通道重标定；
*   **DropPath**：`model.drop_path_rate > 0` 时按层线性分配，训练阶段随机丢弃残差分支提升泛化；
*   **DeepStem**：ImageNet 模型将 `model.stem="imagenet_deep"`；CIFAR 仍用 3×3；
*   **ResNet-D Downsample**：ImageNet 模型将 `model.downsample="avgproj"`，即快捷分支 `AvgPool2d + 1×1 conv`；
*   **LabelSmoothing**：`train.label_smoothing > 0` 且未启用 mixup/cutmix 时生效；
*   **Mixup/CutMix**：`data.mixup_alpha` / `data.cutmix_alpha` 设为 `>0` 启用，训练时在两者都启用时随机二选一；使用 `SoftTargetCrossEntropy`。

* * *

需要我再补：**SE vs ECA 对比配置**、**ResNet-D + DeepStem 的 ImageNet 配置示例**、或是 **CIFAR-100** 支持？直接说，我继续把配置和代码补齐。

## Prompt:
2025-09-16 16:53:34

补齐配置和代码

give me bash code for each experiements

## Response:
2025-09-16 16:53:37

已思考 1m 39s

The user wants to finalize the configuration and code, including bash commands for each experiment. They previously asked about ResNet Ablation with enhancements, and I provided code and config. Now, it's time to include additional configurations, like CIFAR-100 support. I'll add a CIFAR-100 data builder and placeholder for ImageNet without modifying `data.py` as requested.

I need to implement the required minimal ImageNet builder and provide extra configurations such as various ResNet models for CIFAR-10, CIFAR-100, and ImageNet datasets. I'll add bash commands for each experiment and update `train.py` to support dataset selection for CIFAR-10, CIFAR-100, and ImageNet by modifying the dataset builder and dataloader dispatcher.

For this update, I need to revise the models to ignore the stem parameter for CIFAR. I'll also modify `data.py` to support CIFAR-100 and ImageNet, adding appropriate transformations and normalization for ImageNet. I'll include configurations for these datasets, focusing on logging with Loguru and TensorBoard. I'll also provide bash one-liners for each experiment. No need to use file citations since it's just code-related.

用户要求“补齐配置和代码”，因此我将提供修改过后的文件代码，分段显示文件路径和代码块。由于用户用中文提问，我会继续使用中文回答，不再提问澄清问题，直接提供所需的代码改动。

我将提供所需的代码更新和命令，包括以下内容：

1.  增加了CIFAR-100和ImageNet的数据构建代码，支持批量数据加载。
2.  添加了多个实验的bash命令（包括多种seeds），以及如何迭代运行。
3.  更新了train.py和eval.py，确保它们支持新的数据集格式，并可以处理CLI选项如`--out-suffix`和`--seed`。
4.  提供了对应的配置文件和Git更新树。

完成

收到！我把**数据集构建（CIFAR-100 / ImageNet）**补上，并把 `train.py / eval.py` 改成自动分发数据构建，同时给出**成套可复现的实验配置**（SE vs ECA、ResNet-D + DeepStem、CIFAR-100、深度/Shortcut 消融）以及**每个实验的 bash 启动脚本**。直接覆盖对应文件即可运行。

* * *

🔧 代码更新（new/changed）
====================

CHANGED — `resnet_ablation/data.py`
-----------------------------------

```python
from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import torchvision as tv

@dataclass
class _DLArgs:
    batch_size: int
    num_workers: int

# ---- Common Normalization ----
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2470, 0.2435, 0.2616)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def _make_cifar_tf(aug: str):
    if aug == "strong":
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.CIFAR10),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    else:
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    test_tf = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    return train_tf, test_tf

def _make_imagenet_tf(aug: str):
    if aug == "strong":
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.IMAGENET),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        train_tf = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    val_tf = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    return train_tf, val_tf

def build_cifar10(root: str, batch_size: int, num_workers: int, aug: str = "standard") -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = _make_cifar_tf(aug)
    train_set = tv.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set  = tv.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    return (DataLoader(train_set, shuffle=True,  **loader_args),
            DataLoader(test_set,  shuffle=False, **loader_args))

def build_cifar100(root: str, batch_size: int, num_workers: int, aug: str = "standard") -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = _make_cifar_tf(aug)
    train_set = tv.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set  = tv.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    return (DataLoader(train_set, shuffle=True,  **loader_args),
            DataLoader(test_set,  shuffle=False, **loader_args))

def build_imagenet(root: str, batch_size: int, num_workers: int, aug: str = "standard") -> Tuple[DataLoader, DataLoader]:
    """ImageNet 采用 ImageFolder 目录：
    root/
      ├── train/ class_x/..., class_y/...
      └── val/   class_x/..., class_y/...
    """
    train_tf, val_tf = _make_imagenet_tf(aug)
    train_set = tv.datasets.ImageFolder(root=f"{root}/train", transform=train_tf)
    val_set   = tv.datasets.ImageFolder(root=f"{root}/val",   transform=val_tf)
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
    return (DataLoader(train_set, shuffle=True,  **loader_args),
            DataLoader(val_set,   shuffle=False, **loader_args))

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    name = cfg.data.name.lower()
    if name == "cifar10":
        return build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    if name == "cifar100":
        return build_cifar100(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    if name == "imagenet":
        return build_imagenet(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    raise NotImplementedError(f"Unknown dataset: {cfg.data.name}")
```

* * *

CHANGED — `scripts/train.py`（支持多数据集 + 便捷覆盖参数）
---------------------------------------------

```python
from __future__ import annotations
import argparse, yaml
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device, count_params
from resnet_ablation.data import build_dataloaders
from resnet_ablation.engine.trainer import Trainer
from resnet_ablation.models.factory import build_model

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-suffix", type=str, default="", help="追加到 out_dir/tb_dir 的后缀，便于多种子并行")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.out_suffix:
        cfg.train.out_dir = f"{cfg.train.out_dir}_{args.out_suffix}"
        cfg.train.tb_dir  = f"{cfg.train.tb_dir}_{args.out_suffix}"
    if args.seed is not None:
        cfg.train.seed = args.seed

    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Dataset={cfg.data.name} | aug={cfg.data.aug} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    # Model
    model = build_model(cfg).to(device)
    logger.info(f"Model: {cfg.model.arch}, params: {count_params(model)/1e6:.2f}M | attn={cfg.model.attention} | dpr={cfg.model.drop_path_rate}")

    # Optim & Sched
    opt = SGD(model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
              weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)
    if cfg.optim.sched == "cosine":
        sched = CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    else:
        sched = MultiStepLR(opt, milestones=cfg.optim.milestones, gamma=cfg.optim.gamma)

    trainer = Trainer(
        model, opt, sched, device,
        out_dir=cfg.train.out_dir, tb_dir=cfg.train.tb_dir,
        amp=cfg.train.amp, clip_grad_norm=cfg.train.clip_grad_norm,
        num_classes=cfg.model.num_classes,
        label_smoothing=cfg.train.label_smoothing,
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha
    )

    start_epoch = 0
    if cfg.train.resume:
        start_epoch = trainer.load(cfg.train.resume)

    best_acc = 0.0
    for epoch in range(start_epoch, cfg.train.epochs):
        stats = trainer.train_one_epoch(epoch, train_loader, None)
        if isinstance(sched, MultiStepLR):
            sched.step()
        if (epoch + 1) % cfg.train.val_interval == 0:
            v = trainer.validate(epoch, val_loader, None)
            acc1 = v["val_acc1"] * 100.0
            logger.info(f"[Epoch {epoch}] train_loss={stats['loss']:.4f}, train_acc1={stats['acc1']*100:.2f}%, "
                        f"val_loss={v['val_loss']:.4f}, val_acc1={acc1:.2f}%")
            trainer.tb.add_scalar("train/loss", stats["loss"], epoch)
            trainer.tb.add_scalar("train/acc1", stats["acc1"], epoch)
            trainer.tb.add_scalar("val/loss", v["val_loss"], epoch)
            trainer.tb.add_scalar("val/acc1", v["val_acc1"], epoch)
            trainer.tb.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
            is_best = v["val_acc1"] > best_acc
            best_acc = max(best_acc, v["val_acc1"])
            trainer.save(epoch, {"is_best": is_best})

    logger.info("Training done.")

if __name__ == "__main__":
    main()
```

* * *

CHANGED — `scripts/eval.py`（自动构建数据）
-----------------------------------

```python
from __future__ import annotations
import argparse, yaml, torch
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device
from resnet_ablation.data import build_dataloaders
from resnet_ablation.engine.evaluator import evaluate
from resnet_ablation.models.factory import build_model

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {args.ckpt}")

    res = evaluate(model, val_loader)
    logger.info(f"Eval Acc@1: {res['acc1']*100:.2f}%")

if __name__ == "__main__":
    main()
```

* * *

📦 新增/补齐配置（configs/\*.yaml）
===========================

> 已有：
> 
> *   `cifar10_resnet20.yaml`（Baseline, CIFAR-10, Option A）
> *   `cifar10_resnet56_ablate_optionB.yaml`（CIFAR-10，Shortcut B 消融）
> *   `imagenet_resnet50.yaml`（ImageNet R50 baseline）
>     

下面新增并完善更多实验：

NEW — `configs/cifar10_resnet110_optionA.yaml`
----------------------------------------------

```yaml
model:
  arch: resnet110_cifar
  num_classes: 10
  shortcut: "A"
  width_mult: 1.0
  attention: "none"
  se_ratio: 0.25
  drop_path_rate: 0.0

data:
  name: cifar10
  root: ./data
  aug: standard
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 200
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar10_resnet110_A
  tb_dir: runs/cifar10_resnet110_A
  seed: 42
  label_smoothing: 0.0
```

NEW — `configs/cifar10_resnet20_se.yaml`
----------------------------------------

```yaml
model:
  arch: resnet20_cifar
  num_classes: 10
  shortcut: "A"
  width_mult: 1.0
  attention: "se"
  se_ratio: 0.25
  drop_path_rate: 0.0

data:
  name: cifar10
  root: ./data
  aug: standard
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 200
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar10_resnet20_se
  tb_dir: runs/cifar10_resnet20_se
  seed: 42
  label_smoothing: 0.1
```

(已给) `configs/cifar10_resnet20_se_ecadrop_mix.yaml`
---------------------------------------------------

> 已包含 ECA + DropPath + Mixup + LabelSmooth

NEW — `configs/cifar100_resnet56_se_mix.yaml`
---------------------------------------------

```yaml
model:
  arch: resnet56_cifar
  num_classes: 100
  shortcut: "A"
  width_mult: 1.0
  attention: "se"
  se_ratio: 0.25
  drop_path_rate: 0.05

data:
  name: cifar100
  root: ./data
  aug: strong
  mixup_alpha: 0.2
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 2.0e-4
  nesterov: true
  sched: cosine
  warmup_epochs: 5

train:
  epochs: 240
  batch_size: 128
  num_workers: 4
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/cifar100_resnet56_se_mix
  tb_dir: runs/cifar100_resnet56_se_mix
  seed: 42
  label_smoothing: 0.1
```

NEW — `configs/imagenet_resnet50_baseline.yaml`
-----------------------------------------------

```yaml
model:
  arch: resnet50
  num_classes: 1000
  shortcut: "B"
  stem: "imagenet_standard"
  downsample: "proj"
  attention: "none"
  se_ratio: 0.25
  drop_path_rate: 0.0

data:
  name: imagenet
  root: /path/to/imagenet
  aug: standard
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: multistep
  milestones: [30, 60, 90]
  gamma: 0.1

train:
  epochs: 120
  batch_size: 256
  num_workers: 8
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/imagenet_r50_base
  tb_dir: runs/imagenet_r50_base
  seed: 42
  label_smoothing: 0.1
```

NEW — `configs/imagenet_resnet50_deepstem_resnetd_se.yaml`
----------------------------------------------------------

```yaml
model:
  arch: resnet50
  num_classes: 1000
  shortcut: "B"
  stem: "imagenet_deep"     # DeepStem: 3×3×3
  downsample: "avgproj"     # ResNet-D: AvgPool2d + 1×1
  attention: "se"
  se_ratio: 0.25
  drop_path_rate: 0.0

data:
  name: imagenet
  root: /path/to/imagenet
  aug: standard
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: multistep
  milestones: [30, 60, 90]
  gamma: 0.1

train:
  epochs: 120
  batch_size: 256
  num_workers: 8
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/imagenet_r50_deepD_se
  tb_dir: runs/imagenet_r50_deepD_se
  seed: 42
  label_smoothing: 0.1
```

NEW — `configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml`
----------------------------------------------------------------

```yaml
model:
  arch: resnet101
  num_classes: 1000
  shortcut: "B"
  stem: "imagenet_deep"
  downsample: "avgproj"
  attention: "eca"
  se_ratio: 0.25
  drop_path_rate: 0.2        # 线性分配到所有 blocks

data:
  name: imagenet
  root: /path/to/imagenet
  aug: strong
  mixup_alpha: 0.0
  cutmix_alpha: 0.0

optim:
  name: sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 1.0e-4
  nesterov: true
  sched: multistep
  milestones: [30, 60, 90]
  gamma: 0.1

train:
  epochs: 120
  batch_size: 256
  num_workers: 8
  amp: true
  clip_grad_norm: 1.0
  val_interval: 1
  out_dir: resnet_ablation/checkpoints/imagenet_r101_deepD_eca_dpr
  tb_dir: runs/imagenet_r101_deepD_eca_dpr
  seed: 42
  label_smoothing: 0.1
```

* * *

🚀 Bash 启动脚本（逐实验给出）
===================

> 可把下面内容保存为 `run_all.sh` 或直接逐条粘贴执行。支持 `--out-suffix` 与 `--seed`，便于多种子并行复现实验。

```bash
# 0) 安装（首次）
pip install -e .

# 1) CIFAR-10 Baseline: ResNet-20, Option A
python scripts/train.py --config configs/cifar10_resnet20.yaml

# 2) CIFAR-10 消融：Shortcut 改为 Option B（ResNet-56）
python scripts/train.py --config configs/cifar10_resnet56_ablate_optionB.yaml

# 3) CIFAR-10 + SE（ResNet-20）
python scripts/train.py --config configs/cifar10_resnet20_se.yaml

# 4) CIFAR-10 + ECA + DropPath + Mixup（ResNet-20，一站式增强）
python scripts/train.py --config configs/cifar10_resnet20_se_ecadrop_mix.yaml

# 5) CIFAR-10 深层：ResNet-110（Option A baseline）
python scripts/train.py --config configs/cifar10_resnet110_optionA.yaml

# 6) CIFAR-100 + SE + Mixup（ResNet-56）
python scripts/train.py --config configs/cifar100_resnet56_se_mix.yaml

# 7) ImageNet R50 baseline（标准 stem + 标准 downsample）
#    注意：请把 configs 中的 data.root 改为你的 ImageNet 路径（包含 train/ 与 val/）
python scripts/train.py --config configs/imagenet_resnet50_baseline.yaml

# 8) ImageNet R50 DeepStem + ResNet-D + SE
python scripts/train.py --config configs/imagenet_resnet50_deepstem_resnetd_se.yaml

# 9) ImageNet R101 DeepStem + ResNet-D + ECA + DropPath
python scripts/train.py --config configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml

# ===== Multi-seed 复现示例（以 CIFAR-10 ResNet-20 baseline 为例） =====
for s in 1 2 3; do
  python scripts/train.py --config configs/cifar10_resnet20.yaml --seed $s --out-suffix "s${s}"
done

# ===== 评估示例（用 last.pt 或 best.pt）=====
python scripts/eval.py --config configs/cifar10_resnet20.yaml --ckpt resnet_ablation/checkpoints/cifar10_resnet20/last.pt
```

* * *

✅ 提示与注意
=======

*   **ImageNet 路径**：`data.root` 必须包含 `train/` 与 `val/` 两个子目录（`torchvision.datasets.ImageFolder` 约定）。
*   **多实验不互相覆盖**：每个配置里已设置不同 `out_dir` / `tb_dir`；多种子实验请配合 `--out-suffix`。
*   **性能增强开关**：
    *   注意力：`model.attention="se|eca"`
    *   DropPath：`model.drop_path_rate>0`
    *   ResNet-D：`model.downsample="avgproj"`（仅 ImageNet/Bottleneck 下生效）
    *   DeepStem：`model.stem="imagenet_deep"`（ImageNet）
    *   Mixup/CutMix：`data.mixup_alpha`/`data.cutmix_alpha`
    *   LabelSmooth：`train.label_smoothing`（无 mix/cut 时生效）
*   **TensorBoard**：`tensorboard --logdir runs` 即可对比不同实验曲线。

如果你还要补**CIFAR-10 vs CIFAR-100**在**Option A/B/C**上的系统化栅格实验（自动生成多配置并运行），我可以再加一个 `scripts/sweep.py` 与 YAML 模版与命名策略，一键全跑；需要就说。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)