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