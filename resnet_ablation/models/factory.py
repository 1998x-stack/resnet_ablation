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