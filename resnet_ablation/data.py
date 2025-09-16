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