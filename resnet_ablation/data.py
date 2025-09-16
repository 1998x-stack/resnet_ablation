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