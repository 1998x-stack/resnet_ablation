"""Factory registry and auditable data/schedule contracts.

Train/validation are disjoint *within CIFAR training data*. CIFAR test images are
never downloaded by a training run. Original paper used per-pixel mean subtraction;
this reproducible implementation uses published per-channel means and documents
the approximation in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from resnet_ablation.models.factory import build_model


class Factory:
    def __init__(self, name: str):
        self.name = name
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def decorator(fn: Callable[..., Any]):
            if name in self._builders:
                raise KeyError(f"duplicate {self.name} factory: {name}")
            self._builders[name] = fn
            return fn
        return decorator

    def build(self, name: str, **kwargs: Any) -> Any:
        try:
            return self._builders[name](**kwargs)
        except KeyError as exc:
            raise ValueError(f"unknown {self.name} '{name}', options: {sorted(self._builders)}") from exc

    def names(self) -> list[str]:
        return sorted(self._builders)


MODELS = Factory("model")
DATASETS = Factory("dataset")
OPTIMIZERS = Factory("optimizer")
LOSSES = Factory("loss")
SCHEDULES = Factory("schedule")


@MODELS.register("resnet")
def model_factory(cfg):
    return build_model(cfg)


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    test: Dataset | None
    split_sha256: str
    split_seed: int
    normalization: str
    train_size: int
    val_size: int
    test_size: int


def split_indices(size: int, val_size: int, seed: int) -> tuple[list[int], list[int], str]:
    if not (0 < val_size < size):
        raise ValueError(f"validation size must be in (0, {size})")
    permutation = np.random.default_rng(seed).permutation(size)
    val = sorted(int(x) for x in permutation[:val_size])
    train = sorted(int(x) for x in permutation[val_size:])
    checksum = hashlib.sha256(json.dumps({"train": train, "val": val}, separators=(",", ":")).encode()).hexdigest()
    assert len(train) + len(val) == size and not set(train).intersection(val)
    return train, val, checksum


def _cifar(kind: str, root: str, split_seed: int, val_size: int, final_test: bool) -> DataBundle:
    import torchvision as tv
    means = {"cifar10": (0.4914, 0.4822, 0.4465),
             "cifar100": (0.5071, 0.4867, 0.4408)}
    # Fixed across all conditions in a matched sweep; this is modern per-channel
    # normalization and NOT the original paper's exact per-pixel subtraction.
    stds = {"cifar10": (0.2470, 0.2435, 0.2616),
            "cifar100": (0.2675, 0.2565, 0.2761)}
    norm = tv.transforms.Normalize(means[kind], stds[kind])
    train_tf = tv.transforms.Compose([tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(), tv.transforms.ToTensor(), norm])
    eval_tf = tv.transforms.Compose([tv.transforms.ToTensor(), norm])
    cls = tv.datasets.CIFAR10 if kind == "cifar10" else tv.datasets.CIFAR100
    augmented = cls(root=root, train=True, download=True, transform=train_tf)
    clean = cls(root=root, train=True, download=False, transform=eval_tf)
    ti, vi, digest = split_indices(len(augmented), val_size, split_seed)
    train, val = Subset(augmented, ti), Subset(clean, vi)
    test = cls(root=root, train=False, download=True, transform=eval_tf) if final_test else None
    return DataBundle(train, val, test, digest, split_seed, "channel_mean_std_not_original_pixel_mean",
                      len(ti), len(vi), len(test) if test is not None else 0)


@DATASETS.register("cifar10")
def cifar10(root: str, split_seed: int = 2026, val_size: int = 5000,
            final_test: bool = False) -> DataBundle:
    return _cifar("cifar10", root, split_seed, val_size, final_test)


@DATASETS.register("cifar100")
def cifar100(root: str, split_seed: int = 2026, val_size: int = 5000,
             final_test: bool = False) -> DataBundle:
    return _cifar("cifar100", root, split_seed, val_size, final_test)


@DATASETS.register("imagenet")
def imagenet(root: str, split_seed: int = 2026, val_size: int = 5000,
             final_test: bool = False) -> DataBundle:
    """ImageFolder: root/train and root/val. The 'val' split is benchmark validation,
    NOT a held-out official test set; test is unavailable in local ImageNet."""
    from pathlib import Path
    import torchvision as tv
    tf = tv.transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = tf.Compose([tf.RandomResizedCrop(224), tf.RandomHorizontalFlip(),
                           tf.ToTensor(), tf.Normalize(mean, std)])
    val_tf = tf.Compose([tf.Resize(256), tf.CenterCrop(224),
                         tf.ToTensor(), tf.Normalize(mean, std)])
    tr = tv.datasets.ImageFolder(str(Path(root) / "train"), transform=train_tf)
    va = tv.datasets.ImageFolder(str(Path(root) / "val"), transform=val_tf)
    if tr.class_to_idx != va.class_to_idx:
        raise ValueError("ImageNet train/val class mapping differs")
    digest = hashlib.sha256(json.dumps({"train_classes": tr.class_to_idx, "seed": split_seed},
                                        sort_keys=True).encode()).hexdigest()
    return DataBundle(tr, va, None, digest, split_seed, "imagenet_channel_mean_std",
                      len(tr), len(va), 0)


@OPTIMIZERS.register("sgd")
def sgd(model: nn.Module, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)


@OPTIMIZERS.register("adamw")
def adamw(model: nn.Module, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


@LOSSES.register("cross_entropy")
def cross_entropy(label_smoothing: float = 0.0):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


class StepSchedule:
    """Explicit iteration -> LR. No hidden stepping state; safe to resume at step N."""
    def __init__(self, name: str, base_lr: float, total_steps: int,
                 steps_per_epoch: int, milestones: tuple[int, ...] = (),
                 gamma: float = 0.1, warmup110: bool = False):
        if total_steps <= 0 or steps_per_epoch <= 0 or base_lr <= 0:
            raise ValueError("invalid scheduler horizon or LR")
        self.name, self.base_lr, self.total_steps = name, base_lr, total_steps
        self.steps_per_epoch, self.milestones, self.gamma = steps_per_epoch, milestones, gamma
        self.warmup110 = warmup110

    def lr(self, completed_steps: int) -> float:
        if not 0 <= completed_steps < self.total_steps:
            raise ValueError("step is outside run horizon")
        if self.name == "paper_cifar_64k":
            # §4.2: base LR 0.1; drops at 32k and 48k; 110 warmup first ~400.
            rate = self.base_lr * (0.1 if completed_steps >= 32000 else 1.0)
            rate *= 0.1 if completed_steps >= 48000 else 1.0
            return 0.01 if self.warmup110 and completed_steps < 400 else rate
        if self.name == "multistep_epoch":
            epochs_completed = completed_steps // self.steps_per_epoch
            return self.base_lr * self.gamma ** sum(epochs_completed >= m for m in self.milestones)
        if self.name == "cosine_step":
            import math
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * completed_steps / self.total_steps))
        raise ValueError(f"unknown schedule {self.name}")

    def set_lr(self, optimizer, completed_steps: int) -> float:
        lr = self.lr(completed_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


@SCHEDULES.register("paper_cifar_64k")
def paper_cifar_schedule(cfg, total_steps: int, steps_per_epoch: int, warmup110: bool = False):
    if cfg.lr != 0.1 or total_steps != 64000:
        raise ValueError("paper_cifar_64k requires lr=0.1 and 64000 steps; use other schedule for smoke runs")
    return StepSchedule("paper_cifar_64k", cfg.lr, total_steps, steps_per_epoch, warmup110=warmup110)


@SCHEDULES.register("multistep_epoch")
def multistep_schedule(cfg, total_steps: int, steps_per_epoch: int, warmup110: bool = False):
    return StepSchedule("multistep_epoch", cfg.lr, total_steps, steps_per_epoch,
                        tuple(cfg.milestones), cfg.gamma)


@SCHEDULES.register("cosine_step")
def cosine_schedule(cfg, total_steps: int, steps_per_epoch: int, warmup110: bool = False):
    return StepSchedule("cosine_step", cfg.lr, total_steps, steps_per_epoch)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset: Dataset, batch_size: int, workers: int, seed: int,
           shuffle: bool, device: torch.device) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, pin_memory=device.type == "cuda",
                      persistent_workers=False, worker_init_fn=worker_seed)
