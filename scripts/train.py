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