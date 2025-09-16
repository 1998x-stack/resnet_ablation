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