from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from torch.optim import SGD
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device, count_params
from resnet_ablation.data import build_dataloaders
from resnet_ablation.scheduler import build_scheduler
from resnet_ablation.engine.trainer import Trainer
from resnet_ablation.models.factory import build_model


def load_config(path: str) -> Config:
    return Config.from_yaml(path)


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

    # 转储 resolved config（含运行环境），保证可复现性
    runtime = {"device": str(device),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seed": cfg.train.seed}
    full = cfg.as_dict()
    full["runtime"] = runtime
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.out_dir) / "config.json").write_text(json.dumps(full, indent=2))

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Dataset={cfg.data.name} | aug={cfg.data.aug} | train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    # Model
    model = build_model(cfg).to(device)
    logger.info(f"Model: {cfg.model.arch}, params: {count_params(model)/1e6:.2f}M | attn={cfg.model.attention} | dpr={cfg.model.drop_path_rate}")

    # Optim & Sched
    opt = SGD(model.parameters(), lr=cfg.optim.lr, momentum=cfg.optim.momentum,
              weight_decay=cfg.optim.weight_decay, nesterov=cfg.optim.nesterov)
    sched = build_scheduler(opt, cfg.optim, cfg.train.epochs)

    trainer = Trainer(
        model, opt, sched, device,
        out_dir=cfg.train.out_dir, tb_dir=cfg.train.tb_dir,
        amp=cfg.train.amp, clip_grad_norm=cfg.train.clip_grad_norm,
        num_classes=cfg.model.num_classes,
        label_smoothing=cfg.train.label_smoothing,
        mixup_alpha=cfg.data.mixup_alpha,
        cutmix_alpha=cfg.data.cutmix_alpha,
        log_interval=cfg.train.log_interval
    )

    # 初始化 CSV 元数据（时间戳 + 运行信息），供 write_results_row 使用
    trainer._csv_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    trainer._csv_meta = {"dataset": cfg.data.name, "arch": cfg.model.arch,
                         "seed": cfg.train.seed, "attn": cfg.model.attention,
                         "shortcut": cfg.model.shortcut}

    start_epoch = 0
    best_acc = 0.0
    if cfg.train.resume:
        start_epoch, best_acc = trainer.load(cfg.train.resume)

    topk = (1, 5) if cfg.data.name == "imagenet" else (1,)
    for epoch in range(start_epoch, cfg.train.epochs):
        stats = trainer.train_one_epoch(epoch, train_loader)
        # 精确 per-epoch stepping: 所有调度器(含 warmup/cosine)每次 epoch 恰好前进一次
        sched.step()
        if (epoch + 1) % cfg.train.val_interval == 0:
            v = trainer.validate(epoch, val_loader, topk)
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
            trainer.save(epoch, {"is_best": is_best, "best_acc": best_acc})
            trainer.write_results_row(epoch, stats, v, opt.param_groups[0]["lr"])

    logger.info("Training done.")


if __name__ == "__main__":
    main()