"""Factory-mode training: step-accurate LR, separate validation, recoverable epochs.

Usage: python -m resnet_ablation.research.runner run --config CONFIG --schedule
paper_cifar_64k --steps 64000 --run-dir runs/paper/resnet20_seed1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from resnet_ablation.config import Config
from .factories import (MODELS, DATASETS, OPTIMIZERS, LOSSES, SCHEDULES,
                        loader, seed_all)
from resnet_ablation.models.blocks import BasicBlock, Bottleneck


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf8")
    os.replace(temp, path)


def _event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def _save(path: Path, checkpoint: dict) -> None:
    """Atomic local replacement; never expose half-written checkpoints."""
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_config(config_path: str, seed: int | None) -> Config:
    cfg = Config.from_yaml(config_path)
    if seed is not None:
        cfg.train.seed = seed
    cfg.validate()
    if cfg.data.mixup_alpha or cfg.data.cutmix_alpha or cfg.train.label_smoothing:
        raise ValueError("paper-control runner forbids mixup/cutmix/smoothing; use legacy extension runs")
    if cfg.data.aug != "standard":
        raise ValueError("paper-control data factory currently supports only standard augmentation")
    if cfg.data.name == "imagenet" and cfg.model.num_classes != 1000:
        raise ValueError("ImageNet requires num_classes=1000")
    if cfg.data.name == "cifar10" and cfg.model.num_classes != 10:
        raise ValueError("CIFAR-10 requires 10 classes")
    if cfg.data.name == "cifar100" and cfg.model.num_classes != 100:
        raise ValueError("CIFAR-100 requires 100 classes")
    return cfg


def _metadata(cfg: Config, schedule: str, steps: int, data, model, device) -> dict:
    return {"config": cfg.as_dict(), "schedule": schedule, "target_steps": steps,
            "git_sha": _git_sha(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "python": platform.python_version(),
            "host": platform.platform(), "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_size": data.train_size, "val_size": data.val_size,
            "test_size": data.test_size, "split_sha256": data.split_sha256,
            "split_seed": data.split_seed, "normalization": data.normalization,
            "paper_url": "https://arxiv.org/abs/1512.03385",
            "fidelity": "matched-modern-protocol"
            if schedule != "paper_cifar_64k" else "paper-iteration-schedule-not-exact-preprocessing"}


def _evaluate(model, dataset, cfg, device) -> dict[str, float]:
    model.eval()
    data_loader = loader(dataset, cfg.train.batch_size, cfg.train.num_workers,
                         999, False, device)
    loss_sum = 0.0
    hits1 = hits5 = n = 0
    with torch.no_grad():
        for x, target in data_loader:
            x, target = x.to(device), target.to(device)
            logits = model(x)
            loss_sum += torch.nn.functional.cross_entropy(logits, target, reduction="sum").item()
            ranks = logits.topk(min(5, logits.shape[1]), dim=1).indices
            hits1 += int((ranks[:, 0] == target).sum().item())
            hits5 += int((ranks == target[:, None]).any(dim=1).sum().item())
            n += target.shape[0]
    if n == 0:
        raise ValueError("empty validation dataset")
    return {"loss": loss_sum / n, "acc1": hits1 / n, "acc5": hits5 / n}


def _gradient_metrics(model) -> dict[str, float]:
    total = torch.zeros((), device=next(model.parameters()).device)
    stages = {"layer1": [], "layer2": [], "layer3": [], "layer4": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm().item()
        total += parameter.grad.detach().float().square().sum()
        for stage in stages:
            if name.startswith(stage + "."):
                stages[stage].append(norm ** 2)
    result = {"grad/global_l2": float(total.sqrt().item())}
    result.update({f"grad/{k}_l2": sum(v) ** 0.5 for k, v in stages.items() if v})
    return result


def run(config_path: str, run_dir: str, schedule: str, steps: int | None,
        seed: int | None = None, split_seed: int = 2026, val_size: int = 5000,
        resume: bool = False, device_str: str = "auto") -> dict:
    cfg = _run_config(config_path, seed)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    seed_all(cfg.train.seed)
    data = DATASETS.build(cfg.data.name, root=cfg.data.root, split_seed=split_seed,
                          val_size=val_size, final_test=False)
    # No test objects exist in training mode: protects against repeated test tuning.
    model = MODELS.build("resnet", cfg=cfg).to(device)
    optimizer = OPTIMIZERS.build(cfg.optim.name, model=model, cfg=cfg.optim)
    steps_per_epoch = (data.train_size + cfg.train.batch_size - 1) // cfg.train.batch_size
    target = steps or cfg.train.epochs * steps_per_epoch
    if schedule == "paper_cifar_64k" and cfg.data.name != "cifar10":
        raise ValueError("paper_cifar_64k applies only to CIFAR-10")
    sched = SCHEDULES.build(schedule, cfg=cfg.optim, total_steps=target,
                            steps_per_epoch=steps_per_epoch,
                            warmup110=cfg.model.arch == "resnet110_cifar")
    criterion = LOSSES.build("cross_entropy", label_smoothing=0.0)
    out = Path(run_dir)
    if out.exists() and not resume and any(out.iterdir()):
        raise FileExistsError(f"run directory is not empty: {out}; specify --resume or a new path")
    out.mkdir(parents=True, exist_ok=True)
    manifest = _metadata(cfg, schedule, target, data, model, device)
    manifest_bytes = json.dumps(manifest["config"], sort_keys=True).encode()
    manifest["config_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if resume:
        previous = json.loads((out / "manifest.json").read_text())
        for key in ("config_sha256", "split_sha256", "schedule", "target_steps"):
            if previous[key] != manifest[key]:
                raise ValueError(f"resume metadata mismatch: {key}")
    else:
        _json(out / "manifest.json", manifest)
    from resnet_ablation.logger import setup_logger
    setup_logger(str(out))
    tb = SummaryWriter(log_dir=str(out / "tensorboard"))
    use_amp = bool(cfg.train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    global_step, first_epoch, best = 0, 0, -1.0
    if resume:
        # Loading arbitrary pickle checkpoints is unsafe: only use trusted self-generated files.
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        global_step, first_epoch, best = state["global_step"], state["next_epoch"], state["best"]
        _restore_rng(state["rng"])
    log_path = out / "metrics.jsonl"
    diagnostic_blocks = [(name, block) for name, block in model.named_modules()
                         if isinstance(block, (BasicBlock, Bottleneck))]
    logger.info(f"factory={schedule}, arch={cfg.model.arch}, steps={target}, start_step={global_step}")
    for epoch in range(first_epoch, cfg.train.epochs):
        if global_step >= target:
            break
        model.train()
        epoch_loader = loader(data.train, cfg.train.batch_size, cfg.train.num_workers,
                              cfg.train.seed * 1000003 + epoch, True, device)
        total_loss, correct, samples = 0.0, 0, 0
        started = time.perf_counter()
        last_grad = {}
        lr_last = None
        for x, y in epoch_loader:
            if global_step >= target:
                break
            lr_used = sched.set_lr(optimizer, global_step)
            lr_last = lr_used
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            diagnose = (global_step + 1) % cfg.train.log_interval == 0
            if diagnose:
                for _, block in diagnostic_blocks:
                    block.capture_statistics = True
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            if cfg.train.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            if (global_step + 1) % cfg.train.log_interval == 0:
                last_grad = _gradient_metrics(model)
            scaler.step(optimizer)
            scaler.update()
            count = len(y)
            total_loss += float(loss.detach()) * count
            correct += int((logits.detach().argmax(1) == y).sum())
            samples += count
            global_step += 1
            if global_step % cfg.train.log_interval == 0:
                event = {"kind": "train_step", "step": global_step, "epoch": epoch + 1,
                         "lr_used": lr_used, "loss": float(loss.detach()),
                         "acc1": float((logits.detach().argmax(1) == y).float().mean())}
                event.update(last_grad)
                # Proxy for paper Figure 7: RMS of F(x) divided by RMS of shortcut.
                # Computed only on diagnostic steps; not the paper\'s exact statistic.
                per_stage = {}
                for block_name, block in diagnostic_blocks:
                    if hasattr(block, "last_residual_ratio"):
                        stage = block_name.split(".")[0]
                        per_stage.setdefault(stage, []).append(block.last_residual_ratio)
                    block.capture_statistics = False
                for stage, values in per_stage.items():
                    event[f"residual/{stage}_rms_ratio"] = sum(values) / len(values)
                _event(log_path, event)
                tb.add_scalar("step/loss", event["loss"], global_step)
                tb.add_scalar("step/lr_used", lr_used, global_step)
                for k, v in last_grad.items():
                    tb.add_scalar(k, v, global_step)
                for k, v in event.items():
                    if k.startswith("residual/"):
                        tb.add_scalar(k, v, global_step)
        if samples == 0:
            break
        elapsed = time.perf_counter() - started
        stats = {"kind": "train_epoch", "epoch": epoch + 1, "step": global_step,
                 "loss": total_loss / samples, "acc1": correct / samples,
                 "lr_used_last": lr_last, "images_per_sec": samples / max(elapsed, 1e-6),
                 "seconds": elapsed}
        _event(log_path, stats)
        tb.add_scalar("train/loss", stats["loss"], epoch + 1)
        tb.add_scalar("train/acc1", stats["acc1"], epoch + 1)
        tb.add_scalar("train/error1", 1 - stats["acc1"], epoch + 1)
        tb.add_scalar("train/images_per_sec", stats["images_per_sec"], epoch + 1)
        is_best = False
        if (epoch + 1) % cfg.train.val_interval == 0 or global_step == target:
            v = _evaluate(model, data.val, cfg, device)
            v.update({"kind": "val_epoch", "epoch": epoch + 1, "step": global_step})
            _event(log_path, v)
            tb.add_scalar("val/loss", v["loss"], epoch + 1)
            tb.add_scalar("val/acc1", v["acc1"], epoch + 1)
            tb.add_scalar("val/error1", 1 - v["acc1"], epoch + 1)
            is_best = v["acc1"] > best
            best = max(best, v["acc1"])
            logger.info(f"epoch={epoch + 1} step={global_step} train_loss={stats['loss']:.4f}"
                        f" val_acc1={v['acc1']:.4f}, best={best:.4f}")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "global_step": global_step,
                 "next_epoch": epoch + 1, "best": best, "rng": _rng(),
                 "config_sha256": manifest["config_sha256"],
                 "split_sha256": data.split_sha256}
        _save(out / "last.pt", state)
        if is_best:
            _save(out / "best.pt", state)
        tb.flush()
    tb.close()
    result = {"run_dir": str(out), "completed_steps": global_step,
              "target_steps": target, "best_val_acc1": best,
              "is_complete": global_step == target}
    _json(out / "status.json", result)
    return result


def final_test(config_path: str, run_dir: str, device_str: str = "auto") -> dict:
    """Opt-in ONE final test after experiment selection; no training or tuning here."""
    out = Path(run_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    cfg = _run_config(config_path, None)
    cfg.train.seed = manifest["config"]["train"]["seed"]
    if hashlib.sha256(json.dumps(cfg.as_dict(), sort_keys=True).encode()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("evaluation config differs from trained run")
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available()
                          else "cpu" if device_str == "auto" else device_str)
    if cfg.data.name not in ("cifar10", "cifar100"):
        raise ValueError("only CIFAR supplies a local sealed test set in this runner")
    data = DATASETS.build(cfg.data.name, root=cfg.data.root,
                          split_seed=manifest["split_seed"], final_test=True)
    model = MODELS.build("resnet", cfg=cfg).to(device)
    # Trusted generated checkpoints only.
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    metrics = _evaluate(model, data.test, cfg, device)
    result = {"kind": "final_test", **metrics, "checkpoint": "best.pt",
              "model_step": state["global_step"]}
    _json(out / "final_test.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Factory-mode paper vs extension ResNet")
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("run")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--schedule", choices=SCHEDULES.names(), default="multistep_epoch")
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--seed", type=int)
    train.add_argument("--split-seed", type=int, default=2026)
    train.add_argument("--val-size", type=int, default=5000)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", default="auto")
    test = sub.add_parser("final-test")
    test.add_argument("--config", required=True)
    test.add_argument("--run-dir", required=True)
    test.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.cmd == "run":
        result = run(args.config, args.run_dir, args.schedule, args.steps,
                     args.seed, args.split_seed, args.val_size, args.resume, args.device)
    else:
        result = final_test(args.config, args.run_dir, args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
