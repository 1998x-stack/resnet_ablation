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