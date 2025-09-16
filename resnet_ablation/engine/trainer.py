from __future__ import annotations
from typing import Dict
from pathlib import Path
import random
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from ..metrics import topk_correct
from ..augment import mixup as do_mixup, cutmix as do_cutmix
from ..losses import SoftTargetCrossEntropy


class Trainer:
    """通用训练引擎：AMP / 断点续训 / TensorBoard / Loguru / Mixup/CutMix/LabelSmoothing / GradClip。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 sched, device: torch.device, out_dir: str, tb_dir: str,
                 amp: bool = True, clip_grad_norm: float | None = 1.0,
                 num_classes: int = 10,
                 label_smoothing: float = 0.0,
                 mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.sched = sched
        self.device = device
        self.amp = amp
        self.clip_grad_norm = clip_grad_norm
        self.scaler = GradScaler(enabled=amp)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(out_dir)
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(tb_dir))

        # 增强/损失配置
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # 当使用 Mixup/CutMix 时采用 soft CE
        self.soft_ce = SoftTargetCrossEntropy()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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

    def _maybe_augment(self, images: torch.Tensor, targets: torch.Tensor):
        """根据 alpha 启用 Mixup/CutMix；当两者都>0时随机二选一。"""
        if self.mixup_alpha <= 0.0 and self.cutmix_alpha <= 0.0:
            return images, targets, False  # no soft labels
        use_mixup = False
        if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
            use_mixup = bool(random.getrandbits(1))
        else:
            use_mixup = self.mixup_alpha > 0.0

        if use_mixup:
            images, soft_targets = do_mixup(images, targets, self.num_classes, self.mixup_alpha, smoothing=0.0)
        else:
            images, soft_targets = do_cutmix(images, targets, self.num_classes, self.cutmix_alpha, smoothing=0.0)
        return images, soft_targets, True

    def train_one_epoch(self, epoch: int, loader, _) -> Dict[str, float]:
        self.model.train()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # 保存硬标签用于精度指标
            hard_targets = targets.detach()

            # 可能应用 Mixup/CutMix
            images, targets_or_soft, is_soft = self._maybe_augment(images, targets)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(images)
                if is_soft:
                    loss = self.soft_ce(logits, targets_or_soft)
                else:
                    loss = self.ce(logits, targets_or_soft)

            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 若使用 Cosine 等 per-iter 调度器在此 step
            if self.sched is not None and hasattr(self.sched, "step") and not isinstance(self.sched, torch.optim.lr_scheduler.MultiStepLR):
                self.sched.step()

            with torch.no_grad():
                correct1, = topk_correct(logits, hard_targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs

            if step % 50 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Step {step}: loss={loss.item():.4f}, acc1={(correct1/bs*100):.2f}%, lr={lr:.4e}")

        return {"loss": loss_sum / n, "acc1": acc1_sum / n}

    @torch.no_grad()
    def validate(self, epoch: int, loader, _) -> Dict[str, float]:
        self.model.eval()
        loss_sum, acc1_sum, n = 0.0, 0.0, 0
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.ce(logits, targets)  # 验证阶段不使用 mix/cut
            correct1, = topk_correct(logits, targets, (1,))
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc1_sum += correct1.item()
            n += bs
        return {"val_loss": loss_sum / n, "val_acc1": acc1_sum / n}