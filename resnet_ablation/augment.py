from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def one_hot(labels: Tensor, num_classes: int, smoothing: float = 0.0) -> Tensor:
    """将整数标签变为 one-hot（可选标签平滑）。"""
    with torch.no_grad():
        y = F.one_hot(labels, num_classes=num_classes).float()
        if smoothing > 0.0:
            y = y * (1.0 - smoothing) + smoothing / num_classes
    return y


def rand_bbox(W: int, H: int, lam: float) -> tuple[int, int, int, int]:
    """CutMix 随机框生成。"""
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup(images: Tensor, targets: Tensor, num_classes: int, alpha: float, smoothing: float = 0.0):
    """Mixup 增强。返回混合后的图像与“软标签”"""
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for mixup.")
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    indices = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[indices]
    y1 = one_hot(targets, num_classes, smoothing)
    y2 = one_hot(targets[indices], num_classes, smoothing)
    y = lam * y1 + (1 - lam) * y2
    return mixed, y


def cutmix(images: Tensor, targets: Tensor, num_classes: int, alpha: float, smoothing: float = 0.0):
    """CutMix 增强。返回混合后的图像与“软标签”"""
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for cutmix.")
    lam = torch._sample_dirichlet(torch.tensor([alpha, alpha]))[0].item()
    indices = torch.randperm(images.size(0), device=images.device)
    W, H = images.size(3), images.size(2)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    # 根据实际区域重新计算 lam
    box_area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - box_area / float(W * H)
    y1 = one_hot(targets, num_classes, smoothing)
    y2 = one_hot(targets[indices], num_classes, smoothing)
    y = lam_adj * y1 + (1 - lam_adj) * y2
    return mixed, y