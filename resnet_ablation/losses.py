from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):
    """用于 Mixup/CutMix 的交叉熵（targets 为概率分布）。"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        loss = -(targets * logp).sum(dim=1).mean()
        return loss