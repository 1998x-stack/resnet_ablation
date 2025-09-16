from __future__ import annotations
from typing import Dict
import torch
from torch import nn
from ..metrics import topk_correct


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    top1, n = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        correct1, = topk_correct(logits, targets, (1,))
        top1 += correct1.item()
        n += images.size(0)
    return {"acc1": top1 / n}