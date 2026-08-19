from __future__ import annotations
from typing import Dict
import torch
from torch import nn
from ..metrics import topk_correct


@torch.no_grad()
def evaluate(model: nn.Module, loader, topk=(1,)) -> Dict[str, float]:
    """Evaluate and return acc at requested topk levels (default top-1)."""
    model.eval()
    device = next(model.parameters()).device
    sums = {k: 0.0 for k in topk}
    n = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
        corrects = topk_correct(logits, targets, topk)
        for k, c in zip(topk, corrects):
            sums[k] += c.item()
        n += images.size(0)
    return {f"acc{k}": sums[k] / n for k in topk}