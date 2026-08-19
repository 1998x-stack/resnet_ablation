from __future__ import annotations
from typing import Tuple
import torch


@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> Tuple[torch.Tensor, ...]:
    """Return per-k counts of correct top-k predictions as a tuple of scalars.

    Args:
        logits: (N, C) model output.
        targets: (N,) integer class labels.
        topk: tuple of k values whose correct counts are returned, e.g. (1, 5).

    Returns:
        One 0-dim tensor sum per k in ``topk``.
    """
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum())
    return tuple(res)