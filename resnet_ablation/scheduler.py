"""Learning-rate scheduler construction with linear warmup.

Stepping policy (important)
---------------------------
All schedulers produced by :func:`build_scheduler` are stepped exactly ONCE
per training epoch, from the training script (``scripts/train.py``), never
from inside the trainer. This is the correct per-epoch contract for both the
base schedules (``CosineAnnealingLR`` with ``T_max=total_epochs`` and
``MultiStepLR`` with epoch-based milestones) and the warmup wrapper.

Historically the trainer also stepped Cosine schedules on *every training
iteration* inside ``train_one_epoch``. Combined with ``T_max=total_epochs``
that over-annealed the cosine curve by a factor of ``iters_per_epoch`` —
severe for real runs. That per-iteration stepping has been removed; all
schedules now advance once per epoch, symmetrically.

Warmup semantics
----------------
When ``warmup_epochs > 0`` the schedule is a ``SequentialLR`` that runs a
short linear warmup (``LambdaLR``) for ``warmup_epochs`` epochs and then hands
off to the main schedule. The warmup lambda is
``factor(step) = min(1.0, (step + 1) / warmup_epochs)`` which is strictly
positive on the first warmup epoch and reaches exactly 1.0 at the warmup
boundary, so the main schedule resumes from the base LR without a discontinuity
(or a zero-LR first step).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    MultiStepLR,
    SequentialLR,
    _LRScheduler,
)


def linear_warmup(n: int) -> Callable[[int], float]:
    """Return a ``step -> factor`` lambda for a linear 0->1 warmup over ``n`` epochs.

    ``(step + 1) / max(1, n)`` guarantees a strictly positive LR on the very
    first warmup step and lands at exactly 1.0 on the ``n``-th warmup step.
    """

    def _fac(step: int) -> float:
        return min(1.0, (step + 1) / max(1, n))

    return _fac


def build_scheduler(
    optimizer: Optimizer,
    opt_cfg,
    total_epochs: int,
) -> _LRScheduler:
    """Build an LR schedule honoring ``opt_cfg.warmup_epochs``.

    Returns a plain cosine/multistep schedule when ``warmup_epochs <= 0``
    (identical behaviour to a plain schedule), or a ``SequentialLR`` wrapping a
    linear warmup followed by the base schedule when warmup is enabled.
    """
    if opt_cfg.sched == "cosine":
        main: _LRScheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    else:
        main = MultiStepLR(
            optimizer, milestones=list(opt_cfg.milestones), gamma=opt_cfg.gamma
        )

    warm = int(opt_cfg.warmup_epochs)
    if warm <= 0:
        return main

    warm_sched = LambdaLR(optimizer, lr_lambda=linear_warmup(warm))
    return SequentialLR(optimizer, [warm_sched, main], milestones=[warm])