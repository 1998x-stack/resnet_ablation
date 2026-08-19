import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from resnet_ablation.config import OptimConfig
from resnet_ablation.scheduler import build_scheduler


def _make(epochs=200, warmup=5, sched="cosine"):
    o = OptimConfig(lr=0.1, warmup_epochs=warmup, sched=sched)
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=o.lr)
    return opt, build_scheduler(opt, o, total_epochs=epochs)


def test_warmup_first_step_positive():
    opt, sched = _make(warmup=5)
    # first step inside warmup phase gives a tiny-but-positive LR, well below base
    sched.step()
    lr = opt.param_groups[0]["lr"]
    assert 0.0 < lr < 0.1


def test_warmup_ends_at_base():
    opt, sched = _make(warmup=5)
    # after exactly warmup_epochs per-epoch steps, warmup => 1.0 and main resumes from base
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_no_warmup_equals_cosine():
    o = OptimConfig(lr=0.1, warmup_epochs=0, sched="cosine")
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=o.lr)
    sched = build_scheduler(opt, o, total_epochs=200)
    ref = CosineAnnealingLR(opt, T_max=200)
    sched.step()
    ref.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(ref.get_last_lr()[0])


def test_multistep_kept_for_multistep():
    o = OptimConfig(lr=0.1, warmup_epochs=0, sched="multistep", milestones=[10], gamma=0.1)
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=o.lr)
    sched = build_scheduler(opt, o, total_epochs=200)
    # warmup=0 => plain MultiStepLR; stepping to (and including) milestone 10 lowers LR by gamma
    for _ in range(10):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 0.1)