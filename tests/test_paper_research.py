"""Contract tests tied to arXiv:1512.03385; no downloads or benchmark claims."""
from __future__ import annotations

import pytest
import torch

from resnet_ablation.config import Config
from resnet_ablation.models.blocks import BasicBlock, Bottleneck, ShortcutA
from resnet_ablation.models.factory import build_model
from resnet_ablation.research.factories import (MODELS, DATASETS, OPTIMIZERS, LOSSES,
                                                SCHEDULES, StepSchedule, split_indices)
from resnet_ablation.research.report import matched_factors
from resnet_ablation.research.sweep import build_plan


def test_factory_has_explicit_registered_components():
    assert "resnet" in MODELS.names()
    assert {"cifar10", "cifar100", "imagenet"} <= set(DATASETS.names())
    assert "sgd" in OPTIMIZERS.names()
    assert "cross_entropy" in LOSSES.names()
    assert "paper_cifar_64k" in SCHEDULES.names()


@pytest.mark.parametrize("depth,blocks", [(20, 9), (32, 15), (44, 21), (56, 27), (110, 54)])
def test_cifar_6n_plus_2(depth, blocks):
    cfg = Config()
    cfg.model.arch = f"resnet{depth}_cifar"
    model = build_model(cfg)
    assert sum(isinstance(x, BasicBlock) for x in model.modules()) == blocks


def test_plain_and_option_a_identical_parameter_budget_and_finite_backward():
    cfg = Config()
    cfg.model.arch = "resnet20_cifar"
    cfg.model.shortcut = "A"
    cfg.model.residual = True
    residual = build_model(cfg)
    cfg.model.residual = False
    plain = build_model(cfg)
    assert sum(x.numel() for x in residual.parameters()) == sum(x.numel() for x in plain.parameters())
    x = torch.randn(2, 3, 32, 32)
    for net in (plain, residual):
        loss = net(x).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_abc_shortcut_semantics_at_equal_and_changed_shapes():
    a = BasicBlock(16, 16, shortcut="A")
    b = BasicBlock(16, 16, shortcut="B")
    c = BasicBlock(16, 16, shortcut="C")
    assert a.downsample is None and b.downsample is None
    assert isinstance(c.downsample, torch.nn.Sequential)
    down_a = BasicBlock(16, 32, stride=2, shortcut="A")
    down_b = BasicBlock(16, 32, stride=2, shortcut="B")
    assert isinstance(down_a.downsample, ShortcutA)
    assert isinstance(down_b.downsample, torch.nn.Sequential)
    assert sum(p.numel() for p in c.parameters()) > sum(p.numel() for p in b.parameters())


def test_shortcut_C_on_each_block_not_only_downsampling():
    cfg = Config()
    cfg.model.arch = "resnet20_cifar"
    cfg.model.shortcut = "C"
    model = build_model(cfg)
    assert all(m.downsample is not None for m in model.modules() if isinstance(m, BasicBlock))


def test_bottleneck_original_vs_torchvision_stride_location():
    original = Bottleneck(256, 128, stride=2, bottleneck_stride="v1")
    modern = Bottleneck(256, 128, stride=2, bottleneck_stride="v1.5")
    assert (original.conv1.stride, original.conv2.stride) == ((2, 2), (1, 1))
    assert (modern.conv1.stride, modern.conv2.stride) == ((1, 1), (2, 2))
    projected = Bottleneck(256, 64, shortcut="C", bottleneck_stride="v1")
    assert projected.downsample is not None


def test_cifar_train_val_split_fixed_disjoint_and_hashed():
    a, b, digest = split_indices(50000, 5000, seed=2026)
    x, y, again = split_indices(50000, 5000, seed=2026)
    assert len(a) == 45000 and len(b) == 5000
    assert set(a).isdisjoint(b)
    assert (a, b, digest) == (x, y, again)
    assert digest != split_indices(50000, 5000, seed=2027)[2]
    with pytest.raises(ValueError):
        split_indices(100, 100, 1)


def test_original_cifar_paper_lr_at_boundaries():
    schedule = StepSchedule("paper_cifar_64k", 0.1, 64000, 352)
    expected = {0: 0.1, 31999: 0.1, 32000: 0.01, 47999: 0.01,
                48000: 0.001, 63999: 0.001}
    for step, value in expected.items():
        assert schedule.lr(step) == pytest.approx(value)
    warm = StepSchedule("paper_cifar_64k", 0.1, 64000, 352, warmup110=True)
    assert warm.lr(0) == warm.lr(399) == 0.01
    assert warm.lr(400) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        schedule.lr(64000)


def test_report_rejects_uncontrolled_changes():
    cfg = Config().as_dict()
    x = {"config": cfg, "split_sha256": "same", "schedule": "paper_cifar_64k",
         "target_steps": 64000}
    import copy
    y = copy.deepcopy(x)
    y["config"]["model"]["residual"] = False
    assert matched_factors(x, y, {"model.residual"}) == ["model.residual"]
    with pytest.raises(ValueError):
        matched_factors(x, y, set())
    y["split_sha256"] = "other"
    with pytest.raises(ValueError):
        matched_factors(x, y, {"model.residual"})


def test_sweep_plan_is_deterministic_and_separates_config_from_run_directory():
    plan = build_plan("configs/paper/cifar10_depth_matrix.yaml", "runs/paper")
    assert len(plan) == 18
    assert len({x["run_dir"] for x in plan}) == len(plan)
    assert plan[0]["config_file"] not in plan[0]["run_dir"]
    assert plan[0]["schedule"] == "paper_cifar_64k"


def test_profile_reports_actual_macs_and_restores_training_mode():
    from resnet_ablation.research.profile import analyze
    cfg = Config()
    cfg.model.arch = "resnet20_cifar"
    model = build_model(cfg)
    assert model.training
    stats = analyze(model, (1, 3, 32, 32))
    assert model.training
    assert stats["total_macs"] == stats["conv2d_macs"] + stats["linear_macs"]
    assert stats["total_macs"] > 0 and stats["parameters"] > 0
    assert stats["output_shape"] == [1, 10]
