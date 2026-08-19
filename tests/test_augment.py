import torch
from resnet_ablation.augment import mixup, cutmix, rand_bbox, one_hot


def _batch(n=16, c=10, hw=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3, hw, hw)
    y = torch.randint(0, c, (n,))
    return x, y


def test_mixup_shape_and_softness():
    x, y = _batch()
    mixed, lab = mixup(x, y, num_classes=10, alpha=1.0)
    assert mixed.shape == x.shape
    assert lab.shape == (x.size(0), 10)
    assert torch.allclose(lab.sum(dim=1), torch.ones(x.size(0)))  # rows sum to 1


def test_cutmix_shape_and_softness():
    x, y = _batch()
    mixed, lab = cutmix(x, y, num_classes=10, alpha=1.0)
    assert mixed.shape == x.shape
    assert lab.shape == (x.size(0), 10)
    assert torch.allclose(lab.sum(dim=1), torch.ones(16))


def test_mixup_alpha_zero_raises():
    import pytest
    x, y = _batch()
    with pytest.raises(ValueError):
        mixup(x, y, num_classes=10, alpha=0.0)


def test_cutmix_alpha_zero_raises():
    import pytest
    x, y = _batch()
    with pytest.raises(ValueError):
        cutmix(x, y, num_classes=10, alpha=0.0)


def test_rand_bbox_bounds():
    for _ in range(50):
        x1, y1, x2, y2 = rand_bbox(32, 32, 0.5)
        assert 0 <= x1 < x2 <= 32
        assert 0 <= y1 < y2 <= 32


def test_one_hot_smoothing():
    labs = torch.tensor([0, 3])
    y = one_hot(labs, 10, smoothing=0.1)
    assert y.shape == (2, 10)
    assert torch.allclose(y.sum(dim=1), torch.ones(2))
    assert torch.allclose(y[0, 0], torch.full((), 0.91))  # (1-0.1) + 0.1/10