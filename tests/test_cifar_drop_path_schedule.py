"""Regression tests for per-block stochastic-depth scheduling."""

import pytest

from resnet_ablation.models.resnet_cifar import CIFARResNet


@pytest.mark.parametrize("depth", [20, 56, 110])
def test_drop_path_reaches_configured_rate(depth):
    rate = 0.36
    model = CIFARResNet(depth=depth, drop_path_rate=rate)
    blocks = [block for layer in (model.layer1, model.layer2, model.layer3)
              for block in layer]
    assert len(blocks) == (depth - 2) // 2
    probabilities = [block.drop_path.drop_prob for block in blocks]
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[-1] == pytest.approx(rate)
    assert probabilities == sorted(probabilities)
    assert len(set(probabilities)) == len(probabilities)


@pytest.mark.parametrize("depth", [2, 7, 21, 0])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError, match="6n\\+2"):
        CIFARResNet(depth=depth)


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_invalid_drop_path_rate_rejected(rate):
    with pytest.raises(ValueError, match="drop_path_rate"):
        CIFARResNet(depth=20, drop_path_rate=rate)


def test_invalid_width_rejected():
    with pytest.raises(ValueError, match="width_mult"):
        CIFARResNet(depth=20, width_mult=0)
