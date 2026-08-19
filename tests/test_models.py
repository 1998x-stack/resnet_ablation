import torch
import pytest
from resnet_ablation.config import Config
from resnet_ablation.models.factory import build_model


@pytest.mark.parametrize("arch,num_classes", [
    ("resnet20_cifar", 10), ("resnet56_cifar", 100),
    ("resnet110_cifar", 10), ("resnet18", 1000), ("resnet34", 1000),
    ("resnet50", 1000), ("resnet101", 1000), ("resnet152", 1000),
])
def test_forward_shape(arch, num_classes):
    cfg = Config()
    cfg.model.arch = arch
    cfg.model.num_classes = num_classes
    if "_cifar" not in arch:
        cfg.model.stem = "imagenet_standard"
    model = build_model(cfg).eval()
    size = 32 if "_cifar" in arch else 224
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes), f"{arch} -> {out.shape}"


def test_unknown_arch_raises():
    cfg = Config()
    cfg.model.arch = "resnet999"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_param_count_ranges():
    from resnet_ablation.utils import count_params
    cfg = Config()
    cfg.model.arch = "resnet20_cifar"
    n = count_params(build_model(cfg))
    assert 0.1e6 < n < 1.0e6  # ResNet-20 CIFAR ~273k