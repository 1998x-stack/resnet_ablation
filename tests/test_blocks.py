import torch
from torch import nn
from resnet_ablation.models.blocks import ShortcutA, SEModule, ECAModule, DropPath, _eca_kernel, BasicBlock


def test_shortcut_a_zero_pad():
    s = ShortcutA(64, 128, stride=1)
    x = torch.randn(2, 64, 8, 8)
    out = s(x)
    assert out.shape == (2, 128, 8, 8)
    assert torch.allclose(out[:, 64:], torch.zeros(2, 64, 8, 8))  # padded zeros


def test_shortcut_a_stride_downsample():
    s = ShortcutA(64, 64, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = s(x)
    assert out.shape == (2, 64, 4, 4)


def test_se_module_shape():
    m = SEModule(channels=64, ratio=0.25).eval()
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_eca_module_shape():
    m = ECAModule(channels=64).eval()
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_eca_kernel_odd_gte3():
    for c in range(16, 1024, 32):
        k = _eca_kernel(c)
        assert k >= 3 and k % 2 == 1


def test_droppath_identity_when_disabled():
    m = DropPath(drop_prob=0.0).eval()
    x = torch.randn(4, 8, 2, 2)
    with torch.no_grad():
        assert torch.equal(m(x), x)


def test_droppath_zero_keep_in_train():
    m = DropPath(drop_prob=1.0).train()  # keep=0
    x = torch.randn(4, 8, 2, 2)
    out = m(x)
    assert torch.allclose(out, torch.zeros_like(x))  # keep=0 -> /0 guarded, mask all zero


def test_basicblock_forward_shape():
    b = BasicBlock(in_planes=16, planes=16, stride=1, shortcut="A")
    x = torch.randn(2, 16, 16, 16)
    out = b(x)
    assert out.shape == x.shape