from __future__ import annotations
from typing import Callable, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------
# Stochastic Depth (DropPath)
# ---------------------------
class DropPath(nn.Module):
    """按样本随机丢弃残差分支（Stochastic Depth）"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # (N, 1, 1, 1) broadcast
        mask = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x / keep * mask


# -------------
# Attention: SE
# -------------
class SEModule(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w


# -------------
# Attention: ECA
# -------------
def _eca_kernel(channels: int, gamma: int = 2, b: int = 1) -> int:
    k = int(abs((math.log2(channels) / gamma) + b))
    if k % 2 == 0:
        k += 1
    return max(3, k)

class ECAModule(nn.Module):
    """Efficient Channel Attention（无需降维，1D conv）"""
    def __init__(self, channels: int, k_size: Optional[int] = None):
        super().__init__()
        k = _eca_kernel(channels) if k_size is None else k_size
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)                               # (N, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))# (N, 1, C)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


def build_attention(kind: str, channels: int, se_ratio: float = 0.25) -> Optional[nn.Module]:
    kind = (kind or "none").lower()
    if kind == "none":
        return None
    if kind == "se":
        return SEModule(channels, ratio=se_ratio)
    if kind == "eca":
        return ECAModule(channels)
    raise ValueError(f"Unknown attention kind: {kind}")


# -------------
# Shortcut A/B
# -------------
class ShortcutA(nn.Module):
    """Option A: 恒等+零填充，用于 CIFAR：通道不匹配时右侧零填充。"""
    def __init__(self, in_planes: int, out_planes: int, stride: int):
        super().__init__()
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pad_c = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, ::self.stride, ::self.stride]
        if self.pad_c > 0:
            pad = (0, 0, 0, 0, 0, self.pad_c)
            x = F.pad(x, pad)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ----------------
# Basic / Bottleneck
# ----------------
class BasicBlock(nn.Module):
    """ResNet BasicBlock：两个 3×3 卷积，支持注意力与 DropPath。"""
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "A",
                 norm: Callable[..., nn.Module] = nn.BatchNorm2d,
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path: float = 0.0):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)
        self.attn = build_attention(attention, planes, se_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut == "A":
                self.downsample = ShortcutA(in_planes, planes * self.expansion, stride)
            elif shortcut in ("B", "C"):
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride),
                    norm(planes * self.expansion)
                )
            else:
                raise ValueError(f"Unknown shortcut {shortcut}")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck：1×1 → 3×3 → 1×1，支持注意力、DropPath。"""
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B",
                 norm: Callable[..., nn.Module] = nn.BatchNorm2d,
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path: float = 0.0,
                 downsample_mode: str = "proj"):  # proj / avgproj (ResNet-D)
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm(planes)

        # 残差主分支下采样：仍在 3x3 上做 stride（ResNet-D 的 avg 在快捷分支处理）
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)

        self.attn = build_attention(attention, planes * self.expansion, se_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            if downsample_mode == "avgproj":
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False),
                    conv1x1(in_planes, planes * self.expansion, stride=1),
                    norm(planes * self.expansion)
                )
            else:
                self.downsample = nn.Sequential(
                    conv1x1(in_planes, planes * self.expansion, stride=stride),
                    norm(planes * self.expansion)
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.attn is not None:
            out = self.attn(out)
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out