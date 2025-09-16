from __future__ import annotations
from typing import Callable, Optional
import torch
from torch import nn
import torch.nn.functional as F


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
            pad = (0, 0, 0, 0, 0, self.pad_c)  # (W_left, W_right, H_bottom, H_top, C_front, C_back)
            x = F.pad(x, pad)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock：两个 3x3 卷积。"""
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "A", norm: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        self.shortcut_type = shortcut
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
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck：1x1 → 3x3 → 1x1。ImageNet 风格更节省计算。"""
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 shortcut: str = "B", norm: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm(planes * self.expansion)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_planes != planes * self.expansion:
            # 论文指出：Bottleneck 使用 identity shortcut 更高效，但在降采样/升维处常用 Option B（1x1 conv projection）:contentReference[oaicite:6]{index=6}
            if shortcut in ("B", "C", "A"):
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
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out