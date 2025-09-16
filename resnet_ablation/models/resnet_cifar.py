from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock


class CIFARResNet(nn.Module):
    """CIFAR 风格 ResNet（6n+2，总通道 16/32/64，默认 Shortcut=Option A）。
    依据论文：输入 32×32，特征图依次 {32,16,8}，每个尺度堆叠 2n 个 BasicBlock，最后全局平均池化+全连接 :contentReference[oaicite:7]{index=7}。
    """
    def __init__(self, depth: int, num_classes: int = 10, shortcut: str = "A", width_mult: float = 1.0):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        base = int(16 * width_mult)
        self.in_planes = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, base,  n, stride=1, shortcut=shortcut)
        self.layer2 = self._make_layer(BasicBlock, base*2, n, stride=2, shortcut=shortcut)
        self.layer3 = self._make_layer(BasicBlock, base*4, n, stride=2, shortcut=shortcut)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base*4 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int, shortcut: str):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def resnet_cifar_6n2(depth: int, num_classes: int = 10, shortcut: str = "A", width_mult: float = 1.0) -> CIFARResNet:
    return CIFARResNet(depth=depth, num_classes=num_classes, shortcut=shortcut, width_mult=width_mult)