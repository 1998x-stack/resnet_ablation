from __future__ import annotations

from typing import List, Type

from torch import nn

from .blocks import BasicBlock


class CIFARResNet(nn.Module):
    """CIFAR ResNet with depth 6n+2, optional channel attention and DropPath."""

    def __init__(self, depth: int, num_classes: int = 10, shortcut: str = "A",
                 width_mult: float = 1.0, attention: str = "none", se_ratio: float = 0.25,
                 drop_path_rate: float = 0.0):
        super().__init__()
        if depth < 8 or (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must be 6n+2 with n >= 1")
        if width_mult <= 0 or int(16 * width_mult) < 1:
            raise ValueError("width_mult must produce at least one base channel")
        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError("drop_path_rate must be in [0, 1]")

        n = (depth - 2) // 6
        base = int(16 * width_mult)
        self.in_planes = base

        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.relu = nn.ReLU(inplace=True)

        # Each of the three stages has n BasicBlocks. Using 6n here counts
        # convolutional layers instead of blocks and silently halves the
        # maximum applied DropPath probability.
        total_blocks = 3 * n
        dpr = [drop_path_rate * i / max(1, total_blocks - 1)
               for i in range(total_blocks)]
        dp_iter = iter(dpr)

        self.layer1 = self._make_layer(BasicBlock, base, n, stride=1, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)
        self.layer2 = self._make_layer(BasicBlock, base * 2, n, stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)
        self.layer3 = self._make_layer(BasicBlock, base * 4, n, stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 4 * BasicBlock.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int,
                    shortcut: str, attention: str, se_ratio: float, dp_iter):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut,
                            attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter)))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut,
                                attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def resnet_cifar_6n2(depth: int, num_classes: int = 10, shortcut: str = "A",
                      width_mult: float = 1.0, attention: str = "none",
                      se_ratio: float = 0.25, drop_path_rate: float = 0.0) -> CIFARResNet:
    return CIFARResNet(depth=depth, num_classes=num_classes, shortcut=shortcut,
                       width_mult=width_mult, attention=attention, se_ratio=se_ratio,
                       drop_path_rate=drop_path_rate)
