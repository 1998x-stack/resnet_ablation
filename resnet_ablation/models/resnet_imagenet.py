from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock, Bottleneck


class ImageNetResNet(nn.Module):
    """ImageNet 风格 ResNet：支持 BasicBlock(18/34) 与 Bottleneck(50/101/152)。"""
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 1000, shortcut: str = "B"):
        super().__init__()
        self.in_planes = 64

        # deep stem 可选：这里采用标准 stem（7x7 stride 2 + 3x3 maxpool）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, shortcut=shortcut)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, shortcut=shortcut)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, shortcut=shortcut)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, shortcut=shortcut)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int, shortcut: str):
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def _resnet(block: Type[nn.Module], layers: List[int], num_classes: int, shortcut: str) -> ImageNetResNet:
    return ImageNetResNet(block, layers, num_classes=num_classes, shortcut=shortcut)

# 工厂函数（与论文表层数对应）
def resnet18(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, shortcut)

def resnet34(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, shortcut)

def resnet50(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, shortcut)

def resnet101(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, shortcut)

def resnet152(num_classes: int = 1000, shortcut: str = "B") -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes, shortcut)