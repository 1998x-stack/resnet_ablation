from __future__ import annotations
from typing import Type, List
from torch import nn
from .blocks import BasicBlock, Bottleneck


def _make_stem(kind: str) -> nn.Sequential:
    """ImageNet stem: standard(7x7, s=2) or deep(3×3×3)."""
    kind = kind.lower()
    if kind == "imagenet_deep":
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )


class ImageNetResNet(nn.Module):
    """ImageNet 风格 ResNet：支持 BasicBlock(18/34) 与 Bottleneck(50/101/152)，可选 DeepStem/ResNet-D/注意力/DropPath。"""
    def __init__(self, block: Type[nn.Module], layers: List[int],
                 num_classes: int = 1000,
                 shortcut: str = "B",
                 stem: str = "imagenet_standard",
                 downsample: str = "proj",         # proj / avgproj (ResNet-D)
                 attention: str = "none",
                 se_ratio: float = 0.25,
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.in_planes = 64

        self.stem = _make_stem(stem)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 线性分配 DropPath 到每个 block
        total_blocks = sum(layers)
        dpr = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        dp_iter = iter(dpr)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, shortcut=shortcut,
                                       attention=attention, se_ratio=se_ratio, dp_iter=dp_iter, downsample=downsample)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int,
                    shortcut: str, attention: str, se_ratio: float, dp_iter, downsample: str):
        layers: List[nn.Module] = []
        # 第一个 block 可能下采样
        layers.append(block(self.in_planes, planes, stride=stride, shortcut=shortcut,
                            attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter),
                            downsample_mode=downsample if block is Bottleneck else "proj"))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, shortcut=shortcut,
                                attention=attention, se_ratio=se_ratio, drop_path=next(dp_iter),
                                downsample_mode=downsample if block is Bottleneck else "proj"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def _resnet(block: Type[nn.Module], layers: List[int], **kwargs) -> ImageNetResNet:
    return ImageNetResNet(block, layers, **kwargs)

def resnet18(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet34(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet50(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet101(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, shortcut=shortcut, **extras)

def resnet152(num_classes: int = 1000, shortcut: str = "B", **extras) -> ImageNetResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, shortcut=shortcut, **extras)