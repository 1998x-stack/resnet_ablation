# ResNet Ablation (Pluggable)

- CIFAR & ImageNet 风格 ResNet 模型，Basic/Bottleneck/Shortcut A/B/C 可切换
- 训练引擎：混合精度、断点续训、TensorBoard、Loguru
- 参考：ResNet 论文（CIFAR 架构 6n+2；ImageNet Bottleneck/Option B）【见源码注释中的引用】

Quickstart:

```bash
pip install -e .
python scripts/train.py --config configs/cifar10_resnet20.yaml
```# resnet_ablation
