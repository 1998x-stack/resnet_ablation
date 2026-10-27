# 01｜ResNet 原论文—代码证据映射

原始资料：[He et al. Deep Residual Learning for Image Recognition, arXiv:1512.03385](https://ar5iv.labs.arxiv.org/html/1512.03385)，[CVPR 2016 官方全文](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)，[PyTorch 官方 ResNet 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)。

核心假设不是网络越深越好，而是：**残差连接降低深层网络的优化难度；可直接用固定深度和训练条件的 Plain/ResNet 配对实验检验。**

| 原文位置 | 原始实验/公式 | 本分支实现 | 客观验收 |
| --- | --- | --- | --- |
| §3.1–3.2 / Eq.(1) | y=F(x)+x，随后 ReLU | models/blocks.py 的 residual | Plain 与 ResNet 控制深度、宽度、Option A 参数数一致 |
| §3.2 / Eq.(2) | y=F(x)+Ws(x) | ShortcutA；1×1+BN | stride 2、channels 翻倍时逐元素可加 |
| §3.3、§4.1 Table 3 | Shortcut A 零填充；B 仅变化维度投影；C 全部投影 | BasicBlock 和 Bottleneck | B 的 equal-shape 块无投影，C 有投影 |
| Table 1 | 18/34 的 BasicBlock；50/101/152 的 1×1,3×3,1×1 Bottleneck | models/resnet_imagenet.py | 堆叠数 [2,2,2,2]、[3,4,6,3]、[3,4,6,3]、[3,4,23,3]、[3,8,36,3] |
| §3.4 | ImageNet: SGD 0.1/0.9/1e-4；BN；无 dropout | configs/paper/imagenet_control.yaml | 日志公开实际超参和环境 |
| §4.2，Fig.1/6 | CIFAR 层数 6n+2、每层 16/32/64 通道 | models/resnet_cifar.py | 20/32/44/56/110/1202 计数和空间尺寸 |
| §4.2，Table 6 | 64k steps，32k/48k 降 LR，110 前约400步 0.01 | research/factories.py: StepSchedule | 在 399/400/31999/32000/47999/48000 step 查验 |
| §4.2 | 随机裁剪 pad 4、水平翻转；45k/5k train/val | research/factories.py: split_indices | 分割无重叠，split SHA 稳定 |
| §4.1/4.2 | 训练错误、验证错误和最终 test 不同 | research/runner.py、report.py | train vs validation 分开，final-test 要单独执行 |

## 论文结果与本仓库数据严禁混用

原论文 ImageNet Table 3 的 ResNet-34 A/B/C 10-crop top-1 错误率分别为 25.03%、24.52%、24.19%；CIFAR Table 6 ResNet-20/56/110 的 test error 分别为 8.75%、6.97%、6.43%（110 是报告最佳值，另有重复试验统计）。这些**不是本仓库跑出的结果**，不可作为本项目 benchmark。

## 忠实度和差异登记

1. 当前 CIFAR 预处理为 per-channel mean/std，原文为 per-pixel mean subtraction；学习率/步数可以匹配，但不能声称全流程 bitwise 或 exact-paper。
2. 当前 ImageNet RandomResizedCrop + CenterCrop 与固定 epoch milestones 是现代研究配方，尚未实现原文的随机短边 [256,480]、颜色增强、最多 600k iterations、plateau 调度和 10-crop/多尺度推理。
3. Bottleneck-v1 下采样在第一个 1×1，v1.5 在 3×3。后者来源于后续工程实践，参见 [torchvision 文档](https://docs.pytorch.org/vision/stable/models/resnet.html)。
4. ImageNet ResNet-34 ABC 对照与论文 Table 3 对应；CIFAR 上的 B/C 是新实验，不要伪称原论文本来的 CIFAR 设置。
5. Plain/Residual Option A 的参数量可以一致；B/C 引入 Projection+BN，参数量必然不同，需报告参数数量和速度。
6. 110 层的 400-step warmup 是额外干预因素。比较 110 Plain/Residual 时应统一 warmup 才能纯粹估计残差效果；原文配方比较则单独注明条件。

## 源码导航

- models/blocks.py：BasicBlock、Bottleneck、A/B/C、v1/v1.5
- models/resnet_cifar.py 和 models/resnet_imagenet.py：结构深度、宽度、特征图
- models/factory.py 与 research/factories.py：模型/数据/优化器/损失/调度注册工厂
- research/runner.py：训练状态、真实训练 step、JSONL、TensorBoard、checkpoint
- research/sweep.py：多因素/多种子实验生成
- research/report.py：比较约束、真实图表、统计
