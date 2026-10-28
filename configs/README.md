# Configuration Guide · 配置导航

本目录同时包含**Factory Mode 受控研究配置**与**Legacy 训练示例**。它们的实验协议、数据隔离和输出格式不同；请不要混用分数或以旧脚本做论文准确率声明。

## Factory Mode — configs/paper/

**论文架构/控制配置**

| 文件 | 用途 |
| --- | --- |
| [cifar10_paper_control.yaml](paper/cifar10_paper_control.yaml) | CIFAR-10 ResNet-20 控制配置；配合明确的 paper_cifar_64k 调度 |
| [imagenet_control.yaml](paper/imagenet_control.yaml) | ImageNet ResNet-34 控制配置；当前仍为现代训练/增强近似 |

**实验矩阵（可 dry-run）**

| 文件 | 变量及注意事项 |
| --- | --- |
| [cifar10_depth_matrix.yaml](paper/cifar10_depth_matrix.yaml) | Plain/Residual、20/56/110 深度；6 组 × 3 seeds；64k steps/组 |
| [imagenet_degradation_matrix.yaml](paper/imagenet_degradation_matrix.yaml) | 18/34 × Plain/Residual，研究优化退化 |
| [imagenet_shortcut_matrix.yaml](paper/imagenet_shortcut_matrix.yaml) | ResNet-34 Shortcut A/B/C；参数量可能不同 |
| [imagenet_bottleneck_matrix.yaml](paper/imagenet_bottleneck_matrix.yaml) | ResNet-50 Bottleneck v1/v1.5；现代扩展 |
| [cifar10_attention_extension.yaml](paper/cifar10_attention_extension.yaml) | SE/ECA/DropPath；现代扩展 |
| [imagenet_stem_downsample_extension.yaml](paper/imagenet_stem_downsample_extension.yaml) | Deep Stem/avgproj；现代扩展 |

~~~bash
# 仅展开计划，无数据下载
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml

# 显式执行全部任务（长时间且可能下载数据）
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml --execute
~~~

**研究约束**：保持 train/val split SHA、优化器、训练 step、学习率和种子集合一致。Matrix 中标注的 allowed_factors 是允许变化的白名单，不是自动证明实验完全无混杂因素。原论文精确预处理与当前实现差异见 [论文映射](../docs/paper/01_PAPER_TO_CODE.md)。

## Legacy — configs/*.yaml

根目录的旧 YAML 与 scripts/train.py、scripts/eval.py 配套，保留路径避免破坏历史命令。代表文件：

- cifar10_resnet20.yaml、cifar10_resnet110_optionA.yaml：早期 CIFAR 基准；
- cifar10_resnet20_se.yaml、cifar10_resnet20_se_ecadrop_mix.yaml：早期 SE/混合增强示例；
- cifar100_resnet56_se_mix.yaml：CIFAR-100 扩展示例；
- imagenet_resnet50_baseline.yaml、imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml：早期 ImageNet 设计示例。

**限制**：Legacy 的 CIFAR test 被反复用于训练期间验证；断点恢复和 LR 日志仍有已知问题。根目录 run_all.sh 会连续执行多个耗时训练，**不是 CI/smoke 入口**。详见 [历史审查](../docs/REVIEW_2026-09-21.md)。

## 修改配置时

1. 为每个 paper experiment 保留可解释的 baseline 和单因素 treatment；新实验记录论文锚点、假设与验证标准。
2. 先运行 sweep dry-run，再在需要时执行训练；默认不将 output 或真实数据提交 Git。
3. 不要为展示性能随意复用官方 test，亦不要把不同配方/硬件上的误差直接合并。
4. 对新的配置枚举或模型架构提供对应的测试，确保非法组合 fail-fast。
