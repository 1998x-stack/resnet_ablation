# 0) 安装（首次）
pip install -e .

# 1) CIFAR-10 Baseline: ResNet-20, Option A
python scripts/train.py --config configs/cifar10_resnet20.yaml

# 2) CIFAR-10 消融：Shortcut 改为 Option B（ResNet-56）
python scripts/train.py --config configs/cifar10_resnet56_ablate_optionB.yaml

# 3) CIFAR-10 + SE（ResNet-20）
python scripts/train.py --config configs/cifar10_resnet20_se.yaml

# 4) CIFAR-10 + ECA + DropPath + Mixup（ResNet-20，一站式增强）
python scripts/train.py --config configs/cifar10_resnet20_se_ecadrop_mix.yaml

# 5) CIFAR-10 深层：ResNet-110（Option A baseline）
python scripts/train.py --config configs/cifar10_resnet110_optionA.yaml

# 6) CIFAR-100 + SE + Mixup（ResNet-56）
python scripts/train.py --config configs/cifar100_resnet56_se_mix.yaml

# 7) ImageNet R50 baseline（标准 stem + 标准 downsample）
#    注意：请把 configs 中的 data.root 改为你的 ImageNet 路径（包含 train/ 与 val/）
python scripts/train.py --config configs/imagenet_resnet50_baseline.yaml

# 8) ImageNet R50 DeepStem + ResNet-D + SE
python scripts/train.py --config configs/imagenet_resnet50_deepstem_resnetd_se.yaml

# 9) ImageNet R101 DeepStem + ResNet-D + ECA + DropPath
python scripts/train.py --config configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml

# ===== Multi-seed 复现示例（以 CIFAR-10 ResNet-20 baseline 为例） =====
for s in 1 2 3; do
  python scripts/train.py --config configs/cifar10_resnet20.yaml --seed $s --out-suffix "s${s}"
done

# ===== 评估示例（用 last.pt 或 best.pt）=====
python scripts/eval.py --config configs/cifar10_resnet20.yaml --ckpt resnet_ablation/checkpoints/cifar10_resnet20/last.pt