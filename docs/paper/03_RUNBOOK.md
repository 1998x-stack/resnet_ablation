# 03｜执行手册：Factory Mode、日志、可视化

## 安装和不下载数据的检查

    pip install -e .
    pip install pytest matplotlib
    python -m compileall -q resnet_ablation
    python -m pytest -q tests/test_paper_research.py

## 先预览实验矩阵（默认不会训练）

    python -m resnet_ablation.research.sweep --manifest configs/paper/cifar10_depth_matrix.yaml

执行全部种子组合须显式加参数：

    python -m resnet_ablation.research.sweep --manifest configs/paper/cifar10_depth_matrix.yaml --execute

## 原论文 CIFAR 迭代学习率的单组实验

    python -m resnet_ablation.research.runner run --config configs/paper/cifar10_paper_control.yaml --schedule paper_cifar_64k --steps 64000 --run-dir runs/paper/single/seed1

成本较低的现代调试版（10步，绝不是论文基准）：

    python -m resnet_ablation.research.runner run --config configs/paper/cifar10_paper_control.yaml --schedule cosine_step --steps 10 --run-dir runs/smoke/seed1 --device cpu

## 配对结果展示

    python -m resnet_ablation.research.report runs/paper/cifar10_depth/baseline/seed_1 runs/paper/cifar10_depth/plain20/seed_1 --allow-factor model.residual --out reports/plain_vs_residual

自动产生 runs.csv、summary.json，以及训练 loss、验证 error、学习率、梯度 L2 和完成 seed 的均值标准差图。TensorBoard 每个 run 目录保存 train/loss、train/acc1、train/images_per_sec、val/loss、val/acc1、val/error1、step/lr_used、grad/* 等标量。

显式最后一次测试，不能用于超参选择：

    python -m resnet_ablation.research.runner final-test --config configs/paper/cifar10_paper_control.yaml --run-dir runs/paper/single/seed1

## 输出契约

每个 run 目录下：
- manifest.json：resolved config、git SHA、设备与 torch 版本、参数量、split SHA 和归一化免责声明；
- metrics.jsonl：每 log_interval 步的 train_step、每轮的 train_epoch/val_epoch；
- tensorboard/ 和 train.log：图形标量与 Loguru 可读日志；
- last.pt / best.pt：checkpoint 含模型、优化器、AMP scaler、Python/NumPy/Torch/CUDA RNG、next_epoch、global_step；
- status.json：真实完成步数及完成状态；
- final_test.json：仅在单独执行 final-test 后存在。

checkpoint 以临时文件加 os.replace 原子更新；用 --resume 从**可信本项目生成文件**恢复，不要导入未知不可信 checkpoint。仅保证 epoch 边界恢复；中途进程终止时，回退至上个完整 epoch，并重新训练未完成的一轮。

## 范围与尚未完成的研究

- 当前提供 original-paper architecture + CIFAR step schedule + modern preprocessing 的控制实验，不是逐项完全复刻论文。
- ImageNet 600k-step plateau scheduler、10-crop/multiscale、per-pixel mean、结果显著性检验及足量硬件评测未完成。
- ResNet-1202 极其耗资源，先运行较小深度、进行容量和资源预估，再独立安排。

## 优化退化的关键可视化

新增 training_error.png（每轮训练错误率）、validation_error.png（验证错误率）。两者均以 optimizer completed steps 为横坐标，并保留训练样本与验证样本定义。只有训练误差在深度增加时明显上升，才能对原论文所讨论的 optimization degradation 提供直接证据；仅凭验证误差不能断言退化。
