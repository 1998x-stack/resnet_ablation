# ResNet Ablation (Pluggable) / 可插拔 ResNet 消融库

> A pluggable, from-scratch PyTorch ResNet library for controlled ablation studies.
> 一个从零实现、可插拔配置的 PyTorch ResNet 消融实验库。

---

## What it is（简介）

A from-scratch PyTorch 3.9 ResNet implementation designed for *isolating one variable at a time* in ablation studies. Shortcut type, stem, downsampling, attention, stochastic depth, and data augmentation are all independently togglable through YAML config, so you can measure the contribution of each technique without touching model code. The library ships a strict, validated configuration system, per-epoch structured logging (`results.csv`), a resolved-config reproducibility dump (`config.json`), and a CPU-runnable pytest suite so the whole pipeline can be verified before launching long runs.

一个从零实现、面向消融研究的 PyTorch ResNet 库：短接类型（shortcut）、主干（stem）、下采样、注意力、随机深度与数据增强均可在 YAML 配置中独立开关，从而在不改动模型代码的前提下隔离并量化每一处改进的贡献。库内置严格校验的配置体系、逐 epoch 结构化日志（`results.csv`）、解析后配置转储（`config.json`）以及可在 CPU 上运行的 pytest 测试套件，保证长训练前的全流程可验证性。

## Feature map / 功能总览

Capability matrix — every row is independently toggleable via YAML:

- **Shortcuts**（短连接）A / B / C — Option A identity / B projection / C padding variants.
- **Stems**（主干）`cifar` / `imagenet_standard` / `imagenet_deep` — standard 7×7 or 3×3×3 DeepStem.
- **Downsampling**（下采样）`proj` (1×1 conv) / `avgproj` (ResNet-D avg-pool + 1×1).
- **Attention**（注意力）`none` / `se` / `eca` (+ `se_ratio`).
- **DropPath**（随机深度）`drop_path_rate` — Stochastic Depth, linearly scheduled across blocks.
- **Mixup / CutMix / Label Smoothing**（数据增强与标签平滑）— toggled via `mixup_alpha` / `cutmix_alpha` / `label_smoothing`（二者同时>0 时随机二选一；有 mix/cut 时采用 SoftTargetCE）.
- **AMP**（自动混合精度）+ gradient clipping.
- **Resume**（断点续训）from an existing checkpoint.
- **`results.csv`** per-epoch logging（逐 epoch 结构化日志）.
- **`config.json`** resolved-config dump including runtime environment（含运行环境的配置转储，保障可复现）.
- **Linear warmup LR schedule**（线性预热学习率）with **per-epoch stepping**（逐 epoch 步进）.
- **CPU test suite**（CPU pytest 测试套件）for the whole pipeline.

## Project structure / 项目结构（module map）

| Path | Responsibility | 职责 |
|------|----------------|------|
| `resnet_ablation/config.py` | strict `Config` dataclasses + YAML loading / enum validation | 配置校验 |
| `resnet_ablation/models/blocks.py` | Basic/Bottleneck blocks, Shortcut A/B/C, attention, DropPath | 网络基础块 |
| `resnet_ablation/models/resnet_cifar.py` | CIFAR-style ResNet (6n+2), Option A shortcuts | CIFAR 网络 |
| `resnet_ablation/models/resnet_imagenet.py` | ImageNet-style ResNet18–152, Option B | ImageNet 网络 |
| `resnet_ablation/models/factory.py` | `build_model(cfg)` — pluggable model construction | 模型工厂/构建 |
| `resnet_ablation/engine/trainer.py` | trainer: AMP / checkpoint / `results.csv` | 训练器 |
| `resnet_ablation/engine/evaluator.py` | validation / evaluation loop | 评估器 |
| `resnet_ablation/data.py` | dataset & dataloader construction, transforms | 数据加载 |
| `resnet_ablation/augment.py` | Mixup / CutMix / `rand_bbox` | 数据增强 |
| `resnet_ablation/scheduler.py` | linear warmup + cosine / multistep LR | 学习率调度 |
| `resnet_ablation/metrics.py` | Top-k accuracy metrics | 指标计算 |
| `resnet_ablation/losses.py` | loss functions (incl. SoftTargetCE) | 损失函数 |
| `resnet_ablation/logger.py` | loguru setup | 日志 |
| `resnet_ablation/utils.py` | seed / device / param counting helpers | 工具函数 |
| `scripts/train.py` | CLI training entry point | 训练入口 |
| `scripts/eval.py` | CLI evaluation entry point | 评估入口 |
| `tests/` | CPU pytest suite (config / model / scheduler / metrics / e2e) | 测试套件 |
| `configs/` | 10 ready-made ablation YAML configs | 消融实验配置 |

## Quickstart / 快速开始

Install the package (editable):

```bash
pip install -e .
```

Train a single run (CIFAR-10 ResNet-20 baseline):

```bash
python scripts/train.py --config configs/cifar10_resnet20.yaml
```

Evaluate a trained checkpoint:

```bash
python scripts/eval.py --config configs/cifar10_resnet20.yaml --ckpt resnet_ablation/checkpoints/cifar10_resnet20/last.pt
```

Run multiple seeds for variance analysis (`--out-suffix`, `--seed` override config):

```bash
for s in 1 2 3; do
  python scripts/train.py --config configs/cifar10_resnet20.yaml --seed $s --out-suffix "s$s"
done
```

Outputs produced per run (under `train.out_dir`):
- `last.pt` / `best.pt` — checkpoints（断点）
- `results.csv` — per-epoch structured log（逐 epoch 日志）
- `config.json` — resolved config + runtime env（解析后配置与运行环境）
- TensorBoard logs under `train.tb_dir`（TensorBoard 日志）

## Config reference / 配置项

Configuration is grouped into `model` / `optim` / `train` / `data`, validated against the whitelist in `resnet_ablation/config.py`. Values marked *enum* are strictly checked.

### model / 模型

| Field | Default | Allowed values |
|-------|---------|----------------|
| `arch` | `resnet20_cifar` | `resnet20_cifar` / `resnet56_cifar` / `resnet110_cifar` / `resnet18` / `resnet34` / `resnet50` / `resnet101` / `resnet152` |
| `num_classes` | `10` | any int |
| `shortcut` | `A` | `A` / `B` / `C` |
| `stem` | `cifar` | `cifar` / `imagenet_standard` / `imagenet_deep` |
| `downsample` | `proj` | `proj` / `avgproj` |
| `width_mult` | `1.0` | any float |
| `drop_path_rate` | `0.0` | any float |
| `attention` | `none` | `none` / `se` / `eca` |
| `se_ratio` | `0.25` | any float |

### optim / 优化器与调度

| Field | Default | Allowed values |
|-------|---------|----------------|
| `name` | `sgd` | any (optimizer name) |
| `lr` | `0.1` | any float |
| `momentum` | `0.9` | any float |
| `weight_decay` | `1e-4` | any float |
| `nesterov` | `true` | `true` / `false` |
| `warmup_epochs` | `5` | any int (0 disables warmup) |
| `sched` | `cosine` | `cosine` / `multistep` |
| `milestones` | `[100, 150]` | any int list |
| `gamma` | `0.1` | any float |

### train / 训练

| Field | Default | Allowed values |
|-------|---------|----------------|
| `epochs` | `200` | any int |
| `batch_size` | `128` | any int |
| `num_workers` | `4` | any int |
| `amp` | `true` | `true` / `false` |
| `clip_grad_norm` | `1.0` | any float or `null` |
| `log_interval` | `50` | any int |
| `val_interval` | `1` | any int |
| `out_dir` | `resnet_ablation/checkpoints` | any path |
| `resume` | `null` | checkpoint path or `null` |
| `seed` | `42` | any int |
| `tb_dir` | `runs` | any path |
| `label_smoothing` | `0.0` | any float |

### data / 数据

| Field | Default | Allowed values |
|-------|---------|----------------|
| `name` | `cifar10` | `cifar10` / `cifar100` / `imagenet` |
| `root` | `./data` | any path |
| `aug` | `standard` | `standard` / `strong` |
| `mixup_alpha` | `0.0` | any float (>0 enables Mixup) |
| `cutmix_alpha` | `0.0` | any float (>0 enables CutMix) |

## Ablation experiments / 消融实验表

All 10 shipped configs — each isolates a specific technique:

| configs/*.yaml | What it ablated / 消融内容 |
|----------------|----------------------------|
| `configs/cifar10_resnet20.yaml` | CIFAR-10 ResNet-20 baseline, Shortcut A — 基线 |
| `configs/cifar10_resnet20_se.yaml` | CIFAR-10 ResNet-20 + SE attention — SE 注意力 |
| `configs/cifar10_resnet20_se_ecadrop_mix.yaml` | CIFAR-10 + ECA + DropPath + Mixup (strong aug) — ECA+随机深度+Mixup |
| `configs/cifar10_resnet56_ablate_optionB.yaml` | Shortcut ablation A→B on ResNet-56 — 短连接 A→B 消融 |
| `configs/cifar10_resnet110_optionA.yaml` | Deep CIFAR baseline (ResNet-110) — 深度基线 |
| `configs/cifar100_resnet56_se_mix.yaml` | CIFAR-100 ResNet-56 + SE + Mixup — CIFAR-100 增强组合 |
| `configs/imagenet_resnet50_baseline.yaml` | ImageNet ResNet-50 baseline (standard stem, no attn) — ImageNet 基线 |
| `configs/imagenet_resnet50.yaml` | ImageNet ResNet-50 (minimal override of baseline) — ImageNet R50 |
| `configs/imagenet_resnet50_deepstem_resnetd_se.yaml` | R50 DeepStem + ResNet-D + SE — DeepStem+ResNet-D+SE |
| `configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml` | R101 DeepStem + ResNet-D + ECA + DropPath — R101 深度组合 |

## Reproducibility & metrics / 可复现性与指标

**`results.csv`** — one row per validation epoch. Columns:
`timestamp, dataset, arch, seed, epoch, schedule, attn, shortcut, lr, train_loss,
train_acc1, val_loss, val_acc1, mixup_alpha, cutmix_alpha, label_smoothing,
checkpoint`. This makes per-run and cross-run comparisons easy and gives you the
exact hyper-parameter snapshot that produced each number.

**`config.json`** — written to `out_dir` at launch, containing the *resolved*
configuration (after any `--seed` / `--out-suffix` overrides) plus a `runtime`
block (`device`, `timestamp`, `seed`). This is the reproducibility anchor: any
row in `results.csv` can be traced back to the exact config that generated it.

逐 epoch 结构化日志：`results.csv` 每个验证 epoch 追加一行，记录时间戳、数据集、架构、种子、epoch、调度器、注意力、短连接、学习率、训练/验证损失与 acc1，以及 mixup/cutmix/标签平滑等增强参数与检查点文件名。`config.json` 在训练开始时转储解析后的完整配置（含命令行覆盖）与运行环境（设备、时间戳、种子），确保任何一条指标都能追溯到其精确配置。

## Dev workflow / 开发

Run the CPU test suite:

```bash
python3 -m pytest -q
```

Byte-compile check across packages:

```bash
python3 -m compileall resnet_ablation scripts tests
```

Optional linting (may not be installed):

```bash
ruff check .      # optional; run `pip install ruff` if missing
```

## Design doc / 设计文档

See **`ChatGPT-ResNet 项目构建.md`** (in the repo root) for the original
architecture walkthrough and design rationale (in Chinese) —
从零构建 ResNet 的架构讲解与设计思路（中文设计文档）。