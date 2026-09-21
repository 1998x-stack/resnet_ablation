# ResNet Ablation / 可插拔 ResNet 消融实验库

A from-scratch PyTorch implementation of CIFAR-style and ImageNet-style ResNet with configurable shortcut, stem, downsampling, SE/ECA attention, stochastic depth and augmentation. Designed for controlled experiments; **the shipped configurations and training pipeline are not a verified reproduction of the original papers**.

一个从零实现、面向可控消融实验的 PyTorch ResNet 项目。支持 CIFAR/ImageNet 网络、注意力、随机深度和可配置训练流程。请注意：仓库中的示例配置及训练流程**尚未经过论文级复现验证**。

> **Current review / 当前审查状态 (2026-09-21):** [Paper-to-code audit, known issues, priorities and acceptance gates](docs/REVIEW_2026-09-21.md). The accompanying PR corrects the CIFAR DropPath block count and adds regression tests. The other listed issues remain open; full tests and benchmark training have not been executed as part of this review.

## Capabilities and limitations / 功能与限制

| Component | Implemented behavior | Important qualification |
| --- | --- | --- |
| CIFAR ResNet | ResNet-20/56/110 with `6n+2` weighted layers; `3n` BasicBlocks | Fixed DropPath scheduling counts blocks, not individual convolutional layers. |
| ImageNet ResNet | 18/34 BasicBlock and 50/101/152 Bottleneck variants | Bottleneck downsamples in its 3×3 convolution (v1.5-style layout), not an exact implementation of every original-paper detail. |
| Shortcut | A: strided identity with zero padding when needed; B: projection when dimensions change | **C currently aliases B in `BasicBlock`, not the original paper's projection on every shortcut.** Do not treat B vs C as an independent ablation. |
| Stem / downsampling | Standard 7×7 ImageNet stem, optional three-3×3 deep stem; `proj`/`avgproj` | `avgproj` is implemented for the Bottleneck shortcut path, not the BasicBlock path. |
| Attention | None, SE, ECA; configurable SE ratio | Attention changes model capacity. Record parameters and computation alongside accuracy. |
| DropPath | Linearly distributed per residual block; final CIFAR block reaches configured maximum after PR fix | `drop_path_rate` is not the same as standard Dropout. |
| Augmentation | Standard/strong transforms, Mixup, CutMix, label smoothing | With Mixup/CutMix, current `train_acc1` is computed against **original hard labels** and is not conventional training accuracy. |
| Training | SGD, per-epoch warmup + cosine/multistep scheduling, AMP, clipping, TensorBoard, checkpoints | Resume can replay an epoch; latest checkpoint is written only on validation epochs; saved RNG state is incomplete. See review before using for controlled interrupted runs. |
| Metrics / metadata | `results.csv` on validation epochs; resolved `config.json` | The logged epoch LR is currently read **after** the scheduler step; `config.json` does not by itself guarantee reproducibility. |
| Validation | CIFAR training loader plus CIFAR **test** loader returned as `val_loader` | Current training evaluates repeatedly on the test set. Create a held-out validation split before tuning hyperparameters. |
| Configuration | YAML key/enum checking and model-level CIFAR depth/width/DropPath checks | Not all numeric ranges, class counts, model/data combinations or optimizer choices are validated; training entry point currently constructs SGD. |

## Repository map / 项目结构

- `resnet_ablation/models/`: BasicBlock/Bottleneck, SE/ECA, CIFAR and ImageNet architectures, model factory.
- `resnet_ablation/config.py`, `configs/`: configuration dataclasses, loader and example YAML files.
- `resnet_ablation/data.py`, `augment.py`: dataset transforms, Mixup and CutMix.
- `resnet_ablation/engine/`, `scheduler.py`: training, evaluation, logging, checkpoints and LR policies.
- `scripts/train.py`, `scripts/eval.py`: command-line entry points.
- `tests/`: unit tests and a small synthetic-data end-to-end test; added DropPath regression tests live in `tests/test_cifar_drop_path_schedule.py`.
- `docs/REVIEW_2026-09-21.md`: paper-to-implementation comparison, risk log, experimental controls and verification plan.

## Quick start / 快速开始

Run from the repository root with a compatible Python/PyTorch environment. The CIFAR training command **downloads the dataset** if it is not already present, and the example config is a long training run; tests are the faster initial smoke check.

```bash
python3 -m pip install -e .
python3 -m pytest -q
python3 -m compileall -q resnet_ablation scripts tests
```

CIFAR-10 ResNet-20 example:

```bash
python3 scripts/train.py --config configs/cifar10_resnet20.yaml
```

Evaluate an existing model using an explicitly selected checkpoint:

```bash
python3 scripts/eval.py \
  --config configs/cifar10_resnet20.yaml \
  --ckpt resnet_ablation/checkpoints/cifar10_resnet20/last.pt
```

Run different seed/output combinations **sequentially**, avoiding checkpoint/log collisions:

```bash
for seed in 1 2 3; do
  python3 scripts/train.py \
    --config configs/cifar10_resnet20.yaml \
    --seed "$seed" \
    --out-suffix "seed_${seed}"
done
```

`--out-suffix` appends a suffix to the `train.out_dir` and `train.tb_dir` defined by the YAML file. Choose independent output directories for all compared conditions. Checkpoints are `last.pt` and (when a validation improvement is recorded) `best.pt` in the run directory. `config.json` records the resolved configuration and basic runtime metadata; `results.csv` gets one row per **validation** epoch, not necessarily every training epoch. For reproducible runs, also record dependency versions, device, code commit, dataset split, transforms and independent seeds.

**Resume limitation / 断点恢复限制:** The current entry point may repeat the saved epoch and does not restore all RNG state. Do not rely on bitwise equivalence between uninterrupted and resumed training until the P0 fixes and recovery tests in the review document have landed.

## Config reference / 配置速查

Four top-level groups: `model`, `optim`, `train`, `data`. Unknown keys and selected enum values are rejected, but additional numerical checks remain to be implemented. Verify requested optimizer against `scripts/train.py`, which currently instantiates SGD regardless of the config's `optim.name` value.

| Group | Principal options | Notes |
| --- | --- | --- |
| `model` | `arch`, `num_classes`, `shortcut`, `stem`, `downsample`, `width_mult`, `attention`, `se_ratio`, `drop_path_rate` | CIFAR architecture has a validated `6n+2` depth; B/C alias issue applies to `BasicBlock`. |
| `optim` | `name`, `lr`, `momentum`, `weight_decay`, `nesterov`, `warmup_epochs`, `sched`, `milestones`, `gamma` | Supported `sched` values: `cosine`, `multistep`; warmup/cosine horizon needs further verification. |
| `train` | `epochs`, `batch_size`, `num_workers`, `amp`, `clip_grad_norm`, `val_interval`, `out_dir`, `tb_dir`, `resume`, `seed`, `label_smoothing` | `resume` points to an existing checkpoint. Load checkpoints only from trusted sources. |
| `data` | `name`, `root`, `aug`, `mixup_alpha`, `cutmix_alpha` | Supported names: `cifar10`, `cifar100`, `imagenet`; `aug`: `standard`, `strong`. |

Representative existing configurations include `configs/cifar10_resnet20.yaml` (baseline), `configs/cifar10_resnet20_se.yaml` (SE), `configs/cifar10_resnet110_optionA.yaml` (deeper CIFAR network), `configs/cifar100_resnet56_se_mix.yaml` (multi-factor), `configs/imagenet_resnet50_baseline.yaml` (ImageNet baseline) and `configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml` (multi-factor). **Multi-factor configurations are examples, not one-variable causal ablations.** Check each YAML file rather than assuming the filename encodes every difference.

## Experiment design / 消融实验设计

Hold dataset and split, preprocessing, model depth/width, optimizer, epoch count, LR trajectory, batch size and seed set fixed while changing one intended variable. For example, compare `attention=none` vs `attention=se` on the same CIFAR-10 ResNet-20 base config, then run an independent `none` vs `eca` pair. For shortcut A vs B, report the projection's parameter-count change rather than calling the parameter budgets identical. Compare `proj` vs `avgproj` separately from a deep-stem change.

Use a held-out training/validation split for tuning and reserve the test set for final evaluation. Report individual seeds, mean and dispersion, parameter count, compute, training time and complete experimental configuration. **No benchmark scores, publication-level reproduction or CI pass are claimed here.** Full paper-linked experiment matrix and acceptance criteria: [review](docs/REVIEW_2026-09-21.md).

## Paper references / 论文依据

- He et al., [Deep Residual Learning for Image Recognition](https://ar5iv.labs.arxiv.org/html/1512.03385): original residual formulation, CIFAR `6n+2` and shortcut A/B/C descriptions.
- He et al., [Identity Mappings in Deep Residual Networks](https://ar5iv.labs.arxiv.org/html/1603.05027): pre-activation is a **separate** architecture; current blocks use post-addition ReLU.
- Hu et al., [Squeeze-and-Excitation Networks](https://ar5iv.labs.arxiv.org/html/1709.01507): SE channel reweighting.
- Wang et al., [ECA-Net](https://ar5iv.labs.arxiv.org/html/1910.03151): lightweight local channel interaction.

For the original Chinese-language architecture walkthrough, see [`ChatGPT-ResNet 项目构建.md`](ChatGPT-ResNet%20%E9%A1%B9%E7%9B%AE%E6%9E%84%E5%BB%BA.md). That document is historical design context; implementation and review notes above describe the present code state.
