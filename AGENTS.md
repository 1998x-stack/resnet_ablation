# AGENTS.md — ResNet Ablation Factory

Repository-wide instructions for Codex, OpenCode and other coding agents. Read this file before modifications. These instructions supplement, but do not override, higher-priority user/system requirements.

## Mission

Build a maintainable, **paper-grounded and scientifically auditable** ResNet ablation platform. Prioritize correct experimental attribution, reproducibility, tests, clean factory architecture and transparent reporting over feature count.

### Non-negotiable rules

1. Primary research source: He et al., [Deep Residual Learning for Image Recognition, arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Map claims to exact section/equation/table/figure and to code/tests.
2. Distinguish **paper-reported**, **repository-measured**, **synthetic smoke**, and **not yet validated**. Never manufacture scores, graphs, completed runs or CI outcomes.
3. **No leakage:** Factory CIFAR training must use held-out train/val from the official training set; official test is accessed only by an explicitly requested final-test after model selection. ImageNet val is not a test set.
4. Change only the experimentally declared factor(s), or explicitly identify a factorial design. Record unavoidable differences in parameters, MACs, training budget and augmentation.
5. Preserve historical scripts/train.py, scripts/eval.py and configs/*.yaml entry points. Do not conflate Legacy and Factory results.
6. Do not automatically download datasets, run extensive experiments, overwrite past runs, merge PRs or announce paper-level reproduction without explicit user direction.

## Reading order and source-of-truth

- README.md — supported capabilities, commands and limitations;
- docs/README.md — documents and archive;
- docs/paper/01_PAPER_TO_CODE.md — paper-to-code evidence and fidelity deltas;
- docs/paper/02_ABLATION_MATRIX.md — confounders, comparison plan and acceptance;
- docs/paper/03_RUNBOOK.md — execution/output conventions;
- configs/README.md — configuration groupings.

**Authority:** current source and passing tests > current paper methodology docs > historical archives. docs/REVIEW_2026-09-21.md and docs/superpowers/ document historical observations, not necessarily live bugs.

## Architecture and file ownership

| Location | Responsibility | Required invariants |
| --- | --- | --- |
| resnet_ablation/models/ | BasicBlock, Bottleneck, CIFAR/ImageNet model construction | Shapes, 6n+2 counts, shortcut A/B/C, stride v1/v1.5, finite gradients |
| resnet_ablation/research/factories.py | Registry, data split, optimizer/loss/schedule | Explicit factory registration, deterministic disjoint splits, valid schedule boundaries |
| resnet_ablation/research/runner.py | Train, evaluate, metadata, RNG and checkpoints | Step-accurate optimizer LR, separate final test, atomically written epoch checkpoints |
| resnet_ablation/research/sweep.py | Multi-seed configuration expansion | Dry-run is the default; fail on unapproved factor changes |
| resnet_ablation/research/report.py | Provenance-aware summaries and figures | Never impute missing values, reject uncontrolled comparisons |
| resnet_ablation/research/profile.py | Parameters/MACs and shapes | State counted operations; MACs are not full FLOPs |
| configs/paper/ | Paper controls, named matrices and extension studies | Distinguish paper features from post-2016 extensions |
| scripts/, configs/*.yaml, resnet_ablation/engine/ | Legacy compatibility | Preserve paths, commands and warning labels |
| tests/ | Offline contracts, synthetic end-to-end tests | No dataset downloads in CI |
| docs/paper/, docs/archive/ | Methodology vs historical plans | Avoid stale claims and maintain redirect paths |

## Implementation workflow

1. **Inspect:** Read relevant code, current tests, paper anchor and corresponding config. Determine whether change affects Factory Mode, Legacy or both.
2. **Frame:** State a testable hypothesis and the sole changed experimental factor (or explicit interaction design); name controlled variables and possible confounders.
3. **Build:** Use small, typed/composable PyTorch modules and existing registries. Avoid arbitrary imports, eval or hidden behavior driven by YAML. Fail fast on unsupported architectures/configurations.
4. **Validate:** Add/extend regression tests for model shapes, parameters, A/B/C projection, v1/v1.5 stride and backward; for data verify disjoint deterministic splits; for LR verify actual per-step optimizer values at boundaries; for resume verify next_epoch/global_step and RNG.
5. **Observe:** Preserve Loguru, metrics.jsonl and TensorBoard conventions; log run metadata, measured losses/error, LR, throughput and any diagnostic units/sampling.
6. **Report:** Include changed paths, evidence anchor, commands actually run, results, known limitations and commit/PR. Never imply a green synthetic test proves benchmark replication.

## Local verification before a PR

Run from repository root (do not start full dataset training as part of these checks):

~~~bash
python -m pip install -e ".[research]"
python -m compileall -q resnet_ablation scripts tests
python -m pytest -q tests/test_paper_research.py tests/test_paper_e2e.py tests/test_docs_links.py
python -m resnet_ablation.research.sweep \
  --manifest configs/paper/cifar10_depth_matrix.yaml
python -m resnet_ablation.research.profile \
  --config configs/paper/cifar10_paper_control.yaml
~~~

Run the full pytest suite when feasible for changes affecting shared interfaces; report tests not executed. Synthetic E2E must not download CIFAR/ImageNet. Keep CI lightweight, reproducible and free of long benchmark jobs.

## Experimental protocol

For any real ablation, document:

- Original reference and precise hypothesis;
- Baseline vs modified condition; changed factors and fixed factors;
- Dataset provenance, split SHA, seed list, preprocessing, normalization and test access policy;
- Optimizer, batch, total optimizer steps, exact learning rate trajectory, architecture parameters and MAC definition;
- Per-seed **training loss/error** and validation metrics, completion status, runtime/throughput, code revision and hardware;
- Aggregate mean and sample std only from completed seeds, along with failures and reasons;
- Frozen final-test results only if actually run, with no repeated test-set-driven model selection.

**Important attribution:** Increasing depth also changes capacity; Shortcut B/C add projection parameters; Plain/Residual ResNet-110 warmup must be controlled to isolate skip connection effects. ImageNet single-crop and multi-crop metrics are not interchangeable. SE/ECA/DropPath/Deep Stem/avgproj are later extensions, not 2016 paper ablations. The residual RMS diagnostic is a proxy, not a literal reproduction of Figure 7.

## Checkpoint, files and security

- Checkpoints should only be loaded from trusted sources (PyTorch pickle loading may execute code).
- Checkpoint writes must remain atomic and include optimizer/AMP/RNG/global_step and next_epoch when available.
- Resume is supported at completed epoch boundaries; do not promise arbitrary batch-level or bitwise cross-hardware replay.
- Keep local datasets, model weights, logs and runs out of Git. Use distinct run directories for each condition and seed.
- Preserve documentation links and compatibility paths; archive obsolete documents with a short redirect at their old location.
- Avoid unrelated code refactors in documentation PRs. Do not delete scientific evidence without checking provenance.

## Deliverable format for coding agents

A completed contribution should identify **what changed**, **why it matches the paper/hypothesis**, **what tests actually passed**, **what was not run**, and **where to review it**. Prefer Chinese explanatory docs while preserving exact Python identifiers and CLI flags in English. Commit in coherent units; open a reviewable PR rather than silently merging.

This root AGENTS.md applies across the repository unless a subdirectory adds a more specific AGENTS.md.
