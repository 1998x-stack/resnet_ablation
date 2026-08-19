# Professional Polish & Hardening — Design Spec

**Date:** 2026-08-19
**Status:** Approved (design), pending written-spec review
**Project:** `resnet_ablation` (Pluggable ResNet ablation: CIFAR + ImageNet)

## 1. Goal

Raise the professional quality of the existing `resnet_ablation` codebase and its
documentation. Work is executed in the order: **correctness → docs → scaffolding**.
No new model capabilities are added. Numerics of the existing `forward` paths must
remain unchanged.

## 2. Scope (what is IN)

- Correctness & quality fixes in existing modules.
- A full pytest harness that runs on CPU (no GPU, no real datasets) plus an
  end-to-end smoke path.
- A bilingual (EN primary / 中文 secondary) README rewrite.
- `results.csv` experiment logging.
- Config validation (fail-fast) + reproducibility hardening.

## 3. Scope (what is OUT)

- Multi-GPU / DataParallel (**not** in this pass; can be added later).
- Restructuring the 100 KB Chinese design doc (kept as-is, only linked from README).
- Any change to trained-checkpoint compatibility or model numerics.

---

## Section 1 — Correctness & quality fixes

### config.py
- Remove the never-wired `drop_prob` field (declared but unused).
- Wire up `warmup_epochs` into a real LR warmup schedule (see below).
- Convert inline `# NEW:` roadmap comments into proper field docstrings.

### LR warmup
- Replace the bare scheduler with a proper schedule chain built in `scripts/train.py`:
  `LambdaLR` linear warmup (0 → base lr over `warmup_epochs`) chained via
  `torch.optim.lr_scheduler.SequentialLR` into cosine/multistep.
- When `warmup_epochs == 0`, behave exactly as today (no chaining).
- Note: current code steps a per-iteration schedule blindly; MultiStepLR is stepped
  per-epoch. Preserve that distinction and make the warmup step mode explicit.

### engine/trainer.py
- Remove the vestigial unused `_` third argument from `train_one_epoch` / `validate`.
- Wire declared-but-unused `log_interval` and `val_interval` into the loop
  (remove hardcoded `step % 50` and `epoch % val_interval` in `scripts/train.py`).
- Resume hardening: `load()` also restores/returns `best_acc`; best-accuracy
  tracking survives resume; `best.pt` selection stays consistent.
- Add docstrings to public methods. `forward` path unchanged (numerics identical).

### engine/evaluator.py
- Use `autocast` consistently under eval (harmless on CPU).
- Add optional top-5 metric for ImageNet.
- Keep returning val loss (validate already does).

### data.py
- Delete unused `_DLArgs` dataclass (dead code).
- Real docstrings; fail-fast clear error on unknown dataset/aug.

### factory.py / blocks.py
- Clean up `extras` handling in `build_model`; ensure unknown arch raises a clear
  `ValueError`. Add docstrings to public model/factory functions.

## Section 2 — Tests (full harness)

`pytest` suite under `tests/`, all CPU-runnable:

- `tests/test_config.py`: load & validate good/bad enums, missing required keys.
- `tests/test_augment.py`: `mixup`/`cutmix` shapes, soft-target rows sum to 1,
  `lam` in [0,1]; `rand_bbox` bounds.
- `tests/test_metrics.py`: `topk_correct` counts for top-1/top-5.
- `tests/test_blocks.py`: Shortcut A zero-pad semantics, SE/ECA parametric,
  DropPath identity in train vs identity when `drop_prob=0`, `_eca_kernel` odd output.
- `tests/test_models.py`: each arch builds, forward returns `(B, num_classes)`,
  param-count sanity, CIFAR `6n+2` assertion.
- `tests/test_e2e.py`: train ResNet-20 for 2 epochs on a seeded synthetic/CIFAR
  subset on CPU (AMP off) — assert loss is finite, val acc computed, `results.csv`
  written; assert determinism across two runs with same seed.

## Section 3 — Documentation (bilingual README, option A)

Rewrite `README.md` bilingual (EN primary + 中文 secondary):
- What it is + one-line feature map.
- Project structure / module map (file → responsibility table).
- Quickstart: install, train, eval, multi-seed, smoke.
- Config reference with enum values.
- Ablation experiment table mapping `configs/*.yaml` → what each tests.
- Dev workflow: `pytest`, `--smoke`, lint.
- Link to existing Chinese design doc (`ChatGPT-ResNet 项目构建.md`).

Keep the 100 KB design doc untouched (only linked).

## Section 4 — Scaffolding (A + C)

### Config validation & reproducibility
- `Config.from_yaml(path)` + `Config.validate()` fail fast on unknown keys and bad
  enum values; implemented via strict `__post_init__` plus a whitelist check on the
  raw YAML dict (keeps dataclass ergonomics).
- `train.py` / `eval.py` dump resolved config (seed, device, timestamp) to
  `<out_dir>/config.yaml` plus a JSON log line.

### results.csv
- Trainer appends one row per validation epoch:
  `timestamp,dataset,arch,seed,epoch,schedule,attn,shortcut,lr,train_loss,`
  `train_acc1,val_loss,val_acc1,mixup_alpha,cutmix_alpha,label_smoothing,checkpoint`.
- Written incrementally, flushed each epoch, no truncation; header written on new file.
- Uses the `--out-suffix`-ed out_dir so parallel-seed runs don't collide.
- `eval.py` optionally appends final results.

### Cleanup
- Remove vestigial dead code (`_DLArgs`, unused `_` args) — lands here so Section 1
  stays purely behavioral.

## Testing & verification
- `pytest -q` green on CPU.
- `ruff check` (or configured linter) — confirm tooling available.
- Smoke run: `python scripts/train.py --config configs/cifar10_resnet20.yaml --smoke`.

## Out of scope (explicitly not in this pass)
- Multi-GPU / DataParallel.
- Design-doc restructuring.
- Numeric/model changes.
DOC
echo "written:" && wc -l docs/superpowers/specs/2026-08-19-professional-polish-design.md
