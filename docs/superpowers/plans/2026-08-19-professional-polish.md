# Professional Polish & Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the professional quality of the `resnet_ablation` codebase and its docs — correctness fixes, a full CPU `pytest` harness, a bilingual README, `results.csv` logging, and fail-fast config validation — without changing model numerics.

**Architecture:** Work is grouped into four phases delivered as independent, self-contained tasks: (1) correctness fixes, (2) the pytest harness, (3) the bilingual README, (4) scaffolding (`results.csv` + config validation + reproducibility dumps). Each task keeps the codebase working and tested. No forward-path numerics change.

**Tech Stack:** Python 3.9+, PyTorch ≥2.2, torchvision, PyYAML, Loguru, TensorBoard, pytest, tqdm.

## Global Constraints

- **Numerics must not change.** The `forward()` paths of all blocks/models are identical to current behavior. Do not alter weight init, arithmetic order, or parameter counts. The two exceptions to "don't change behavior": (1) removing dead code that nothing calls, (2) the explicit new warmup schedule, which defaults to present behavior when `warmup_epochs == 0`.
- **Python floor 3.9.** Use `from __future__ import annotations` for forward-referenced annotations; avoid builtin generic syntax that needs 3.10 only if it can't be string-annotated. In `augment.py` the existing `tuple[int,...]`/`Tensor` annotations already parse on 3.9; keep them consistent.
- **Run paths** are all relative to the repo root: `resnet_ablation/`, `scripts/`, `configs/`, `tests/`.
- **Tests must run on CPU** with no real dataset download, and AMP disabled.
- **Docs:** README is bilingual — English primary, 中文 secondary. The file `ChatGPT-ResNet 项目构建.md` (100 KB) stays untouched and is only linked.
- **`results.csv` must be append-only** (flush per epoch, no truncation), written beside the checkpoint dir, and use the `--out-suffix`-ed path so seeded runs don't collide.
- `Config` must fail fast on unknown keys and invalid enum values, while keeping dataclass ergonomics (still constructed from a `dict`).
- Commit after every green test cycle, with conventional messages (`fix:`, `feat:`, `test:`, `docs:`).

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `resnet_ablation/config.py` | Config dataclasses + strict YAML loading/validation | Modify |
| `resnet_ablation/utils.py` | seed, param count, device helpers | Modify (minor) |
| `resnet_ablation/data.py` | dataloaders, transforms; remove dead `_DLArgs` | Modify |
| `resnet_ablation/engine/trainer.py` | train/validate loop, resume, checkpoint, results.csv | Modify |
| `resnet_ablation/engine/evaluator.py` | eval with top-1/top-5, autocast | Modify |
| `resnet_ablation/metrics.py` | correctness metrics | Modify (add top5) |
| `scripts/train.py` | CLI orchestration, warmup scheduler chain | Modify |
| `scripts/eval.py` | CLI orchestration, config dump, optional CSV | Modify |
| `README.md` | bilingual rewrite | Rewrite |
| `tests/conftest.py` | shared fixtures (models, synthetic data) | Create |
| `tests/test_config.py` | config load/validate tests | Create |
| `tests/test_augment.py` | mixup/cutmix/rand_bbox tests | Create |
| `tests/test_metrics.py` | topk_correct tests | Create |
| `tests/test_blocks.py` | shortcut/attention/droppath tests | Create |
| `tests/test_models.py` | arch build/shape/param tests | Create |
| `tests/test_e2e.py` | 2-epoch CPU end-to-end + determinism + csv | Create |
| `configs/*.yaml` | existing configs | Unchanged |

## Global Constraints

- Preserve exact forward numerics; no init/arith changes.
- Tests CPU-only; subprocess/data stubbed, no real dataset download.
- `results.csv` append-only, per-epoch flush, header written once on new file.
- `Config.from_yaml` + strict validation; dataclass construction unchanged for valid input.
- Conventional commits after each green task.
- Bilingual README only; `ChatGPT-ResNet 项目构建.md` untouched.

---

### Task 1: Config strict loading + remove `drop_prob`, wire warmup fields

**Files:**
- Modify: `resnet_ablation/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing dataclasses `ModelConfig`, `OptimConfig`, `TrainConfig`, `DataConfig`, `Config`.
- Produces:
  - `Config.from_yaml(path: str) -> Config` — raises `ValueError` (with offending key) on unknown keys or invalid enum value.
  - `Config.validate() -> None` — raises `ValueError` on invalid enum.
  - `ModelConfig` drops the `drop_prob` field.
  - `Config.as_dict() -> dict` — plain nested dict of the config (used for dumps in Task 6).
  - Field docstrings replace `# NEW:` comments; `warmup_epochs` stays (real warmup added in Task 4).

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest
import yaml
from tempfile import NamedTemporaryFile
from resnet_ablation.config import Config


def _write(tmp, text: str) -> str:
    with NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        return f.name


def test_from_yaml_valid_roundtrip():
    from resnet_ablation.config import ModelConfig, OptimConfig, TrainConfig, DataConfig
    path = _write(tmp, "model:\n  arch: resnet20_cifar\n  attention: se\n")
    cfg = Config.from_yaml(path)
    assert isinstance(cfg, Config)
    assert cfg.model.arch == "resnet20_cifar"
    assert cfg.model.attention == "se"
    # defaults survive for untouched groups
    assert cfg.optim.sched == "cosine"
    assert cfg.train.epochs == 200
    assert cfg.data.aug == "standard"


def test_from_yaml_unknown_key_raises():
    path = _write(tmp_path:=None, "model:\n  not_a_field: 1\n")
    with pytest.raises(ValueError, match="not_a_field"):
        Config.from_yaml(path)


def test_from_yaml_invalid_enum_raises():
    path = _write(tmp_path:=None, "model:\n  attention: bananas\n")
    with pytest.raises(ValueError, match="attention"):
        Config.from_yaml(path)


def test_config_as_dict_jsonable():
    import json
    cfg = Config()
    json.dumps(cfg.as_dict())  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `Config.from_yaml` does not exist (AttributeError).

- [ ] **Step 3: Implement**

In `resnet_ablation/config.py`, add strict loading. Define the allowed enum sets at module top and a `KNOWN_KEYS` map keyed by nested config name:

```python
from __future__ import annotations
import json
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Any, Dict

_VALID = {
    "ModelConfig": {
        "arch": {"resnet20_cifar","resnet56_cifar","resnet110_cifar",
                 "resnet18","resnet34","resnet50","resnet101","resnet152"},
        "shortcut": {"A","B","C"},
        "stem": {"cifar","imagenet_standard","imagenet_deep"},
        "downsample": {"proj","avgproj"},
        "num_classes": None, "width_mult": None, "drop_path_rate": None,
        "attention": {"none","se","eca"}, "se_ratio": None,
    },
    "OptimConfig": {
        "name": None, "lr": None, "momentum": None, "weight_decay": None,
        "nesterov": None, "warmup_epochs": None, "sched": {"cosine","multistep"},
        "milestones": None, "gamma": None,
    },
    "TrainConfig": {
        "epochs": None, "batch_size": None, "num_workers": None, "amp": None,
        "clip_grad_norm": None, "log_interval": None, "val_interval": None,
        "out_dir": None, "resume": None, "seed": None, "tb_dir": None,
        "label_smoothing": None,
    },
    "DataConfig": {
        "name": {"cifar10","cifar100","imagenet"}, "root": None,
        "aug": {"standard","strong"}, "mixup_alpha": None, "cutmix_alpha": None,
    },
}
_DROP = ("drop_prob",)  # removed field


@dataclass
class ModelConfig:
    """Model build options."""
    arch: str = "resnet20_cifar"
    num_classes: int = 10
    shortcut: str = "A"                 # A/B/C (CIFAR BasicBlock); ImageNet uses B projection
    stem: str = "cifar"                 # cifar / imagenet_standard / imagenet_deep
    downsample: str = "proj"            # proj (1x1 proj) / avgproj (ResNet-D avg+1x1)
    width_mult: float = 1.0
    drop_path_rate: float = 0.0         # Stochastic Depth max ratio
    attention: str = "none"             # none / se / eca
    se_ratio: float = 0.25              # SE reduction ratio (attention=se only)
    # NOTE: `drop_prob` removed (declared but never wired).
```

Remove the `drop_prob` line. Then append module-level loading/validation helpers and methods:

```python
def _check(value, allowed):
    if allowed is not None and value is not None and value not in allowed:
        raise ValueError(f"invalid value for {value}: allowed {sorted(allowed)}")


def _strict_kwargs(cls, raw: Dict[str, Any]):
    """Validate field names and enum values for one nested config dict."""
    allowed_fields = {f.name for f in fields(cls)}
    known = _VAL[cls.__name__]
    for key in raw:
        if key not in allowed_fields:
            raise ValueError(f"unknown key '{key}' for {cls.__name__}")
    for key, value in raw.items():
        _check(value, known.get(key))
    return raw


@dataclass
class TrainConfig:
    # ... (unchanged groups)
```

For each nested dataclass, add a `validate()` and the top-level `Config` gets `from_yaml`, `validate()`, `as_dict()`:

```python
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # split raw into group sections, else keep dataclass defaults
        section = {"model": raw.get("model", {}), "optim": raw.get("optim", {}),
                   "train": raw.get("train", {}), "data": raw.get("data", {})}
        # reject top-level unknown groups too
        known_groups = {"model", "optim", "train", "data"}
        for g in raw:
            if g not in known_groups:
                raise ValueError(f"unknown group '{g}'")
        cfg = cls(
            model=ModelConfig(**_common_kwargs(ModelConfig, section["model"])),
            optim=OptimConfig(**_common_kwargs(OptimConfig, section["optim"])),
            train=TrainConfig(**_common_kwargs(TrainConfig, section["train"])),
            data=DataConfig(**_common_kwargs(DataConfig, section["data"])),
        )
        cfg._validate_required()
        return cfg

    def _validate_required(self):
        pass  # hook for cross-field checks (currently none required)

    def validate(self) -> None:
        # re-run enum checks on current instance fields (cheap; used after edits)
        for grp in (ModelConfig, OptimConfig, TrainConfig, DataConfig):
            inst = getattr(self, {"ModelConfig":"model","OptimConfig":"optim",
                                  "TrainConfig":"train","DataConfig":"data"}[grp.__name__])
            for fld in fields(grp):
                val = getattr(inst, fld.name)
                allowed = _VAL[grp.__name__].get(fld.name)
                _check(val, allowed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model), "optim": asdict(self.optim),
            "train": asdict(self.train), "data": asdict(self.data),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py resnet_ablation/config.py
git commit -m "feat: strict Config.from_yaml/validate, remove dead drop_prob"
```

---

### Task 2: Remove dead code; add top-5 metric

**Files:**
- Modify: `resnet_ablation/data.py`, `resnet_ablation/metrics.py`, `resnet_ablation/models/factory.py`
- Test: `tests/test_metrics.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: existing modules.
- Produces:
  - `topk_correct(logits, targets, topk)` — **unchanged signature** (already returns a tuple of per-k correct counts); add `top5` usage in evaluator/Task 3.
  - `build_model(cfg)` — unchanged signature; internally cleaned-up `extras` handling.
  - `data.py`: no `_DLArgs`; functions `build_cifar10/100`, `build_imagenet`, `build_dataloaders` unchanged.

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:

```python
import torch
from resnet_ablation.metrics import topk_correct


def test_topk_correct_top1_top5():
    logits = torch.tensor([
        [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],   # correct class 1 -> top1 hit
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],   # correct class 0 -> top1 hit
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],   # correct class 5 -> top5 hit only
    ])
    targets = torch.tensor([1, 0, 5])
    c1, c5 = topk_correct(logits, targets, (1, 5))
    assert c1.item() == 2
    assert c5.item() == 3
```
(Note `topk_correct` already exists and passes today — this is a regression guard. Mark the test as the commit gate.)

`tests/test_models.py`:

```python
import torch
import pytest
from resnet_ablation.config import Config
from resnet_ablation.models.factory import build_model


def _arch_shapes():
    for a, nc in [("resnet20_cifar", 10), ("resnet110_cifar", 10),
                  ("resnet18", 1000), ("resnet50", 1000)]:
        yield a, nc


@pytest.mark.parametrize("arch,num_classes", [
    ("resnet20_cifar", 10), ("resnet56_cifar", 100),
    ("resnet110_cifar", 10), ("resnet18", 1000), ("resnet34", 1000),
    ("resnet50", 1000), ("resnet101", 1000), ("resnet152", 1000),
])
def test_forward_shape(arch, num_classes):
    cfg = Config()
    cfg.model.arch = arch
    cfg.model.num_classes = num_classes
    if "_cifar" not in arch:
        cfg.model.stem = "imagenet_standard"
    model = build_model(cfg).eval()
    x = torch.randn(2, 3, 32 if "_cifar" in arch else 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes), f"{arch} -> {out.shape}"


def test_unknown_arch_raises():
    from resnet_ablation.models.factory import build_model
    cfg = Config(); cfg.model.arch = "resnet999"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_param_count_ranges():
    from resnet_ablation.utils import count_params
    cfg = Config(); cfg.model.arch = "resnet20_cifar"
    n = count_params(build_model(cfg))
    assert 0.1e6 < n < 1.0e6  # ResNet-20 CIFAR ~273k
```

- [ ] **Step 2: Run to verify fail (for top5, models pass already; run and note)**

Run: `pytest tests/test_metrics.py tests/test_models.py -v`
Expected: metrics PASS (regression), models PASS (guards). If either FAIL, debug implementation before continuing.

- [ ] **Step 3: Implement**

Remove the dead `_DLArgs` dataclass from `resnet_ablation/data.py` (delete the 3-line block and its import of `dataclass` if unused). Add real docstrings to the `build_*` functions and `build_dataloaders`, and make `build_dataloaders` raise on unknown names before calling (already does via `NotImplementedError` — keep, but make it `ValueError` for consistency):

```python
def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Build (train, val) dataloaders for the configured dataset and aug."""
    name = cfg.data.name.lower()
    if name == "cifar10":
        return build_cifar10(cfg.data.root, cfg.train.batch_size, cfg.train.num_workers, cfg.data.aug)
    ...
    raise ValueError(f"Unknown dataset: {cfg.data.name} (expected cifar10/cifar100/imagenet)")
```

In `factory.py`, simplify `build_model` to avoid passing literal `"imagenet_standard"` when stem isn't set for imagenet (keep default branch but add docstring):

```python
def build_model(cfg) -> "nn.Module":
    """Build the model matching cfg.model.arch. Raises ValueError on unknown arch."""
    m = cfg.model
    cfg.extras = dict(attention=m.attention, se_ratio=m.se_ratio,
                      drop_path_rate=m.drop_path_rate, downsample=m.downsample,
                      stem=_resolved_stem(m.stem))
    ...
```

Where `_resolved_stem` maps `cifar`→`imagenet_standard` for imagenet archs. Implement it inline:
```python
def _resolved_stem(stem: str) -> str:
    return stem if stem in ("imagenet_standard", "imagenet_deep") else "imagenet_standard"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_metrics.py tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add resnet_ablation/data.py resnet_ablation/metrics.py resnet_ablation/models/factory.py tests/test_metrics.py tests/test_models.py
git commit -m "chore: remove dead code, add docstrings, model/metric tests"
```

---

### Task 3: Trainer cleanup (drop `_` arg, wire log_interval) + evaluator top-5/autocast

**Files:**
- Modify: `resnet_ablation/engine/trainer.py`, `resnet_ablation/engine/evaluator.py`, `scripts/train.py`, `scripts/eval.py`
- Test: `tests/test_e2e.py` establishes trainer works (written in Task 5; here run existing tests + hand smoke)

**Interfaces:**
- Consumes: `Config`, `topk_correct((1,))` / new `topk_correct(logits, targets, (1,5))`.
- Produces:
  - `Trainer.train_one_epoch(self, epoch, loader) -> Dict[str,float]` — **no 3rd arg**.
  - `Trainer.validate(self, epoch, loader, topk=(1,)) -> Dict[str,float]` — returns `{"val_loss", "val_acc1", <opt> "val_acc5"}`.
  - `Trainer.__init__(..., log_interval=50)` — accepts `log_interval`.
  - `evaluate(model, loader, topk=(1,)) -> Dict[str,float]` — returns `acc1` and `acc5`.

- [ ] **Step 1: Make the change in trainer.py**

Remove the `_` third parameter from both `train_one_epoch(self, epoch, loader)` and `validate(self, epoch, loader, ...)`. Add `log_interval` param to `__init__` with default `50`, store `self.log_interval`, and replace the hardcoded `if step % 50 == 0:` with `if step % self.log_interval == 0:`.

Change `validate` signature to accept `topk` and return top-5 when requested:

```python
@torch.no_grad()
def validate(self, epoch: int, loader, topk=(1,)) -> Dict[str, float]:
    self.model.eval()
    loss_sum, acc1_sum, n = 0.0, 0.0, 0
    acc5_sum = 0.0
    for images, targets in loader:
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        logits = self.model(images)
        loss = self.ce(logits, targets)
        corrects = topk_correct(logits, targets, topk)  # 1 or 2-tuple
        bs = images.size(0)
        loss_sum += loss.item() * bs
        acc1_sum += corrects[0].item()
        if len(corrects) > 1:
            acc5_sum += corrects[1].item()
        n += bs
    res = {"val_loss": loss_sum / n, "val_acc1": acc1_sum / n}
    if len(corrects) > 1:
        res["val_acc5"] = acc5_sum / n
    return res
```
(Ensure `corrects` defined in scope for the return check — restructure to define `res` after loop using collected sums.)

- [ ] **Step 2: Update evaluator.py**

```python
@torch.no_grad()
def evaluate(model: nn.Module, loader, topk=(1,)) -> Dict[str, float]:
    """Evaluate and return acc at requested topk levels (default top-1)."""
    model.eval()
    device = next(model.parameters()).device
    sums = {k: 0.0 for k in topk}
    n = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
        corrects = topk_correct(logits, targets, topk)
        for k, c in zip(topk, corrects):
            sums[k] += c.item()
        n += images.size(0)
    out = {f"acc{k}": sums[k] / n for k in topk}
    return out
```

- [ ] **Step 3: Update scripts/train.py and eval.py call sites**

`scripts/train.py` — remove the `None` third arg at both calls; pass topk for ImageNet:

```python
stats = trainer.train_one_epoch(epoch, train_loader)
...
v = trainer.validate(epoch, val_loader, (1, 5) if cfg.data.name == "imagenet" else (1,))
```

`scripts/eval.py` — use evaluator top-5 for imagenet:

```python
res = evaluate(model, val_loader, (1, 5) if cfg.data.name == "imagenet" else (1,))
```

- [ ] **Step 4: Hand smoke**

Run: `python -c "
import sys; sys.path.insert(0,'.')
from resnet_ablation.data import build_dataloaders
from resnet_ablation.config import Config
from resnet_ablation.models.factory import build_model
cfg = Config(); cfg.data.name='cifar10'
m = build_model(cfg)
from resnet_ablation.engine.evaluator import evaluate
import torch
m.eval()
x = torch.randn(2,3,32,32)
print('smoke ok', evaluate(m, None) if False else 'shape', m(x).shape)
"
```
Expected: prints shape without error; trainer imports fine.

- [ ] **Step 5: Commit**

```bash
git add resnet_ablation/engine/trainer.py resnet_ablation/engine/evaluator.py scripts/train.py scripts/eval.py
git commit -m "refactor: drop unused trainer arg, wire log_interval, add top5+autocast"
```

---

### Task 4: LR warmup via SequentialLR

**Files:**
- Modify: `resnet_ablation/config.py` (warmup documented — already), `scripts/train.py`
- Test: `tests/test_schedulers.py` (new)

**Interfaces:**
- Consumes: `cfg.optim.warmup_epochs`, `cfg.optim.lr`, `cfg.optim.sched`, `cfg.optim.milestones`, `cfg.optim.gamma`.
- Produces: `build_scheduler(optimizer, optim_cfg, epochs) -> torch.optim.lr_scheduler._LRScheduler` used by `train.py`, respecting `warmup_epochs == 0` → original behavior.

- [ ] **Step 1: Write failing test**

`tests/test_schedulers.py`:

```python
import torch
from resnet_ablation.config import OptimConfig
from resnet_ablation.scheduler import build_scheduler
from torch.optim import SGD


def _make(epochs=200, warmup=5):
    o = OptimConfig(lr=0.1, warmup_epochs=warmup, sched="cosine")
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=o.lr)
    return opt, build_scheduler(opt, o, epochs)


def test_warmup_linear_initial():
    opt, sched = _make(warmup=5)
    # first step after warmup phase start gives tiny lr
    sched.step()
    lr = opt.param_groups[0]["lr"]
    assert 0.0 < lr < 0.1


def test_no_warmup_equals_cosine():
    from torch.optim.lr_scheduler import CosineAnnealingLR
    o = OptimConfig(lr=0.1, warmup_epochs=0, sched="cosine")
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=o.lr)
    sched = build_scheduler(opt, o, epochs=200)
    ref = CosineAnnealingLR(opt, T_max=200)
    sched.step()
    ref.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(ref.get_last_lr()[0])
```
(When `warmup_epochs == 0`, `build_scheduler` returns the plain cosine schedule, so a single `step()` matches `CosineAnnealingLR`.)

- [ ] **Step 2: Run, verify fails**

Run: `pytest tests/test_schedulers.py -v`
Expected: FAIL — `build_scheduler` not found.

- [ ] **Step 3: Implement a scheduler module**

Create `resnet_ablation/scheduler.py`:

```python
from __future__ import annotations
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, MultiStepLR, SequentialLR

def _linear_warmup(total_warmup: int) -> callable:
    """Return a step -> fraction lambda for a linear 0->1 warmup."""
    return lambda step: min(1.0, step / max(1, total_warmup))

def build_scheduler(optimizer: Optimizer, ocfg, total_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR schedule honoring warmup_epochs. warmup_epochs=0 → plain schedule."""
    warm = int(ocfg.warmup_epochs)
    if ocfg.sched == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=total_epochs)
    else:
        main = MultiStepLR(optimizer, milestones=list(ocfg.milestones), gamma=ocfg.gamma)
    if warm <= 0:
        return main
    warm_sched = LambdaLR(optimizer, lr_lambda=_from_warmup(warm))
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm_sched, main], milestones=[warm])
```

(Also make the comment `# step mode` explicit: cosine/per-iter in trainer currently steps each iteration; with warmup+SequentialLR we step per epoch in `train.py`. Keep the existing per-iteration behavior for cosine to avoid numerics change: because warmup uses per-epoch milestones, simplest correct approach is step both warm and main per epoch. Document the choice in the scheduler docstring.)

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_schedulers.py -v`
Expected: PASS.

- [ ] **Step 5: Wire train.py uses build_scheduler**

Replace the inline `if cfg.optim.sched == "cosine": sched = ... else ...` block with:

```python
from resnet_ablation.scheduler import build_scheduler
sched = build_scheduler(opt, cfg.optim, cfg.train.epochs)
```

- [ ] **Step 6: Commit**

```bash
git add resnet_ablation/scheduler.py scripts/train.py tests/test_schedulers.py
git commit -m "feat: linear-warmup LR scheduler (warmup_epochs)"
```

---

### Task 5: End-to-end CPU test (2 epochs, results.csv) + determinism

**Files:**
- Create: `tests/conftest.py`, `tests/test_e2e.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Consumes: `build_dataloaders` (stubbed), `Trainer`, `Config`, `build_scheduler`.
- Produces: pytest fixtures + E2E harness; **no csv assertion here** (results.csv writing is Task 6). No production API changes.

- [ ] **Step 1: Write conftest fixtures**

`tests/conftest.py`:

```python
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from resnet_ablation.config import Config


@pytest.fixture
def tiny_data():
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    y = torch.randint(0, 10, (64,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    return loader


@pytest.fixture
def train_cfg():
    cfg = Config()
    cfg.train.batch_size = 8
    cfg.train.epochs = 2
    cfg.train.amp = False
    cfg.data.name = "cifar10"
    cfg.model.arch = "resnet20_cifar"
    cfg.train.out_dir = "_tmptest_ckpt"
    cfg.train.tb_dir = "_tmptest_runs"
    return cfg
```

- [ ] **Step 2: Write the failing E2E test**

`tests/test_e2e.py`:

```python
import torch
from pathlib import Path
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from resnet_ablation.config import Config
from resnet_ablation.utils import set_seed
from resnet_ablation.engine.trainer import Trainer
from resnet_ablation.models.factory import build_model
from resnet_ablation.scheduler import build_scheduler


def _make_loader(n=64, bs=8, num_classes=10):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True)


def _train_run(tmp_path):
    """Run 2 epochs on a tiny CPU loader; return (cfg, last_train, last_val)."""
    set_seed(0)
    cfg = Config()
    cfg.train.batch_size = 8
    cfg.train.epochs = 2
    cfg.train.amp = False
    cfg.model.arch = "resnet20_cifar"
    cfg.model.num_classes = 10
    out = str(tmp_path / "ckpt")
    tb = str(tmp_path / "runs")
    model = build_model(cfg)
    opt = SGD(model.parameters(), lr=0.01)
    sched = build_scheduler(opt, cfg.optim, cfg.train.epochs)
    trainer = Trainer(model, opt, sched, torch.device("cpu"),
                      out_dir=out, tb_dir=tb, amp=False, num_classes=10)
    loader = _make_loader()
    for e in range(cfg.train.epochs):
        s = trainer.train_one_epoch(e, loader)
        v = trainer.validate(e, loader)
    return cfg, s, v


def test_e2e_loss_finite(tmp_path):
    cfg, s, v = _train_run(tmp_path)
    assert s["loss"] == s["loss"]  # NaN check (NaN != NaN)
    assert -1e9 < v["val_loss"] < 1e9
    assert 0 <= v["val_acc1"] <= 1


def test_e2e_determinism(tmp_path):
    r1 = _train_run(tmp_path)
    r2 = _train_run(tmp_path)
    assert r1[1]["loss"] == r2[1]["loss"]  # exact float determinism on CPU
    assert r1[2]["val_acc1"] == r2[2]["val_acc1"]
```

- [ ] **Step 3: Run, verify passes**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS (trainer + tiny loader works, loss finite, determinism holds).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_e2e.py
git commit -m "test: e2e CPU 2-epoch train + csv + determinism"
```

---

### Task 5a: augment unit tests (mixup / cutmix / rand_bbox)

**Files:**
- Create: `tests/test_augment.py`
- Test: `tests/test_augment.py`

**Interfaces:**
- Consumes: `resnet_ablation.augment.mixup(images, targets, num_classes, alpha, smoothing)` → `(mixed, soft)`, `cutmix(...)` → `(mixed, soft)`, `rand_bbox(W, H, lam)`.
- Produces: regression guarantees for the existing augment helpers (no API change).

- [ ] **Step 1: Write the tests**

`tests/test_augment.py`:

```python
import torch
from resnet_ablation.augment import mixup, cutmix, rand_bbox, one_hot


def _batch(n=16, c=10, hw=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3, hw, hw)
    y = torch.randint(0, c, (n,))
    return x, y


def test_mixup_shape_and_softness():
    x, y = _batch()
    mixed, lab = mixup(x, y, num_classes=10, alpha=1.0)
    assert mixed.shape == x.shape
    assert lab.shape == (x.size(0), 10)
    assert torch.allclose(lab.sum(dim=1), torch.ones(x.size(0)))  # rows sum to 1


def test_cutmix_shape_and_softness():
    x, y = _batch()
    mixed, lab = cutmix(x, y, num_classes=10, alpha=1.0)
    assert mixed.shape == x.shape
    assert lab.shape == (x.size(0), 10)
    assert torch.allclose(lab.sum(dim=1), torch.ones(16))


def test_mixup_alpha_zero_raises():
    import pytest
    x, y = _batch()
    with pytest.raises(ValueError):
        mixup(x, y, num_classes=10, alpha=0.0)


def test_cutmix_alpha_zero_raises():
    import pytest
    x, y = _batch()
    with pytest.raises(ValueError):
        cutmix(x, y, num_classes=10, alpha=0.0)


def test_rand_bbox_bounds():
    for _ in range(50):
        x1, y1, x2, y2 = rand_bbox(32, 32, 0.5)
        assert 0 <= x1 < x2 <= 32
        assert 0 <= y1 < y2 <= 32


def test_one_hot_smoothing():
    labs = torch.tensor([0, 3])
    y = one_hot(labs, 10, smoothing=0.1)
    assert y.shape == (2, 10)
    assert torch.allclose(y.sum(dim=1), torch.ones(2))
    assert torch.allclose(y[0, 0].item(), 0.91)  # (1-0.1) + 0.1/10
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_augment.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_augment.py
git commit -m "test: augment mixup/cutmix/rand_bbox/one_hot"
```

---

### Task 5b: blocks unit tests (shortcut A, SE/ECA, DropPath, eca kernel)

**Files:**
- Create: `tests/test_blocks.py`
- **Interfaces:**
- Consumes: `resnet_ablation.models.blocks` (ShortcutA, SEModule, ECAModule, DropPath, _eca_kernel, conv3x3).
- Produces: none.

- [ ] **Step 1: Write the tests**

`tests/test_blocks.py`:

```python
import torch
from torch import nn
from resnet_ablation.models.blocks import ShortcutA, SEModule, ECAModule, DropPath, _eca_kernel, BasicBlock


def test_shortcut_a_zero_pad():
    s = ShortcutA(64, 128, stride=1)
    x = torch.randn(2, 64, 8, 8)
    out = s(x)
    assert out.shape == (2, 128, 8, 8)
    assert torch.allclose(out[:, 64:], torch.zeros(2, 64, 8, 8))  # padded zeros


def test_shortcut_a_stride_downsample():
    s = ShortcutA(64, 64, stride=2)
    x = torch.randn(2, 64, 8, 8)
    out = s(x)
    assert out.shape == (2, 64, 4, 4)


def test_se_module_shape():
    m = SEModule(channels=64, ratio=0.25).eval()
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_eca_module_shape():
    m = ECAModule(channels=64).eval()
    x = torch.randn(2, 64, 8, 8)
    with torch.no_grad():
        out = m(x)
    assert out.shape == x.shape


def test_eca_kernel_odd_gte3():
    for c in range(16, 1024, 32):
        k = _eca_kernel(c)
        assert k >= 3 and k % 2 == 1


def test_droppath_identity_when_disabled():
    m = DropPath(drop_prob=0.0).eval()
    x = torch.randn(4, 8, 2, 2)
    with torch.no_grad():
        assert torch.equal(m(x), x)


def test_droppath_zero_keep_in_train():
    m = DropPath(drop_prob=1.0).train()  # keep=0
    x = torch.randn(4, 8, 2, 2)
    out = m(x)
    assert torch.allclose(out, torch.zeros_like(x))  # keep=0 -> /0 guarded, mask all zero


def test_basicblock_forward_shape():
    b = BasicBlock(in_planes=16, planes=16, stride=1, shortcut="A")
    x = torch.randn(2, 16, 16, 16)
    out = b(x)
    assert out.shape == x.shape
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_blocks.py -v`
Expected: PASS. (If `DropPath(drop_prob=1.0)` division-by-zero fails, fix DropPath to guard `keep == 0` — see note.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_blocks.py
git commit -m "test: block unit tests (shortcut/SE/ECA/DropPath)"
```

---

### Task 6: `results.csv` logging + resolved-config dump in train.py and eval.py

**Files:**
- Modify: `resnet_ablation/engine/trainer.py` (csv writer), `resnet_ablation/config.py` (`to_json_line`), `scripts/train.py`, `scripts/eval.py`
- Test: `tests/test_e2e.py` already asserts csv (from Task 5)

**Interfaces:**
- Consumes: `Config.as_dict()`.
- Produces: per-epoch `results.csv` row; `<out_dir>/config.json` dump. Extends `tests/test_e2e.py` with a csv-content assertion.

- [ ] **Step 1: Implement results.csv in Trainer**

Add to `Trainer.__init__`:

```python
self.results_csv = Path(out_dir) / "results.csv"
self._csv_ts = ""
self._csv_meta = {"dataset": None, "arch": None, "seed": None,
                  "attn": "", "shortcut": ""}
```

Add a helper on the Trainer to log one row, set via `train.py` before the loop:

```python
def write_results_row(self, epoch, stats, val_stats, lr):
    import csv
    csv_exists = self.results_csv.exists()
    with open(self.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not csv_exists:
            w.writerow(["timestamp","dataset","arch","seed","epoch","schedule",
                        "attn","shortcut","lr","train_loss","train_acc1",
                        "val_loss","val_acc1","mixup_alpha","cutmix_alpha",
                        "label_smoothing","checkpoint"])
        w.writerow([
            self._csv_ts, self._csv_meta["dataset"], self._csv_meta["arch"],
            self._csv_meta["seed"], epoch, self.sched.__class__.__name__,
            self._csv_meta["attn"], self._csv_meta["shortcut"], lr,
            stats["loss"], stats["acc1"], val_stats["val_loss"], val_stats["val_acc1"],
            self.mixup_alpha, self.cutmix_alpha, self.label_smoothing,
            (self.ckpt_dir / "best.pt").name if (self.ckpt_dir / "best.pt").exists() else "last.pt",
        ])
        f.flush()
```

(Exactly 17 columns: timestamp, dataset, arch, seed, epoch, schedule, attn,
shortcut, lr, train_loss, train_acc1, val_loss, val_acc1, mixup_alpha,
cutmix_alpha, label_smoothing, checkpoint. Call `write_results_row` once per
validation epoch from `train.py`.)

- [ ] **Step 2: Dump resolved config in train.py and eval.py**

In `train.py` after building `cfg` and before training:

```python
runtime = {"device": str(device), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": cfg.train.seed}
full = cfg.as_dict(); full["runtime"] = runtime
import json
Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
(Path(cfg.train.out_dir) / "config.json").write_text(json.dumps(full, indent=2))
```

Set trainer meta and timestamp before the loop, and call `write_results_row`
at each validation step (right where `val_acc1` is logged):

```python
import time
trainer._csv_ts = time.strftime("%Y-%m-%d %H:%M:%S")
trainer._csv_meta = {"dataset": cfg.data.name, "arch": cfg.model.arch,
                     "seed": cfg.train.seed, "attn": cfg.model.attention,
                     "shortcut": cfg.model.shortcut}
...
# inside the validation branch, after computing v:
trainer.write_results_row(epoch, stats, v, opt.param_groups[0]["lr"])
```

- [ ] **Step 3: Add the csv-content assertion to `tests/test_e2e.py`**

Append to `tests/test_e2e.py`:

```python
def test_e2e_writes_csv(tmp_path):
    import csv
    from pathlib import Path
    cfg, s, v = _train_run(tmp_path)   # runs trainer, which calls write_results_row
    csvf = Path(tmp_path / "ckpt" / "results.csv")
    assert csvf.exists()
    rows = list(csv.reader(csvf.open()))
    assert rows[0][0] == "timestamp"  # header present
    assert len(rows) >= 2             # header + >=1 epoch row
    assert rows[-1][4] == "1"         # last epoch index

Note: `test_e2e_writes_csv` needs the trainer path to call `write_results_row`
during `_train_run`. Because csv-writing is wired through `train.py`, add a
direct call in `_train_run` after the loop so the test is self-contained:
`trainer.write_results_row(1, s, v, 0.01)` using the helper directly.
```
Run: `python -m pyflakes scripts/train.py` (or `python -m py_compile scripts/train.py script/eval.py`) — no syntax errors.
Expected: pass / no errors.

- [ ] **Step 4: Commit**

```bash
git add resnet_ablation/engine/trainer.py scripts/train.py scripts/eval.py
git commit -m "feat: results.csv per-epoch logging + resolved-config dump"
```

---

### Task 7: Bilingual README rewrite

**Files:**
- Rewrite: `README.md`

**Interfaces:**
- Consumes: filenames/configs listed below.

- [ ] **Step 1: Write README.md (bilingual)**

Structure the file with EN primary and 中文 secondary as agreed. Minimum sections:

```markdown
# ResNet Ablation (Pluggable) / 可插拔 ResNet 消融库

## What it is（简介）
## Feature map（功能总览）
## Project structure（项目结构 / module map）
| Path | Responsibility | 职责 |
|------|----------------|------|
| `resnet_ablation/config.py` | strict Config dataclasses + YAML loading | 配置校验 |
| `resnet_ablation/models/` | blocks, CIFAR & ImageNet ResNet, factory | 模型构建 |
| `resnet_ablation/engine/` | trainer (AMP/checkpoint/results.csv), evaluator | 训练/评估 |
| `resnet_ablation/data.py` | dataloaders & transforms | 数据 |
| `resnet_ablation/augment.py` | Mixup/CutMix/rand_bbox | 数据增强 |
| `scripts/train.py` | CLI training entry | 训练入口 |
| `scripts/eval.py` | CLI eval entry | 评估入口 |
| `tests/` | pytest harness (CPU) | 单元与端到端测试 |

## Quickstart（快速开始）
pip install -e .
python scripts/train.py --config configs/cifar10_resnet20.yaml
python scripts/eval.py --config configs/cifar10_resnet20.yaml --ckpt <out>/last.pt
for s in 1 2 3; do python scripts/train.py --config configs/cifar10_resnet20.yaml --seed $s --out-suffix "s$s"; done

## Config reference（配置项）— table with groups, fields, allowed enums

## Ablation experiments（消融实验表）
| configs/*.yaml | what it tests |
|---|---|
| `configs/cifar10_resnet20.yaml` | CIFAR-10 baseline, Shortcut A |
| `configs/cifar10_resnet20_se.yaml` | CIFAR-10 + SE |
| `configs/cifar10_resnet20_se_ecadrop_mix.yaml` | CIFAR-10 + SE + ECA + DropPath + Mixup |
| `configs/cifar10_resnet56_ablate_optionB.yaml` | shortcut ablate A→B |
| `configs/cifar10_resnet110_optionA.yaml` | deep CIFAR-110 baseline |
| `configs/cifar100_resnet56_se_mix.yaml` | CIFAR-100 + SE + Mixup |
| `configs/imagenet_resnet50_baseline.yaml` | ImageNet R50 baseline |
| `configs/imagenet_resnet50.yaml` | ImageNet R50 |
| `configs/imagenet_resnet50_deepstem_resnetd_se.yaml` | R50 DeepStem + ResNet-D + SE |
| `configs/imagenet_resnet101_deepstem_resnetd_eca_dpr.yaml` | R101 DeepStem + ResNet-D + ECA + DropPath |

## Dev workflow
pytest -q        # run test suite
ruff check .     # lint (if available)

## Design doc（设计文档）
See `ChatGPT-ResNet 项目构建.md` for the original architecture walkthrough (Chinese).
```

Include every `configs/*.yaml` in the ablation table.

- [ ] **Step 2: Verify links/paths exist**

Run: `ls configs/*.yaml` and confirm every filename referenced exists.
- reflect: run `python -c "import resnet_ablation; print('import ok')"`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite bilingual README (EN + 中文)"
```

---

### Task 8: Full-suite verification + lint + final review

**Files:**
- None (verification).

- [ ] **Step 1: Run the whole suite**

Run: `pytest -q`
Expected: all tests pass (test_config, test_augment, test_metrics, test_blocks, test_models, test_schedulers, test_e2e).

- [ ] **Step 2: Run lint / compile**

Run: `python -m compileall resnet_ablation scripts tests` then `python -m pyflakes resnet_ablation scripts` (if available).
Expected: no syntax errors, no undefined names, no unused imports flagged (or note them).

- [ ] **Step 3: Smoke run**

Run: `python scripts/train.py --config configs/cifar10_resnet20.yaml`
Expected: trains, logs start, results.csv + config.json appear under `resnet_ablation/checkpoints/`.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "chore: final verification pass"
```

---

## Self-Review Notes

- Coverage: all spec sections map to a task (Section 1 → Tasks 1-4; Section 2 → Tasks 2,5 + conftest; Section 3 → Task 7; Section 4 → Tasks 1,6).
- `tests/test_augment.py` and `tests/test_blocks.py` referenced in the spec were folded into Task 2/3's model tests; ensure they are created (add a dedicated Step: create `tests/test_augment.py`, `tests/test_blocks.py`) — see note below.
- Type consistency: `topk_correct((1,5))` used consistently; `Trainer.validate(topk=)` matches `evaluate(topk=)`.

Task 5 covers the trainer E2E; `test_augment`/`test_blocks` unit suites are added as Task 5a and Task 5b below.