from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import Optional, List, Any, Dict

# Whitelist of valid enum values per nested config. `None` means the field is a
# free-form value (any type is accepted, so no enum check is applied).
_VALID: Dict[str, Dict[str, Any]] = {
    "ModelConfig": {
        "arch": {
            "resnet20_cifar", "resnet56_cifar", "resnet110_cifar",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        },
        "shortcut": {"A", "B", "C"},
        "stem": {"cifar", "imagenet_standard", "imagenet_deep"},
        "downsample": {"proj", "avgproj"},
        "num_classes": None,
        "width_mult": None,
        "drop_path_rate": None,
        "attention": {"none", "se", "eca"},
        "se_ratio": None,
    },
    "OptimConfig": {
        "name": None,
        "lr": None,
        "momentum": None,
        "weight_decay": None,
        "nesterov": None,
        "warmup_epochs": None,
        "sched": {"cosine", "multistep"},
        "milestones": None,
        "gamma": None,
    },
    "TrainConfig": {
        "epochs": None,
        "batch_size": None,
        "num_workers": None,
        "amp": None,
        "clip_grad_norm": None,
        "log_interval": None,
        "val_interval": None,
        "out_dir": None,
        "resume": None,
        "seed": None,
        "tb_dir": None,
        "label_smoothing": None,
    },
    "DataConfig": {
        "name": {"cifar10", "cifar100", "imagenet"},
        "root": None,
        "aug": {"standard", "strong"},
        "mixup_alpha": None,
        "cutmix_alpha": None,
    },
}

# Map dataclass name -> its key within the top-level Config (for validate()).
_GROUP_KEY = {
    "ModelConfig": "model",
    "OptimConfig": "optim",
    "TrainConfig": "train",
    "DataConfig": "data",
}

_TOP_LEVEL_GROUPS = ("model", "optim", "train", "data")


def _check(key: str, value: Any, allowed: Any) -> None:
    """Raise ValueError naming `key` if value is outside its allowed enum set."""
    if allowed is not None and value is not None and value not in allowed:
        raise ValueError(
            f"invalid value '{value}' for '{key}': allowed {sorted(allowed)}"
        )


def _strict_kwargs(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Validate field names and enum values for one nested config dict."""
    allowed_fields = {f.name for f in fields(cls)}
    for key in raw:
        if key not in allowed_fields:
            raise ValueError(f"unknown key '{key}' for {cls.__name__}")
    for key, value in raw.items():
        _check(key, value, _VALID[cls.__name__].get(key))
    return raw


@dataclass
class ModelConfig:
    """Model architecture and build options."""

    arch: str = "resnet20_cifar"  # resnet20/56/110_cifar, resnet18/34/50/101/152
    num_classes: int = 10  # classification head width
    shortcut: str = "A"  # A/B/C (CIFAR BasicBlock); B (projection) for ImageNet
    stem: str = "cifar"  # cifar / imagenet_standard / imagenet_deep
    downsample: str = "proj"  # proj (1x1) / avgproj (ResNet-D avg-pool + 1x1)
    width_mult: float = 1.0  # width scaling factor for the whole network
    drop_path_rate: float = 0.0  # Stochastic Depth max global ratio
    attention: str = "none"  # none / se / eca
    se_ratio: float = 0.25  # SE reduction ratio (used when attention=se)


@dataclass
class OptimConfig:
    """Optimizer and learning-rate schedule options."""

    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    warmup_epochs: int = 5  # linear LR warmup length in epochs
    sched: str = "cosine"  # cosine / multistep
    milestones: List[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1


@dataclass
class TrainConfig:
    """Training loop options (data loading, logging, checkpoints)."""

    epochs: int = 200
    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    clip_grad_norm: Optional[float] = 1.0
    log_interval: int = 50  # log loss every N batches
    val_interval: int = 1  # validate every N epochs
    out_dir: str = "resnet_ablation/checkpoints"
    resume: Optional[str] = None  # checkpoint path to resume from
    seed: int = 42
    tb_dir: str = "runs"
    label_smoothing: float = 0.0  # label smoothing (applies when no Mixup/CutMix)


@dataclass
class DataConfig:
    """Dataset and augmentation options."""

    name: str = "cifar10"  # cifar10 / cifar100 / imagenet
    root: str = "./data"
    aug: str = "standard"  # standard / strong
    mixup_alpha: float = 0.0  # Mixup strength; >0 enables Mixup
    cutmix_alpha: float = 0.0  # CutMix strength; >0 enables CutMix


@dataclass
class Config:
    """Aggregate training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a Config from a YAML file, validating keys and enum values."""
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError("top-level YAML must be a mapping")

        for group in raw:
            if group not in _TOP_LEVEL_GROUPS:
                raise ValueError(f"unknown group '{group}'")

        model_raw = raw.get("model") or {}
        optim_raw = raw.get("optim") or {}
        train_raw = raw.get("train") or {}
        data_raw = raw.get("data") or {}

        for group, value in (("model", model_raw), ("optim", optim_raw),
                             ("train", train_raw), ("data", data_raw)):
            if not isinstance(value, dict):
                raise ValueError(f"group '{group}' must be a mapping")

        cfg = cls(
            model=ModelConfig(**_strict_kwargs(ModelConfig, model_raw)),
            optim=OptimConfig(**_strict_kwargs(OptimConfig, optim_raw)),
            train=TrainConfig(**_strict_kwargs(TrainConfig, train_raw)),
            data=DataConfig(**_strict_kwargs(DataConfig, data_raw)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Re-check the current enum values on this instance (used after edits)."""
        for grp in (ModelConfig, OptimConfig, TrainConfig, DataConfig):
            inst = getattr(self, _GROUP_KEY[grp.__name__])
            known = _VALID[grp.__name__]
            for fld in fields(grp):
                _check(fld.name, getattr(inst, fld.name), known.get(fld.name))

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict, json-serializable via dataclasses.asdict."""
        return {
            "model": asdict(self.model),
            "optim": asdict(self.optim),
            "train": asdict(self.train),
            "data": asdict(self.data),
        }