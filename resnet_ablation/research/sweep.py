"""One-factor, paired-seed ablation plans; dry-run is the default.

python -m resnet_ablation.research.sweep --manifest configs/paper/cifar10_matrix.yaml
python -m resnet_ablation.research.sweep --manifest configs/paper/cifar10_matrix.yaml --execute
"""
from __future__ import annotations
import argparse
import copy
import json
import re
from pathlib import Path

import yaml
from resnet_ablation.config import Config
from .runner import run


ALLOWED_CHANGES = {"model.arch", "model.residual", "model.shortcut",
                   "model.bottleneck_stride", "model.attention", "model.drop_path_rate",
                   "model.stem", "model.downsample", "model.width_mult"}


def build_plan(manifest_path: str, root: str) -> list[dict]:
    spec = yaml.safe_load(Path(manifest_path).read_text(encoding="utf8"))
    if not isinstance(spec, dict) or not all(k in spec for k in ("base_config", "variants", "seeds")):
        raise ValueError("manifest requires base_config, variants and seeds")
    base_path = Path(spec["base_config"])
    if not base_path.exists():
        base_path = Path(manifest_path).parent / spec["base_config"]
    cfg = Config.from_yaml(str(base_path))
    base = cfg.as_dict()
    variants = spec["variants"]
    seeds = spec["seeds"]
    if not variants or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("variants and unique seeds are required")
    out = Path(root)
    plan = []
    names = set()
    for var in variants:
        name = var["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("variant ids must be unique filesystem-safe strings")
        names.add(name)
        overrides = var.get("overrides", {})
        changed = set()
        resolved = copy.deepcopy(base)
        for group, values in overrides.items():
            if not isinstance(values, dict) or group not in resolved:
                raise ValueError(f"invalid override group {group}")
            for key, value in values.items():
                dotted = f"{group}.{key}"
                if dotted not in ALLOWED_CHANGES:
                    raise ValueError(f"field {dotted} is not an allowed single-factor architecture ablation")
                resolved[group][key] = value
                if base[group][key] != value:
                    changed.add(dotted)
        if not changed and len(variants) > 1 and name != "baseline":
            raise ValueError(f"variant {name} duplicates the baseline")
        expected = set(spec.get("allowed_factors", []))
        if not changed <= expected:
            raise ValueError(f"{name} changes factors {sorted(changed - expected)} outside experiment group")
        for seed in seeds:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("seeds must be nonnegative integers")
            obj = copy.deepcopy(resolved)
            obj["train"]["seed"] = seed
            destination = out / spec.get("name", "matrix") / name / f"seed_{seed}"
            plan.append({"variant": name, "seed": seed, "factors_changed": sorted(changed),
                         "config": obj, "config_file": str(destination / "resolved.yaml"),
                         "run_dir": str(destination), "schedule": spec.get("schedule", "multistep_epoch"),
                         "steps": spec.get("steps"), "split_seed": spec.get("split_seed", 2026)})
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="runs/paper")
    parser.add_argument("--execute", action="store_true", help="download/train; otherwise dry-run only")
    args = parser.parse_args()
    plan = build_plan(args.manifest, args.out)
    print(json.dumps([{k: v for k, v in row.items() if k != "config"} for row in plan], indent=2))
    if not args.execute:
        print(f"DRY RUN: {len(plan)} planned jobs. No training started.")
        return
    for row in plan:
        config_path = Path(row["config_file"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(row["config"], sort_keys=False), encoding="utf8")
        # runner refuses nonempty dirs on a fresh run: configs live in a sibling
        # directory instead of the run directory.
        run(row["config_file"], row["run_dir"], row["schedule"], row["steps"],
            seed=None, split_seed=row["split_seed"])


if __name__ == "__main__":
    main()
