"""Research-grade, provenance-aware CSV/PNG outputs from actual run logs.

No benchmark number is estimated, filled, or inferred when a run is missing.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_events(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    events = [json.loads(s) for s in (directory / "metrics.jsonl").read_text(encoding="utf8").splitlines() if s]
    return manifest, events


def matched_factors(a: dict, b: dict, allowed: set[str]) -> list[str]:
    """Reject uncontrolled changes even if the plotted result looks impressive."""
    changes = []
    for group in ("model", "optim", "train", "data"):
        left, right = a["config"][group], b["config"][group]
        for field in set(left) | set(right):
            if group == "train" and field in {"seed", "out_dir", "tb_dir", "resume"}:
                continue
            if left.get(field) != right.get(field):
                changes.append(f"{group}.{field}")
    if a["split_sha256"] != b["split_sha256"]:
        changes.append("split_sha256")
    if a["schedule"] != b["schedule"] or a["target_steps"] != b["target_steps"]:
        changes.append("training_schedule")
    if set(changes) - allowed:
        raise ValueError(f"uncontrolled comparison differences: {sorted(set(changes) - allowed)}")
    return sorted(changes)


def summarize(directories: list[str], output_dir: str, allowed: set[str]) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for name in directories:
        directory = Path(name)
        manifest, events = read_events(directory)
        valid = [e for e in events if e["kind"] == "val_epoch"]
        if not valid:
            raise ValueError(f"no validation events: {directory}")
        final = valid[-1]
        best = max(e["acc1"] for e in valid)
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        runs.append({"directory": name, "label": directory.parent.name,
                     "seed": manifest["config"]["train"]["seed"],
                     "arch": manifest["config"]["model"]["arch"],
                     "residual": manifest["config"]["model"]["residual"],
                     "shortcut": manifest["config"]["model"]["shortcut"],
                     "params": manifest["parameters"], "git_sha": manifest["git_sha"],
                     "split_sha256": manifest["split_sha256"],
                     "completed_steps": status.get("completed_steps"),
                     "is_complete": status.get("is_complete", False),
                     "best_val_error1": 1 - best,
                     "last_val_error1": 1 - final["acc1"],
                     "last_val_loss": final["loss"],
                     "events": events, "manifest": manifest})
    if runs:
        ref = runs[0]["manifest"]
        for r in runs[1:]:
            matched_factors(ref, r["manifest"], allowed)
    grouped: dict[str, list] = defaultdict(list)
    for row in runs:
        grouped[row["label"]].append(row)
    summary = []
    for label, items in sorted(grouped.items()):
        if len(set(x["seed"] for x in items)) != len(items):
            raise ValueError(f"duplicate seed within group {label}")
        values = [r["best_val_error1"] for r in items if r["is_complete"]]
        # An incomplete run is never silently included in the final statistics.
        summary.append({"label": label, "completed_n": len(values),
                        "total_n": len(items),
                        "mean_best_val_error1": statistics.mean(values) if values else None,
                        "std_best_val_error1": statistics.stdev(values) if len(values) > 1 else None,
                        "seeds": [r["seed"] for r in items]})
    with (out / "runs.csv").open("w", newline="", encoding="utf8") as handle:
        keys = ["directory","label","seed","arch","residual","shortcut","params",
                "git_sha","split_sha256","completed_steps","is_complete",
                "best_val_error1","last_val_error1","last_val_loss"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k:r[k] for k in keys} for r in runs])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    return {"runs": runs, "summary": summary, "output": str(out)}


def plot(result: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install optional plotting dependency: pip install matplotlib") from exc
    out = Path(result["output"])
    figures = [
        ("training_loss", "train_epoch", "loss", "Training loss", "Loss"),
        ("validation_error", "val_epoch", "acc1", "Validation error (1 - accuracy)", "Error"),
        ("learning_rate", "train_step", "lr_used", "Learning rate actually used", "LR"),
        ("gradient_global", "train_step", "grad/global_l2", "Gradient L2", "Gradient norm"),
    ]
    for filename, kind, key, title, ylabel in figures:
        fig, ax = plt.subplots(figsize=(9, 5))
        found = 0
        for run in result["runs"]:
            points = [e for e in run["events"] if e["kind"] == kind and key in e]
            if not points:
                continue
            x = [e["step"] for e in points]
            y = [1 - e[key] if filename == "validation_error" else e[key] for e in points]
            ax.plot(x, y, label=f"{run['label']} / seed={run['seed']}", alpha=0.8)
            found += 1
        ax.set(title=title, xlabel="Completed optimizer steps", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        if found:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)
    groups = [x for x in result["summary"] if x["mean_best_val_error1"] is not None]
    if groups:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [x["label"] for x in groups]
        means = [x["mean_best_val_error1"] * 100 for x in groups]
        std = [x["std_best_val_error1"] * 100 if x["std_best_val_error1"] is not None else 0 for x in groups]
        ax.bar(labels, means, yerr=std, capsize=4)
        ax.set(title="Completed-run best validation error", ylabel="Error (%)")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(out / "multi_seed_comparison.png", dpi=160)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="reports/paper")
    ap.add_argument("--allow-factor", action="append", required=True,
                    help="each intentionally varied field: e.g. model.residual")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    result = summarize(args.run_dirs, args.out, set(args.allow_factor))
    if not args.no_plots:
        plot(result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
