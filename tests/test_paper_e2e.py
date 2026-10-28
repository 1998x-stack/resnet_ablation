"""Exercise the factory research runner without downloading or claiming benchmark data."""
from __future__ import annotations
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import TensorDataset
from resnet_ablation.research.factories import DATASETS, DataBundle
from resnet_ablation.research.runner import final_test, run


def test_factory_training_checkpoint_resume_and_final_test(tmp_path, monkeypatch):
    x = torch.arange(12 * 3 * 32 * 32, dtype=torch.float32).reshape(12, 3, 32, 32) / 20000
    y = torch.arange(12) % 10
    seen = []

    def fake_dataset(root, split_seed, val_size=5000, final_test=False):
        seen.append(final_test)
        return DataBundle(
            TensorDataset(x[:8], y[:8]),
            TensorDataset(x[8:10], y[8:10]),
            TensorDataset(x[10:], y[10:]) if final_test else None,
            "fixed-synthetic-disjoint", split_seed, "synthetic-no-normalization", 8, 2,
            2 if final_test else 0,
        )

    monkeypatch.setitem(DATASETS._builders, "cifar10", fake_dataset)
    config = {
        "model": {"arch": "resnet20_cifar", "num_classes": 10, "shortcut": "A",
                  "residual": True},
        "data": {"name": "cifar10", "root": str(tmp_path / "data"), "aug": "standard"},
        "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.9,
                  "weight_decay": 0.0, "nesterov": False, "warmup_epochs": 0},
        "train": {"epochs": 2, "batch_size": 4, "num_workers": 0,
                  "amp": False, "clip_grad_norm": None,
                  "log_interval": 1, "val_interval": 1, "seed": 7},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    out = tmp_path / "real_run"
    first = run(str(config_path), str(out), "cosine_step", steps=4, device_str="cpu")
    assert first["is_complete"] and first["completed_steps"] == 4
    assert seen == [False]
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "train_step"]) == 4
    assert len([e for e in events if e["kind"] == "val_epoch"]) == 2
    assert len({e["step"] for e in events if e["kind"] == "train_step"}) == 4
    assert all("lr_used" in e for e in events if e["kind"] == "train_step")
    state = torch.load(out / "last.pt", weights_only=False, map_location="cpu")
    assert state["next_epoch"] == 2 and state["global_step"] == 4
    assert state["rng"]["torch"] is not None
    count = len(events)
    restored = run(str(config_path), str(out), "cosine_step", steps=4,
                   resume=True, device_str="cpu")
    assert restored["completed_steps"] == 4
    assert len((out / "metrics.jsonl").read_text().splitlines()) == count
    score = final_test(str(config_path), str(out), device_str="cpu")
    assert seen == [False, False, True]
    assert score["kind"] == "final_test" and 0 <= score["acc1"] <= 1
    assert (out / "final_test.json").exists()
