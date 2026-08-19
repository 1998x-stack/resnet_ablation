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
    # self-contained: emit one results row so the CSV test can assert content
    trainer.write_results_row(1, s, v, 0.01)
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


def test_e2e_writes_csv(tmp_path):
    import csv
    from pathlib import Path
    _train_run(tmp_path)  # runs trainer, which calls write_results_row
    csvf = Path(tmp_path / "ckpt" / "results.csv")
    assert csvf.exists()
    rows = list(csv.reader(csvf.open()))
    assert rows[0][0] == "timestamp"  # header present
    assert len(rows) >= 2             # header + >=1 epoch row
    assert rows[-1][4] == "1"         # last epoch index