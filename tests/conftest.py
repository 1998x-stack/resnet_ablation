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