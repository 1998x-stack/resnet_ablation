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


def test_topk_correct_default_top1():
    logits = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.9, 0.1, 0.0],
    ])
    targets = torch.tensor([1, 0])
    (c1,) = topk_correct(logits, targets)
    assert c1.item() == 2


def test_topk_correct_batch_dim():
    logits = torch.randn(16, 100)
    targets = torch.randint(0, 100, (16,))
    c1, c5 = topk_correct(logits, targets, (1, 5))
    assert c1.shape == torch.Size([])
    assert c5.shape == torch.Size([])