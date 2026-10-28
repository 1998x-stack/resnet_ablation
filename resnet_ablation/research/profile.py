"""Transparent MAC accounting and intermediate-shape report.

MACs cover Conv2d and Linear multiply-accumulate operations only (not BN,
activation, elementwise additions, pooling, softmax, etc). MACs are NOT
identical to a consistent FLOP counting convention across papers/frameworks.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict

import torch
from torch import nn
from resnet_ablation.config import Config
from resnet_ablation.models.factory import build_model


def analyze(model: nn.Module, input_shape: tuple[int, int, int, int]) -> dict:
    if input_shape[0] != 1:
        raise ValueError("Use batch=1 for comparable single-image MAC counts")
    counts = defaultdict(int)
    features = {}
    hooks = []
    original_mode = model.training

    def hook(name: str):
        def callback(module: nn.Module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if isinstance(module, nn.Conv2d):
                kh, kw = module.kernel_size
                macs = output.numel() * (module.in_channels // module.groups) * kh * kw
                counts["conv2d"] += int(macs)
                if "downsample" in name:
                    counts["projection"] += int(macs)
            elif isinstance(module, nn.Linear):
                counts["linear"] += int(output.numel() * module.in_features)
            prefix = name.split(".")[0]
            if prefix in {"stem", "layer1", "layer2", "layer3", "layer4", "conv1", "avgpool"}:
                features[name] = list(output.shape)
        return callback

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook(name)))
    try:
        model.eval()
        # On the model's current device, with no dataset access.
        device = next(model.parameters()).device
        with torch.inference_mode():
            output = model(torch.zeros(input_shape, device=device))
    finally:
        for handle in hooks:
            handle.remove()
        model.train(original_mode)
    total = int(counts["conv2d"] + counts["linear"])
    return {
        "input_shape": list(input_shape), "output_shape": list(output.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "conv2d_macs": int(counts["conv2d"]), "linear_macs": int(counts["linear"]),
        "projection_macs": int(counts["projection"]),
        "total_macs": total,
        "macs_g": total / 1e9,
        "scope": "Conv2d+Linear MACs only; exclude BN, ReLU, pool and additions",
        "feature_shapes": features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    model = build_model(cfg)
    size = args.image_size or (32 if cfg.model.arch.endswith("_cifar") else 224)
    print(json.dumps(analyze(model, (1, 3, size, size)), indent=2))


if __name__ == "__main__":
    main()
