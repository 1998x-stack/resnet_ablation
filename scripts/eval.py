from __future__ import annotations
import argparse, yaml, torch
from loguru import logger
from resnet_ablation.config import Config
from resnet_ablation.logger import setup_logger
from resnet_ablation.utils import set_seed, get_device
from resnet_ablation.data import build_dataloaders
from resnet_ablation.engine.evaluator import evaluate
from resnet_ablation.models.factory import build_model


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logger(cfg.train.out_dir)
    set_seed(cfg.train.seed)
    device = get_device()

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {args.ckpt}")

    res = evaluate(model, val_loader)
    logger.info(f"Eval Acc@1: {res['acc1']*100:.2f}%")

if __name__ == "__main__":
    main()