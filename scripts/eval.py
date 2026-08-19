from __future__ import annotations
import argparse, yaml, json, time, torch
from pathlib import Path
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

    # 转储 resolved config（含运行环境），保证可复现性（与 train.py 一致）
    runtime = {"device": str(device),
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seed": cfg.train.seed}
    full = cfg.as_dict()
    full["runtime"] = runtime
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.train.out_dir) / "config.json").write_text(json.dumps(full, indent=2))

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {args.ckpt}")

    res = evaluate(model, val_loader, (1, 5) if cfg.data.name == "imagenet" else (1,))
    msg = f"Acc@1: {res['acc1']*100:.2f}%"
    if "acc5" in res:
        msg += f" | Acc@5: {res['acc5']*100:.2f}%"
    logger.info(f"Eval {msg}")

if __name__ == "__main__":
    main()