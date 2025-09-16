from __future__ import annotations
from loguru import logger
import sys
from pathlib import Path


def setup_logger(save_dir: str) -> None:
    """配置 Loguru 日志输出到控制台与文件。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")
    logger.add(Path(save_dir) / "train.log", level="INFO", rotation="10 MB", enqueue=True)