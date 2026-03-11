"""统一日志模块。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from .constants import TIME_FORMAT


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_log_path(log_dir: Path, prefix: str) -> Path:
    """生成带时间戳的日志文件路径。"""
    ensure_dir(log_dir)
    timestamp = datetime.now().strftime(TIME_FORMAT)
    return log_dir / f"{prefix}_{timestamp}.log"


def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """创建统一格式的日志器。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | line %(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

