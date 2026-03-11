"""从 ModelScope 下载模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.logging_utils import build_log_path, ensure_dir, setup_logger
from huanhuan_sft.modelscope_utils import download_model_from_modelscope, resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 ModelScope 下载基座模型")
    parser.add_argument("--config", default="configs/train_config.yaml", help="训练配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)

    log_dir = loaded.resolve_path("logs")
    log_file = build_log_path(log_dir, "download_model")
    logger = setup_logger("download_model", log_file)

    model_cfg = loaded.raw["model"]
    explicit_local_path = model_cfg.get("local_model_path")
    if explicit_local_path:
        logger.info("配置中声明的本地模型目录: %s", explicit_local_path)

    target_dir = loaded.resolve_path(model_cfg["local_dir"])
    ensure_dir(target_dir)

    if explicit_local_path and Path(explicit_local_path).exists():
        local_path = resolve_model_path(model_cfg, loaded.resolve_path)
        logger.info("检测到本地模型已存在，跳过下载，直接使用: %s", local_path)
        return

    logger.info("开始从 ModelScope 下载模型: %s", model_cfg["model_id"])
    logger.info("目标目录: %s", target_dir)

    local_path = download_model_from_modelscope(
        model_id=model_cfg["model_id"],
        target_dir=target_dir,
    )
    logger.info("模型下载完成，本地路径: %s", local_path)


if __name__ == "__main__":
    main()
