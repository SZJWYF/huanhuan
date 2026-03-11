"""停止 vLLM 与 Open WebUI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.deploy_utils import read_pid_file, stop_process
from huanhuan_sft.logging_utils import build_log_path, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="停止部署服务")
    parser.add_argument("--config", default="configs/deploy_config.yaml", help="部署配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path(config["paths"]["log_dir"])
    log_file = build_log_path(log_dir, "stop")
    logger = setup_logger("stop_services", log_file)

    pid_dir = loaded.resolve_path(config["paths"]["pid_dir"])
    for service_name in ("openwebui", "vllm"):
        pid_file = pid_dir / f"{service_name}.pid"
        pid = read_pid_file(pid_file)
        if pid is None:
            logger.warning("未找到 %s 的 PID 文件: %s", service_name, pid_file)
            continue
        stop_process(pid, logger)
        pid_file.unlink(missing_ok=True)
        logger.info("%s 停止流程已处理完成。", service_name)


if __name__ == "__main__":
    main()
