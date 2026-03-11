"""检查 vLLM 与 Open WebUI 是否可访问。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.deploy_utils import wait_for_http_ready
from huanhuan_sft.logging_utils import build_log_path, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="服务健康检查")
    parser.add_argument("--config", default="configs/deploy_config.yaml", help="部署配置文件路径")
    parser.add_argument("--timeout", type=int, default=180, help="最大等待秒数")
    parser.add_argument("--interval", type=int, default=5, help="轮询间隔秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path(config["paths"]["log_dir"])
    log_file = build_log_path(log_dir, "healthcheck")
    logger = setup_logger("healthcheck", log_file)

    server_cfg = config["server"]
    vllm_url = f"http://127.0.0.1:{server_cfg['vllm_port']}/v1/models"
    webui_url = f"http://127.0.0.1:{server_cfg['webui_port']}/"

    vllm_ok = wait_for_http_ready(vllm_url, args.timeout, args.interval, logger)
    webui_ok = wait_for_http_ready(webui_url, args.timeout, args.interval, logger)

    if not (vllm_ok and webui_ok):
        raise SystemExit(1)

    logger.info("所有服务健康检查通过。")


if __name__ == "__main__":
    main()
