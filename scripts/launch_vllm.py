"""启动 vLLM 服务。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.deploy_utils import is_port_open, write_pid_file
from huanhuan_sft.logging_utils import build_log_path, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 vLLM OpenAI 兼容服务")
    parser.add_argument("--config", default="configs/deploy_config.yaml", help="部署配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path(config["paths"]["log_dir"])
    log_file = build_log_path(log_dir, "vllm")
    logger = setup_logger("launch_vllm", log_file)

    model_cfg = config["model"]
    server_cfg = config["server"]
    pid_dir = loaded.resolve_path(config["paths"]["pid_dir"])
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "vllm.pid"

    model_dir = loaded.resolve_path(model_cfg["merged_model_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到合并模型目录: {model_dir}")

    if is_port_open(server_cfg["host"], int(server_cfg["vllm_port"])):
        raise RuntimeError(f"端口已被占用，请先释放: {server_cfg['host']}:{server_cfg['vllm_port']}")

    command = [
        "vllm",
        "serve",
        str(model_dir),
        "--host",
        server_cfg["host"],
        "--port",
        str(server_cfg["vllm_port"]),
        "--served-model-name",
        server_cfg["served_model_name"],
        "--dtype",
        model_cfg["dtype"],
        "--max-model-len",
        str(server_cfg["max_model_len"]),
        "--gpu-memory-utilization",
        str(server_cfg["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(server_cfg["max_num_seqs"]),
        "--api-key",
        server_cfg["api_key"],
        "--trust-remote-code",
    ]

    if server_cfg.get("disable_log_requests", True):
        command.append("--disable-log-requests")
    if not server_cfg.get("uvicorn_access_log", False):
        command.append("--disable-uvicorn-access-log")

    logger.info("将启动 vLLM，命令如下: %s", " ".join(command))
    handle = open(log_file, "a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
    )
    write_pid_file(pid_file, process.pid)
    logger.info("vLLM 已启动，PID=%s，日志文件=%s", process.pid, log_file)


if __name__ == "__main__":
    main()

