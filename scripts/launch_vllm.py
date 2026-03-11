"""启动 vLLM 服务。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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
    metadata_file = model_dir / "merge_metadata.json"
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            logger.info(
                "检测到合并元数据: merged_at_utc=%s, base_model_path=%s, adapter_dir=%s, train_global_step=%s",
                metadata.get("merged_at_utc"),
                metadata.get("base_model_path"),
                metadata.get("adapter_dir"),
                metadata.get("train_global_step"),
            )
        except Exception:
            logger.exception("读取合并元数据失败: %s", metadata_file)
    else:
        logger.warning("未找到合并元数据文件: %s（建议先重新执行 merge）", metadata_file)

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
    handle.close()
    write_pid_file(pid_file, process.pid)
    logger.info("vLLM 已启动，PID=%s，日志文件=%s", process.pid, log_file)

    # vLLM 常见失败是进程秒退（环境/显存/模型加载错误），这里提前探测并给出日志尾部。
    time.sleep(3)
    return_code = process.poll()
    if return_code is not None:
        log_tail = ""
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-40:])
        raise RuntimeError(
            f"vLLM 启动后立即退出，返回码={return_code}，日志文件={log_file}\n"
            f"最近日志:\n{log_tail if log_tail else '(日志为空)'}"
        )


if __name__ == "__main__":
    main()
