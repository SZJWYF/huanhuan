"""启动 Open WebUI。"""

from __future__ import annotations

import argparse
import os
import shutil
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
    parser = argparse.ArgumentParser(description="启动 Open WebUI")
    parser.add_argument("--config", default="configs/deploy_config.yaml", help="部署配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path(config["paths"]["log_dir"])
    log_file = build_log_path(log_dir, "openwebui")
    logger = setup_logger("launch_openwebui", log_file)

    server_cfg = config["server"]
    path_cfg = config["paths"]

    data_dir = loaded.resolve_path(path_cfg["data_dir"])
    pid_dir = loaded.resolve_path(path_cfg["pid_dir"])
    env_file = loaded.resolve_path(path_cfg["openwebui_env_file"])
    pid_file = pid_dir / "openwebui.pid"

    data_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    env_file.parent.mkdir(parents=True, exist_ok=True)

    if is_port_open(server_cfg["webui_host"], int(server_cfg["webui_port"])):
        raise RuntimeError(f"Open WebUI 端口已被占用: {server_cfg['webui_host']}:{server_cfg['webui_port']}")

    env_map = {
        "DATA_DIR": str(data_dir),
        "HOST": server_cfg["webui_host"],
        "PORT": str(server_cfg["webui_port"]),
        "OPENAI_API_BASE_URLS": f"http://127.0.0.1:{server_cfg['vllm_port']}/v1",
        "OPENAI_API_KEYS": server_cfg["api_key"],
        "OPENAI_API_KEY": server_cfg["api_key"],
        "WEBUI_AUTH": "False",
    }
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in env_map.items()) + "\n",
        encoding="utf-8",
    )
    logger.info("Open WebUI 环境变量文件已写入: %s", env_file)

    custom_bin = os.environ.get("OPENWEBUI_BIN")
    python_bin_dir = Path(sys.executable).resolve().parent
    python_sidecar_bin = python_bin_dir / "open-webui"
    resolved_bin = custom_bin or (str(python_sidecar_bin) if python_sidecar_bin.exists() else shutil.which("open-webui"))
    if not resolved_bin:
        raise RuntimeError(
            "未找到 open-webui 可执行文件。请先执行 `bash shell/install_serve_env.sh`，"
            "或通过环境变量 OPENWEBUI_BIN 指定可执行文件绝对路径。"
        )

    command = [resolved_bin, "serve", "--host", server_cfg["webui_host"], "--port", str(server_cfg["webui_port"])]
    logger.info("将启动 Open WebUI，命令如下: %s", " ".join(command))

    runtime_env = os.environ.copy()
    runtime_env.update(env_map)

    handle = open(log_file, "a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=runtime_env,
    )
    write_pid_file(pid_file, process.pid)
    logger.info("Open WebUI 已启动，PID=%s，日志文件=%s", process.pid, log_file)


if __name__ == "__main__":
    main()
