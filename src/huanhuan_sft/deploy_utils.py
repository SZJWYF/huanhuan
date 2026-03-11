"""部署辅助工具。"""

from __future__ import annotations

import os
import signal
import socket
import time
from pathlib import Path
from typing import Any

import requests


def normalize_connect_host(host: str) -> str:
    """将 0.0.0.0 之类的监听地址转换为可用于本机探测的地址。"""
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def wait_for_http_ready(url: str, timeout_seconds: int, interval_seconds: int, logger: Any) -> bool:
    """轮询 HTTP 服务是否已就绪。"""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                logger.info("服务已就绪: %s", url)
                return True
        except requests.RequestException:
            pass

        logger.info("等待服务启动中: %s", url)
        time.sleep(interval_seconds)

    logger.error("服务在规定时间内未就绪: %s", url)
    return False


def write_pid_file(pid_path: Path, pid: int) -> None:
    """写入 PID 文件。"""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")


def read_pid_file(pid_path: Path) -> int | None:
    """读取 PID 文件。"""
    if not pid_path.exists():
        return None
    text = pid_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def stop_process(pid: int, logger: Any) -> bool:
    """优雅停止进程。"""
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("已发送 SIGTERM 给进程 %s", pid)
        return True
    except ProcessLookupError:
        logger.warning("进程 %s 不存在，可能已退出。", pid)
        return False


def is_port_open(host: str, port: int) -> bool:
    """检查端口是否已被使用。"""
    host = normalize_connect_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0
