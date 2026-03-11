#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 使用系统 Python 启动服务: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] 未找到 $PYTHON_BIN，请确认系统已安装 Python 3.12.4"
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

get_listen_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NF' | sort -u
    return 0
  fi

  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | awk 'NF' | sort -u
    return 0
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -lntp "sport = :$port" 2>/dev/null \
      | awk -F 'pid=' 'NR>1 && NF>1 { split($2, a, ","); print a[1] }' \
      | awk 'NF' | sort -u
    return 0
  fi

  echo "[WARN] 无法检测端口 $port 的占用（缺少 lsof/fuser/ss）"
  return 0
}

ensure_port_free() {
  local port="$1"
  local pids=()
  local remaining_pids=()
  local final_pids=()

  mapfile -t pids < <(get_listen_pids "$port" || true)
  if [ "${#pids[@]}" -eq 0 ]; then
    echo "[INFO] 端口 $port 空闲。"
    return 0
  fi

  echo "[WARN] 端口 $port 被占用，准备终止进程: ${pids[*]}"
  for pid in "${pids[@]}"; do
    [ -n "$pid" ] || continue
    ps -fp "$pid" || true
    kill -TERM "$pid" 2>/dev/null || true
  done

  sleep 2
  mapfile -t remaining_pids < <(get_listen_pids "$port" || true)
  if [ "${#remaining_pids[@]}" -gt 0 ]; then
    echo "[WARN] 端口 $port 仍被占用，发送 SIGKILL: ${remaining_pids[*]}"
    for pid in "${remaining_pids[@]}"; do
      [ -n "$pid" ] || continue
      kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1
  fi

  mapfile -t final_pids < <(get_listen_pids "$port" || true)
  if [ "${#final_pids[@]}" -gt 0 ]; then
    echo "[ERROR] 端口 $port 释放失败，仍被占用: ${final_pids[*]}"
    return 1
  fi

  echo "[INFO] 端口 $port 已释放。"
}

"$PYTHON_BIN" - <<'PY'
import os
import shutil
import sys
from pathlib import Path

custom_bin = os.environ.get("OPENWEBUI_BIN")
python_bin_dir = Path(sys.executable).resolve().parent
python_sidecar_bin = python_bin_dir / "open-webui"
resolved_bin = custom_bin or (str(python_sidecar_bin) if python_sidecar_bin.exists() else shutil.which("open-webui"))

if not resolved_bin:
    raise SystemExit(
        "[ERROR] 未找到 open-webui 可执行文件。请先执行 `bash shell/install_serve_env.sh`，"
        "或设置 OPENWEBUI_BIN=/绝对路径/open-webui"
    )

print(f"[INFO] 检测到 open-webui 可执行文件: {resolved_bin}")
PY

read -r VLLM_PORT WEBUI_PORT < <("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/deploy_config.yaml").read_text(encoding="utf-8"))
print(config["server"]["vllm_port"], config["server"]["webui_port"])
PY
)

echo "[INFO] 启动前检查端口占用并自动释放: vLLM=$VLLM_PORT, OpenWebUI=$WEBUI_PORT"
ensure_port_free "$VLLM_PORT"
if [ "$WEBUI_PORT" != "$VLLM_PORT" ]; then
  ensure_port_free "$WEBUI_PORT"
fi

"$PYTHON_BIN" scripts/launch_vllm.py --config configs/deploy_config.yaml
sleep 5
"$PYTHON_BIN" scripts/launch_openwebui.py --config configs/deploy_config.yaml
"$PYTHON_BIN" scripts/healthcheck.py --config configs/deploy_config.yaml --timeout 300 --interval 5

echo "[INFO] 部署完成。"
echo "[INFO] Open WebUI 地址: http://<server-ip>:3000"
echo "[INFO] vLLM API 地址: http://<server-ip>:8000/v1"
