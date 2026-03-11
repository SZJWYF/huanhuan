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

"$PYTHON_BIN" scripts/launch_vllm.py --config configs/deploy_config.yaml
sleep 5
"$PYTHON_BIN" scripts/launch_openwebui.py --config configs/deploy_config.yaml
"$PYTHON_BIN" scripts/healthcheck.py --config configs/deploy_config.yaml --timeout 300 --interval 5

echo "[INFO] 部署完成。"
echo "[INFO] Open WebUI 地址: http://<server-ip>:3000"
echo "[INFO] vLLM API 地址: http://<server-ip>:8000/v1"
