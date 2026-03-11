#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/install_serve_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 项目根目录: $PROJECT_ROOT"
echo "[INFO] 日志文件: $LOG_FILE"
echo "[INFO] 使用系统 Python: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] 未找到 $PYTHON_BIN，请确认系统已安装 Python 3.12.4"
  exit 1
fi

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install -r requirements/serve.txt
"$PYTHON_BIN" - <<'PY'
import torch
print(f"[INFO] torch version: {torch.__version__}")
print(f"[INFO] cuda available: {torch.cuda.is_available()}")
print(f"[INFO] torch cuda version: {torch.version.cuda}")
PY

echo "[INFO] 部署依赖安装完成，未创建虚拟环境。"
