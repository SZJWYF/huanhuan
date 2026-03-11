#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/train_entry_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 使用系统 Python 执行训练: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] 未找到 $PYTHON_BIN，请确认系统已安装 Python 3.12.4"
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import torch
print(f"[INFO] torch version: {torch.__version__}")
print(f"[INFO] cuda available: {torch.cuda.is_available()}")
print(f"[INFO] torch cuda version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"[INFO] gpu[0]: {torch.cuda.get_device_name(0)}")
PY

"$PYTHON_BIN" scripts/train.py --config configs/train_config.yaml

echo "[INFO] 训练脚本执行完成。"
