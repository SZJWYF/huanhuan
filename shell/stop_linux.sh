#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/stop_entry_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] 未找到 $PYTHON_BIN，请确认系统已安装 Python 3.12.4"
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

"$PYTHON_BIN" scripts/stop_services.py --config configs/deploy_config.yaml

echo "[INFO] 停止脚本执行完成。"
