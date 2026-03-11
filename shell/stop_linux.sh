#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/stop_entry_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

source .venv-serve/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python scripts/stop_services.py --config configs/deploy_config.yaml

echo "[INFO] 停止脚本执行完成。"
