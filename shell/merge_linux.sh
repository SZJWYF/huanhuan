#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/merge_entry_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 使用 .venv-train 环境执行 LoRA 合并"
source .venv-train/bin/activate

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python scripts/merge_lora.py --config configs/train_config.yaml

echo "[INFO] 合并脚本执行完成。"

