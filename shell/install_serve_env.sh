#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/install_serve_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 项目根目录: $PROJECT_ROOT"
echo "[INFO] 日志文件: $LOG_FILE"

python3 -m venv .venv-serve
source .venv-serve/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/serve.txt

echo "[INFO] 部署环境安装完成。"

