#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG_FILE="logs/deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] 使用 .venv-serve 环境启动服务"
source .venv-serve/bin/activate

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

python scripts/launch_vllm.py --config configs/deploy_config.yaml
sleep 5
python scripts/launch_openwebui.py --config configs/deploy_config.yaml
python scripts/healthcheck.py --config configs/deploy_config.yaml --timeout 300 --interval 5

echo "[INFO] 部署完成。"
echo "[INFO] Open WebUI 地址: http://<server-ip>:3000"
echo "[INFO] vLLM API 地址: http://<server-ip>:8000/v1"

