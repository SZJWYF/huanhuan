"""与 ModelScope 交互的工具函数。"""

from __future__ import annotations

from pathlib import Path

from modelscope import snapshot_download


def download_model_from_modelscope(model_id: str, target_dir: Path, revision: str | None = None) -> Path:
    """从 ModelScope 下载模型到指定目录。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        model_id=model_id,
        local_dir=str(target_dir),
        revision=revision,
    )
    return Path(downloaded_path).resolve()

