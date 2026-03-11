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


def resolve_model_path(model_cfg: dict, resolve_path_func) -> Path:
    """优先使用本地模型目录，不存在时再回退到 ModelScope 下载目录。

    参数说明：
    - model_cfg: `train_config.yaml` 中的 model 配置段
    - resolve_path_func: 配置对象自带的路径解析函数
    """
    local_model_path = model_cfg.get("local_model_path")
    if local_model_path:
        candidate = Path(local_model_path)
        if not candidate.is_absolute():
            candidate = resolve_path_func(local_model_path)
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    return download_model_from_modelscope(
        model_id=model_cfg["model_id"],
        target_dir=resolve_path_func(model_cfg["local_dir"]),
    )
