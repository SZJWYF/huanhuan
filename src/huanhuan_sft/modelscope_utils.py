"""与 ModelScope 交互的工具函数。"""

from __future__ import annotations

from pathlib import Path

from modelscope import snapshot_download


def _looks_like_model_dir(path: Path) -> bool:
    """粗略判断一个目录是否像 Hugging Face/ModelScope 模型根目录。"""
    expected_files = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    if not path.is_dir():
        return False
    existing_names = {item.name for item in path.iterdir()}
    return bool(expected_files & existing_names)


def _find_nested_model_dir(root: Path) -> Path | None:
    """如果传入的是父目录，尝试自动找到其中真正的模型子目录。"""
    if not root.is_dir():
        return None

    children = sorted([item for item in root.iterdir() if item.is_dir()])
    matched = [item for item in children if _looks_like_model_dir(item)]
    if len(matched) == 1:
        return matched[0]
    return None


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
            if _looks_like_model_dir(candidate):
                return candidate
            nested = _find_nested_model_dir(candidate)
            if nested is not None:
                return nested.resolve()
            return candidate

    return download_model_from_modelscope(
        model_id=model_cfg["model_id"],
        target_dir=resolve_path_func(model_cfg["local_dir"]),
    )
