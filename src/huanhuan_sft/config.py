"""配置加载与路径标准化工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoadedConfig:
    """封装配置内容与配置文件路径。"""

    raw: dict[str, Any]
    config_path: Path

    @property
    def project_root(self) -> Path:
        return self.config_path.parent.parent.resolve()

    def resolve_path(self, value: str) -> Path:
        """把配置里的相对路径统一解析到仓库根目录。"""
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def load_yaml_config(config_path: str | Path) -> LoadedConfig:
    """读取 YAML 配置文件。"""
    resolved = Path(config_path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return LoadedConfig(raw=raw, config_path=resolved)

