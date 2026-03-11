"""训练辅助工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


@dataclass
class TrainingPaths:
    """保存单次训练相关路径。"""

    run_dir: Path
    adapter_dir: Path
    trainer_dir: Path
    summary_file: Path


class SupervisedDataCollator:
    """对已编码样本执行 padding，并屏蔽 padding 的 loss。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            pad_len = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class FileLoggingCallback(TrainerCallback):
    """让训练开始和结束时写出更直观的日志。"""

    def __init__(self, logger: Any):
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self.logger.info("训练开始，总训练步数: %s", state.max_steps)

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self.logger.info("训练结束，最终 global_step: %s", state.global_step)


def prepare_training_paths(output_root: Path, run_name: str) -> TrainingPaths:
    """生成标准训练输出路径。"""
    run_dir = output_root / run_name
    adapter_dir = run_dir / "adapter"
    trainer_dir = run_dir / "trainer_state"
    summary_file = run_dir / "training_summary.json"

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        trainer_dir=trainer_dir,
        summary_file=summary_file,
    )


def save_training_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    """保存训练摘要信息。"""
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

