"""数据集处理模块。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


@dataclass
class EncodedSample:
    """单条样本编码结果。"""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def load_raw_records(dataset_path: Path) -> list[dict[str, Any]]:
    """读取原始 JSON 数据集。"""
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"数据集格式不正确，期望顶层为 list，实际为: {type(records).__name__}")

    return records


def build_user_prompt(instruction: str, user_input: str) -> str:
    """将 instruction 与 input 组合成用户输入。"""
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        return f"{instruction}\n\n补充信息：\n{user_input}"
    return instruction


def encode_conversation_sample(
    record: dict[str, Any],
    tokenizer: Any,
    system_prompt: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    cutoff_len: int,
) -> EncodedSample:
    """把单条记录编码为监督微调样本。

    仅对 assistant 回复部分计算 loss。
    """
    instruction = str(record.get(instruction_field, "") or "")
    user_input = str(record.get(input_field, "") or "")
    assistant_output = str(record.get(output_field, "") or "")

    user_prompt = build_user_prompt(instruction=instruction, user_input=user_input)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_output},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=cutoff_len,
    )
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=cutoff_len,
    )

    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])
    prompt_length = min(len(prompt_tokens["input_ids"]), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    return EncodedSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def build_dataset_dict(
    records: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    instruction_field: str,
    input_field: str,
    output_field: str,
    cutoff_len: int,
    val_split_ratio: float,
) -> DatasetDict:
    """将原始样本编码成 DatasetDict。"""
    encoded_rows = []
    for record in records:
        encoded = encode_conversation_sample(
            record=record,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            instruction_field=instruction_field,
            input_field=input_field,
            output_field=output_field,
            cutoff_len=cutoff_len,
        )
        encoded_rows.append(
            {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
                "labels": encoded.labels,
            }
        )

    full_dataset = Dataset.from_list(encoded_rows)
    if val_split_ratio and 0 < val_split_ratio < 1:
        split = full_dataset.train_test_split(test_size=val_split_ratio, seed=42)
        return DatasetDict(train=split["train"], validation=split["test"])

    return DatasetDict(train=full_dataset)

