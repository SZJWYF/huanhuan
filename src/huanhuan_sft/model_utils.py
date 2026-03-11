"""模型与 tokenizer 构建模块。"""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """将配置中的字符串映射为 torch.dtype。"""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 torch dtype: {dtype_name}")
    return mapping[dtype_name]


def build_tokenizer(model_path: str, trust_remote_code: bool) -> Any:
    """加载 tokenizer，并确保 pad_token 一定存在。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_bnb_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    """根据配置决定是否启用 4bit 量化。"""
    if not model_cfg.get("load_in_4bit", False):
        return None

    compute_dtype = resolve_torch_dtype(model_cfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=model_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_training_model(model_path: str, model_cfg: dict[str, Any], lora_cfg: dict[str, Any]) -> Any:
    """构建训练用 LoRA 模型。"""
    torch_dtype = resolve_torch_dtype(model_cfg["torch_dtype"])
    quantization_config = build_bnb_config(model_cfg)

    attn_impl = model_cfg.get("attn_implementation", "auto")
    if attn_impl == "auto":
        attn_impl = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_cfg["trust_remote_code"],
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_merge_model(model_path: str, trust_remote_code: bool, dtype_name: str) -> Any:
    """构建 LoRA 合并阶段使用的基座模型。"""
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=resolve_torch_dtype(dtype_name),
        device_map="auto",
    )

