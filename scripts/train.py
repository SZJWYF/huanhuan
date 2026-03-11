"""训练脚本。"""

from __future__ import annotations

import argparse
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.data_utils import build_dataset_dict, load_raw_records
from huanhuan_sft.logging_utils import build_log_path, setup_logger
from huanhuan_sft.model_utils import build_tokenizer, build_training_model
from huanhuan_sft.modelscope_utils import resolve_model_path
from huanhuan_sft.train_utils import FileLoggingCallback, SupervisedDataCollator, prepare_training_paths, save_training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen LoRA 微调脚本")
    parser.add_argument("--config", default="configs/train_config.yaml", help="训练配置文件路径")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight_cuda_compatibility(logger, model_cfg: dict) -> None:
    """在真正加载模型前做一次 CUDA 兼容性检查。"""
    if not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability(0)
    current_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = set(torch.cuda.get_arch_list())
    logger.info("当前 GPU 计算能力: %s", current_arch)
    logger.info("当前 PyTorch 编译支持的 CUDA 架构: %s", ", ".join(sorted(compiled_arches)))

    if capability[0] >= 12 and model_cfg.get("load_in_4bit", False):
        logger.warning("检测到 Blackwell/SM120 GPU，默认关闭 4bit bitsandbytes 量化加载以避免 no kernel image 错误。")
        model_cfg["load_in_4bit"] = False

    if compiled_arches and current_arch not in compiled_arches:
        raise RuntimeError(
            f"当前 PyTorch 二进制不支持 {current_arch}，但当前 GPU 是 {torch.cuda.get_device_name(0)}。"
            "这不是项目代码问题，而是 PyTorch/CUDA 二进制兼容问题。"
            "请安装支持 Blackwell/RTX 5090 的 PyTorch 构建后再训练。"
        )


def build_training_arguments(paths, project_cfg: dict, training_cfg: dict, has_validation: bool, logger):
    """根据当前 transformers 版本动态构造 TrainingArguments。"""
    signature = inspect.signature(TrainingArguments.__init__)
    supported_params = set(signature.parameters)

    kwargs = {
        "output_dir": str(paths.trainer_dir),
        "run_name": training_cfg["run_name"],
        "num_train_epochs": training_cfg["num_train_epochs"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "warmup_ratio": training_cfg["warmup_ratio"],
        "lr_scheduler_type": training_cfg["lr_scheduler_type"],
        "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": training_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "gradient_checkpointing": training_cfg["gradient_checkpointing"],
        "max_grad_norm": training_cfg["max_grad_norm"],
        "logging_steps": training_cfg["logging_steps"],
        "save_steps": training_cfg["save_steps"],
        "eval_steps": training_cfg["eval_steps"],
        "save_total_limit": training_cfg["save_total_limit"],
        "max_steps": training_cfg["max_steps"],
        "optim": training_cfg["optim"],
        "bf16": training_cfg["bf16"],
        "fp16": training_cfg["fp16"],
        "tf32": training_cfg["tf32"],
        "dataloader_num_workers": training_cfg["dataloader_num_workers"],
        "report_to": training_cfg["report_to"],
        "logging_dir": str(paths.run_dir / "tensorboard"),
        "remove_unused_columns": False,
        "seed": project_cfg["seed"],
    }

    eval_value = "steps" if has_validation else "no"
    if "eval_strategy" in supported_params:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in supported_params:
        kwargs["evaluation_strategy"] = eval_value

    if "save_safetensors" in supported_params:
        kwargs["save_safetensors"] = True
    else:
        logger.warning("当前 transformers 版本不支持 save_safetensors，已自动跳过该参数。")

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    skipped_kwargs = sorted(set(kwargs) - set(filtered_kwargs))
    if skipped_kwargs:
        logger.warning("以下 TrainingArguments 参数当前版本不支持，已自动跳过: %s", ", ".join(skipped_kwargs))

    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path("logs")
    log_file = build_log_path(log_dir, "train")
    logger = setup_logger("train", log_file)

    project_cfg = config["project"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]

    set_seed(project_cfg["seed"])
    logger.info("随机种子已设置为: %s", project_cfg["seed"])

    if torch.cuda.is_available():
        logger.info("检测到 CUDA，可用 GPU 数量: %s", torch.cuda.device_count())
        logger.info("当前 GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("未检测到 CUDA，本项目目标是 Linux + NVIDIA GPU 环境。")

    preflight_cuda_compatibility(logger, model_cfg)

    logger.info("优先检查本地模型目录: %s", model_cfg.get("local_model_path", "<未配置>"))
    local_model_path = resolve_model_path(model_cfg, loaded.resolve_path)
    logger.info("基座模型已就绪: %s", local_model_path)

    try:
        tokenizer = build_tokenizer(
            model_path=str(local_model_path),
            trust_remote_code=model_cfg["trust_remote_code"],
        )
    except Exception:
        logger.exception("tokenizer 加载失败，请重点检查模型目录和 sentencepiece/tiktoken 依赖。")
        raise
    logger.info("tokenizer 加载完成，pad_token_id=%s, eos_token_id=%s", tokenizer.pad_token_id, tokenizer.eos_token_id)

    dataset_path = loaded.resolve_path(data_cfg["train_file"])
    raw_records = load_raw_records(dataset_path)
    logger.info("数据集读取完成，总样本数: %s", len(raw_records))

    if data_cfg.get("shuffle", True):
        random.shuffle(raw_records)
        logger.info("训练记录已随机打乱。")

    cutoff_len = min(int(data_cfg["cutoff_len"]), int(training_cfg["max_seq_length"]))
    logger.info("当前样本最大编码长度: %s", cutoff_len)

    dataset_dict = build_dataset_dict(
        records=raw_records,
        tokenizer=tokenizer,
        system_prompt=data_cfg["system_prompt"],
        instruction_field=data_cfg["instruction_field"],
        input_field=data_cfg["input_field"],
        output_field=data_cfg["output_field"],
        cutoff_len=cutoff_len,
        val_split_ratio=data_cfg["val_split_ratio"],
    )
    logger.info("编码后训练集大小: %s", len(dataset_dict["train"]))
    if "validation" in dataset_dict:
        logger.info("编码后验证集大小: %s", len(dataset_dict["validation"]))

    model = build_training_model(
        model_path=str(local_model_path),
        model_cfg=model_cfg,
        lora_cfg=lora_cfg,
    )

    paths = prepare_training_paths(
        output_root=loaded.resolve_path(training_cfg["output_root"]),
        run_name=training_cfg["run_name"],
    )
    logger.info("训练输出目录: %s", paths.run_dir)

    training_args = build_training_arguments(
        paths=paths,
        project_cfg=project_cfg,
        training_cfg=training_cfg,
        has_validation="validation" in dataset_dict,
        logger=logger,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        data_collator=SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=[FileLoggingCallback(logger)],
    )

    logger.info("开始执行 trainer.train()")
    train_result = trainer.train()

    logger.info("开始保存 Adapter 到: %s", paths.adapter_dir)
    trainer.model.save_pretrained(paths.adapter_dir)
    tokenizer.save_pretrained(paths.adapter_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    save_training_summary(
        paths.summary_file,
        {
            "project_name": project_cfg["name"],
            "base_model_id": model_cfg["model_id"],
            "base_model_local_path": str(local_model_path),
            "dataset_path": str(dataset_path),
            "adapter_dir": str(paths.adapter_dir),
            "trainer_dir": str(paths.trainer_dir),
            "log_file": str(log_file),
            "metrics": train_result.metrics,
        },
    )
    logger.info("训练流程全部完成。")


if __name__ == "__main__":
    main()
