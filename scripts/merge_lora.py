"""LoRA 合并脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.logging_utils import build_log_path, setup_logger
from huanhuan_sft.model_utils import build_merge_model, build_tokenizer
from huanhuan_sft.modelscope_utils import download_model_from_modelscope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 LoRA Adapter 到基座模型")
    parser.add_argument("--config", default="configs/train_config.yaml", help="训练配置文件路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_yaml_config(PROJECT_ROOT / args.config)
    config = loaded.raw

    log_dir = loaded.resolve_path("logs")
    log_file = build_log_path(log_dir, "merge")
    logger = setup_logger("merge_lora", log_file)

    model_cfg = config["model"]
    training_cfg = config["training"]
    merge_cfg = config["merge"]

    local_model_path = download_model_from_modelscope(
        model_id=model_cfg["model_id"],
        target_dir=loaded.resolve_path(model_cfg["local_dir"]),
    )

    run_dir = loaded.resolve_path(training_cfg["output_root"]) / training_cfg["run_name"]
    adapter_dir = run_dir / merge_cfg["adapter_subdir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"未找到 Adapter 目录: {adapter_dir}")

    merged_output_dir = loaded.resolve_path(merge_cfg["output_dir"])
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载基座模型: %s", local_model_path)
    base_model = build_merge_model(
        model_path=str(local_model_path),
        trust_remote_code=model_cfg["trust_remote_code"],
        dtype_name=model_cfg["torch_dtype"],
    )
    tokenizer = build_tokenizer(
        model_path=str(local_model_path),
        trust_remote_code=model_cfg["trust_remote_code"],
    )

    logger.info("加载 Adapter: %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    logger.info("开始 merge_and_unload()")
    merged_model = peft_model.merge_and_unload()

    logger.info("保存合并模型到: %s", merged_output_dir)
    merged_model.save_pretrained(
        merged_output_dir,
        safe_serialization=merge_cfg["safe_serialization"],
    )
    tokenizer.save_pretrained(merged_output_dir)
    logger.info("LoRA 合并完成。")


if __name__ == "__main__":
    main()
