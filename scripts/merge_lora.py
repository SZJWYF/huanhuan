"""LoRA 合并脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from huanhuan_sft.config import load_yaml_config
from huanhuan_sft.logging_utils import build_log_path, setup_logger
from huanhuan_sft.model_utils import build_merge_model, build_tokenizer
from huanhuan_sft.modelscope_utils import resolve_model_path


def _read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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

    logger.info("优先检查本地模型目录: %s", model_cfg.get("local_model_path", "<未配置>"))
    local_model_path = resolve_model_path(model_cfg, loaded.resolve_path)

    run_dir = loaded.resolve_path(training_cfg["output_root"]) / training_cfg["run_name"]
    adapter_dir = run_dir / merge_cfg["adapter_subdir"]
    if not adapter_dir.exists():
        raise FileNotFoundError(f"未找到 Adapter 目录: {adapter_dir}")
    adapter_weight_files = sorted([path for path in adapter_dir.glob("adapter_model.*") if path.is_file()])
    if not adapter_weight_files:
        raise FileNotFoundError(f"Adapter 目录中未找到权重文件（adapter_model.*）: {adapter_dir}")

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
    training_summary_file = run_dir / "training_summary.json"
    trainer_state_file = run_dir / "trainer_state" / "trainer_state.json"
    training_summary = _read_json_if_exists(training_summary_file)
    trainer_state = _read_json_if_exists(trainer_state_file)
    merged_metadata = {
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_path": str(local_model_path),
        "adapter_dir": str(adapter_dir),
        "merged_output_dir": str(merged_output_dir),
        "adapter_weight_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
            for path in adapter_weight_files
        ],
        "training_summary_file": str(training_summary_file),
        "trainer_state_file": str(trainer_state_file),
        "train_global_step": (trainer_state or {}).get("global_step"),
        "training_metrics": (training_summary or {}).get("metrics"),
    }
    metadata_file = merged_output_dir / "merge_metadata.json"
    metadata_file.write_text(
        json.dumps(merged_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("合并元数据已写入: %s", metadata_file)
    logger.info("LoRA 合并完成。")


if __name__ == "__main__":
    main()
