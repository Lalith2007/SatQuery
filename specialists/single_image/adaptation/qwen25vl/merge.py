"""Optional Utility for Merging Qwen2.5-VL LoRA Adapter into Standalone Base Weights.

IMPORTANT:
Merging a 4-bit QLoRA adapter requires reloading the unquantized base weights in FP16 or BF16,
applying the adapter, executing `merge_and_unload()`, and saving the merged checkpoint.
This operation is an optional deployment pathway and NEVER modifies or overwrites the original adapter.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import torch

from core.logging import get_logger

logger = get_logger("qwen25vl_merge")


def merge_and_export_qwen_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
    device: str = "auto",
) -> Path:
    """Merge LoRA adapter into base model weights and save standalone Hugging Face checkpoint."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else (torch.float16 if torch_dtype == "float16" else torch.float32)

    logger.info(f"Loading unquantized base model: '{base_model_id}' with dtype={dtype}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    logger.info(f"Attaching LoRA adapter from: '{adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Executing merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"Saving merged standalone model to: '{out_p}'...")
    merged_model.save_pretrained(out_p)

    logger.info("Saving processor and tokenizer...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(out_p)

    readme_content = (
        f"# Merged Qwen2.5-VL Remote Sensing Model\n\n"
        f"- **Base Model**: `{base_model_id}`\n"
        f"- **Source Adapter**: `{adapter_path}`\n"
        f"- **Precision**: `{torch_dtype}`\n\n"
        "This is an unquantized standalone merged model exported for production serving.\n"
    )
    with open(out_p / "README.md", "w") as f:
        f.write(readme_content)

    logger.info(f"Merged model successfully exported to: {out_p}")
    return out_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Qwen2.5-VL LoRA adapter into base model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="specialists/single_image/weights/qwen25vl_lora")
    parser.add_argument("--output_dir", default="specialists/single_image/weights/qwen25vl_merged")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    merge_and_export_qwen_adapter(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        torch_dtype=args.dtype,
    )
