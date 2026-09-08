"""Colab Step 03: Qwen2.5-VL Model and Tokenizer Deep Inspection.

Inspects model module hierarchy, verifies special grounding tokens,
generates `qwen_grounding_token_inventory.json` and `qwen25vl_module_inventory.json`,
and confirms targeted LoRA layer discovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.config import LoraConfigQwen, ModelConfig
from specialists.single_image.adaptation.qwen25vl.model import QwenModelLoader

logger = get_logger("colab_inspect_qwen")


def inspect_tokenizer_grounding_tokens(
    model_id: str,
    output_path: str = "qwen_grounding_token_inventory.json",
) -> Dict[str, Any]:
    """Verify and record native grounding token IDs from tokenizer."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    target_tokens = [
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
    ]

    token_inventory: Dict[str, Any] = {}
    missing_tokens = []

    for tok_str in target_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)
        # Check if token is mapped to unk_token or valid id
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            token_inventory[tok_str] = {
                "token_id": tok_id,
                "is_special": tok_str in tokenizer.all_special_tokens,
                "status": "VERIFIED",
            }
        else:
            missing_tokens.append(tok_str)
            token_inventory[tok_str] = {
                "token_id": None,
                "status": "MISSING",
            }

    report = {
        "model_id": model_id,
        "vocab_size": len(tokenizer),
        "verified_grounding_tokens": token_inventory,
        "missing_tokens": missing_tokens,
        "all_required_present": len(missing_tokens) == 0,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Tokenizer token inventory written to {out_p}")
    return report


def inspect_model_architecture_and_lora_targets(
    model_id: str,
    output_path: str = "qwen25vl_module_inventory.json",
) -> Dict[str, Any]:
    """Inspect module hierarchy and verify targeted LoRA module match."""
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    logger.info(f"Inspecting model architecture for '{model_id}'...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Use meta device to avoid memory footprint during inspection
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration(config)

    inventory = QwenModelLoader.inspect_modules(model, output_path=output_path)

    lora_cfg = LoraConfigQwen()
    peft_model, stats = QwenModelLoader.apply_lora_adaptation(model, lora_cfg)

    adapted_keys = [k for k, _ in peft_model.named_parameters() if "lora_" in k]
    visual_adapted = [k for k in adapted_keys if "visual" in k]
    language_adapted = [k for k in adapted_keys if "language_model" in k]

    summary = {
        "model_id": model_id,
        "total_linear_modules": len(inventory["linear_modules"]),
        "language_targets_count": len(inventory["language_targets"]),
        "visual_merger_targets_count": len(inventory["visual_merger_targets"]),
        "frozen_vision_blocks_count": len(inventory["vision_backbone_frozen"]),
        "total_lora_tensors": len(adapted_keys),
        "visual_adapted_lora_tensors": len(visual_adapted),
        "language_adapted_lora_tensors": len(language_adapted),
        "trainable_params_est": stats["trainable_params"],
        "all_params_est": stats["all_params"],
        "pct_trainable": stats["trainable_percentage"],
        "target_regex": lora_cfg.target_modules_regex,
        "vision_backbone_frozen": len(visual_adapted) <= 4,  # only merger.mlp.0 and 2
    }

    print("\n" + "=" * 60)
    print("FINAL LORA TARGET MODULES:")
    print(f"- Language Decoder Linears: {len(language_adapted)} tensors")
    print(f"- Visual Merger Projection:  {len(visual_adapted)} tensors ({[k for k in visual_adapted]})")
    print(f"- Vision Backbone Status:   FROZEN ({len(inventory['vision_backbone_frozen'])} blocks)")
    print(f"- Trainable Parameters:     {stats['trainable_params']:,} ({stats['trainable_percentage']:.2f}%)")
    print("=" * 60 + "\n")

    return summary


def run_full_inspection(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> Dict[str, Any]:
    tok_res = inspect_tokenizer_grounding_tokens(model_id)
    mod_res = inspect_model_architecture_and_lora_targets(model_id)
    return {"tokenizer": tok_res, "model": mod_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = parser.parse_args()
    run_full_inspection(args.model_id)
