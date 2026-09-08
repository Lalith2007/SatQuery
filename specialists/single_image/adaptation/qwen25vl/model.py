"""Qwen2.5-VL Model and Adapter Loading Infrastructure.

Handles:
- BitsAndBytes 4-bit NF4 quantization
- Targeted PEFT LoRA injection (language decoder + visual merger; vision backbone frozen)
- Module inventory inspection and candidate LoRA discovery
- Programmatic parameter audit (trainable vs frozen)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.config import (
    LoraConfigQwen,
    ModelConfig,
    QuantizationConfig,
    Qwen25VLFullConfig,
    inspect_hardware,
)

logger = get_logger("qwen25vl_model")


class QwenModelLoader:
    """Orchestrates model instantiation, quantization, and PEFT adaptation."""

    @classmethod
    def get_bitsandbytes_config(
        cls,
        quant_cfg: QuantizationConfig,
        cuda_available: bool = True,
    ) -> Optional[Any]:
        """Construct BitsAndBytesConfig if running on CUDA."""
        if not cuda_available or not quant_cfg.load_in_4bit:
            return None

        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    @classmethod
    def inspect_modules(
        cls,
        model: torch.nn.Module,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Programmatically inspect the model module tree and categorize candidate LoRA targets."""
        inventory: Dict[str, Any] = {
            "total_modules": 0,
            "linear_modules": [],
            "language_targets": [],
            "visual_merger_targets": [],
            "vision_backbone_frozen": [],
            "other_modules": [],
        }

        all_modules = list(model.named_modules())
        inventory["total_modules"] = len(all_modules)

        for name, mod in all_modules:
            cls_name = mod.__class__.__name__
            if "Linear" in cls_name:
                param_count = sum(p.numel() for p in mod.parameters())
                entry = {
                    "path": name,
                    "class": cls_name,
                    "parameter_count": param_count,
                }
                inventory["linear_modules"].append(entry)

                if "visual.blocks" in name:
                    # Vision encoder backbone - MUST REMAIN FROZEN
                    inventory["vision_backbone_frozen"].append(name)
                elif "visual.merger" in name:
                    # Multimodal visual merger - CANDIDATE TARGET
                    inventory["visual_merger_targets"].append(name)
                elif "language_model" in name or "lm_head" not in name:
                    # Language model linear layers - PRIMARY TARGETS
                    inventory["language_targets"].append(name)
                else:
                    inventory["other_modules"].append(name)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                json.dump(inventory, f, indent=2)
            logger.info(f"Saved module inventory ({len(inventory['linear_modules'])} linear layers) to {output_path}")

        return inventory

    @classmethod
    def apply_lora_adaptation(
        cls,
        model: torch.nn.Module,
        lora_cfg: LoraConfigQwen,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Apply targeted PEFT QLoRA adaptation to language decoder + visual merger."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # Prepare for k-bit training if model is quantized
        if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
            target_modules=lora_cfg.target_modules_regex,
        )

        peft_model = get_peft_model(model, peft_config)

        # Audit parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in peft_model.parameters())
        pct_trainable = (trainable_params / all_params) * 100.0 if all_params > 0 else 0.0

        stats = {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": round(pct_trainable, 4),
            "target_regex": lora_cfg.target_modules_regex,
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
        }

        logger.info(
            f"PEFT LoRA Applied: {trainable_params:,} trainable params / {all_params:,} total ({pct_trainable:.2f}%)"
        )
        return peft_model, stats

    @classmethod
    def load_base_and_adapter(
        cls,
        base_model_id: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        quant_cfg: Optional[QuantizationConfig] = None,
    ) -> Tuple[Any, Any]:
        """Load base Qwen2.5-VL model, processor, and optional LoRA adapter."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel

        hw = inspect_hardware()
        target_device = device or ("cuda" if hw["cuda_available"] else ("mps" if hw["mps_available"] else "cpu"))

        dtype = torch_dtype
        if dtype is None:
            if target_device == "cuda":
                dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16
            elif target_device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32

        logger.info(f"Loading processor for '{base_model_id}'...")
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

        bnb_config = cls.get_bitsandbytes_config(quant_cfg, cuda_available=hw["cuda_available"]) if quant_cfg else None

        logger.info(f"Loading base Qwen2.5-VL model on '{target_device}' with dtype={dtype}...")
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif target_device != "mps":
            model_kwargs["device_map"] = target_device
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            **model_kwargs,
        )

        if target_device == "mps" and bnb_config is None:
            model = model.to("mps")

        if adapter_path and Path(adapter_path).exists():
            logger.info(f"Attaching LoRA adapter from: '{adapter_path}'...")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        return model, processor
