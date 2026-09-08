"""Colab Step 06: Production Qwen2.5-VL 4-Bit QLoRA Training Pipeline.

Integrates TRL SFTTrainer, BitsAndBytes 4-bit NF4 quantization, targeted PEFT LoRA,
multimodal data collation, and telemetry logging for remote-sensing instruction tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import yaml
import torch
from datasets import load_dataset

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.collator import Qwen25VLDataCollator
from specialists.single_image.adaptation.qwen25vl.config import (
    LoraConfigQwen,
    ModelConfig,
    QuantizationConfig,
    Qwen25VLFullConfig,
    ResolutionConfig,
    TrainingConfig,
    inspect_hardware,
    verify_cuda_available,
)
from specialists.single_image.adaptation.qwen25vl.model import QwenModelLoader

logger = get_logger("colab_train_qlora")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Config file not found at '{config_path}'. Using default config.")
        return {}
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}


def train_qwen25vl_qlora(
    config_path: str = "configs/qwen25vl_qlora.yaml",
    train_file: str = "data/qwen_dataset/train.jsonl",
    val_file: str = "data/qwen_dataset/val.jsonl",
    output_dir: Optional[str] = None,
    allow_non_cuda: bool = False,
) -> Dict[str, Any]:
    """Execute complete Qwen2.5-VL QLoRA training on CUDA."""
    print("=" * 60)
    print("SatQuery AI — Division 2 Qwen2.5-VL Production QLoRA Training")
    print("=" * 60)

    # 1. Enforce CUDA Hardware Guard
    hw = inspect_hardware()
    if not hw["cuda_available"] and not allow_non_cuda:
        verify_cuda_available(strict=True)

    t_start = time.perf_counter()
    cfg_dict = load_yaml_config(config_path)

    model_id = cfg_dict.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    out_dir = Path(output_dir or cfg_dict.get("output_dir", "specialists/single_image/weights/qwen25vl_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base Model:  {model_id}")
    print(f"Output Path: {out_dir.resolve()}")
    print(f"Train File:  {train_file}")
    print(f"Val File:    {val_file}")

    # 2. Load Dataset Splits
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
    )
    print(f"Dataset Loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")

    # 3. Model & Quantization Setup
    quant_cfg = QuantizationConfig(
        load_in_4bit=cfg_dict.get("load_in_4bit", True),
        bnb_4bit_quant_type=cfg_dict.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg_dict.get("bnb_4bit_use_double_quant", True),
        compute_dtype="bfloat16" if hw.get("bf16_supported") else "float16",
    )

    print("Loading base Qwen2.5-VL model and processor...")
    device = "cuda" if hw["cuda_available"] else ("mps" if hw.get("mps_available") else "cpu")
    model, processor = QwenModelLoader.load_base_and_adapter(
        base_model_id=model_id,
        device=device,
        quant_cfg=quant_cfg,
    )

    # 4. LoRA Setup
    lora_cfg = LoraConfigQwen(
        r=cfg_dict.get("lora_r", 16),
        lora_alpha=cfg_dict.get("lora_alpha", 32),
        lora_dropout=cfg_dict.get("lora_dropout", 0.05),
    )
    peft_model, lora_stats = QwenModelLoader.apply_lora_adaptation(model, lora_cfg)

    # 5. Multimodal Data Collator
    collator = Qwen25VLDataCollator(processor=processor)

    # 6. SFTConfig and SFTTrainer
    from trl import SFTConfig, SFTTrainer

    batch_size = cfg_dict.get("per_device_train_batch_size", 1)
    grad_accum = cfg_dict.get("gradient_accumulation_steps", 8)
    lr = float(cfg_dict.get("learning_rate", 2e-4))
    epochs = int(cfg_dict.get("num_train_epochs", 3))

    # Check for existing checkpoint to resume
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = sorted(list(checkpoint_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    resume_checkpoint = str(existing_checkpoints[-1]) if existing_checkpoints else None
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")

    # Calculate warmup steps from warmup_ratio
    warmup_ratio = float(cfg_dict.get("warmup_ratio", 0.03))
    train_count = len(dataset["train"])
    steps_per_epoch = max(1, train_count // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(cfg_dict.get("warmup_steps", max(10, int(total_steps * warmup_ratio))))
    print(f"Calculated warmup_steps: {warmup_steps} (total steps: ~{total_steps})")

    training_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim=cfg_dict.get("optimizer", "paged_adamw_8bit"),
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=epochs,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        max_length=None,  # Do not truncate multimodal sequences
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        fp16=(not hw.get("bf16_supported") and hw["cuda_available"]),
        bf16=(hw.get("bf16_supported", False) and hw["cuda_available"]),
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    print("Launching SFTTrainer optimization loop...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 7. Save Final Adapter and Artifacts
    print(f"Saving final trained adapter to: {out_dir}...")
    trainer.model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # Sync checkpoints and adapter to Google Drive if configured
    gdrive_dir = cfg_dict.get("google_drive_dir")
    if gdrive_dir and Path(gdrive_dir).parent.exists():
        import shutil
        gdrive_p = Path(gdrive_dir) / "checkpoints"
        gdrive_p.mkdir(parents=True, exist_ok=True)
        print(f"Syncing adapter and checkpoints to persistent Google Drive: {gdrive_p}...")
        try:
            shutil.copytree(out_dir, Path(gdrive_dir) / "adapter", dirs_exist_ok=True)
            print(f"Persistent backup to Google Drive completed: {Path(gdrive_dir) / 'adapter'}")
        except Exception as e:
            logger.warning(f"Google Drive sync warning: {e}")

    # Calculate adapter SHA-256
    safetensors_path = out_dir / "adapter_model.safetensors"
    adapter_sha = compute_sha256(safetensors_path) if safetensors_path.exists() else "not_generated"

    metrics = {
        "model_id": model_id,
        "adapter_sha256": adapter_sha,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "train_loss": train_result.training_loss,
        "total_train_time_seconds": round(time.perf_counter() - t_start, 2),
        "hardware": hw,
        "lora_stats": lora_stats,
    }

    with open(out_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "training_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    readme_content = (
        f"# SatQuery Division 2 — Qwen2.5-VL Remote-Sensing Adapter\n\n"
        f"- **Base Model**: `{model_id}`\n"
        f"- **Adapter SHA-256**: `{adapter_sha}`\n"
        f"- **Architecture**: Targeted QLoRA (Language Decoder + Visual Merger)\n"
        f"- **Train Loss**: `{train_result.training_loss:.4f}`\n"
        f"- **Training Samples**: `{len(dataset['train'])}`\n"
    )
    with open(out_p := out_dir / "README.md", "w") as f:
        f.write(readme_content)

    print(f"Training completed successfully! Adapter SHA-256: {adapter_sha}")
    print("FULL STAGE 1 QLORA TRAINING: PASS")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25vl_qlora.yaml")
    parser.add_argument("--train_file", default="data/qwen_dataset/train.jsonl")
    parser.add_argument("--val_file", default="data/qwen_dataset/val.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--allow_non_cuda", action="store_true")
    args = parser.parse_args()

    train_qwen25vl_qlora(
        config_path=args.config,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        allow_non_cuda=args.allow_non_cuda,
    )
