"""PEFT / LoRA Configuration for PaliGemma 3B Remote-Sensing Adaptation.

Specifies rank, alpha, target projection modules, and training hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SatQueryLoRAConfig:
    """LoRA hyperparameters for PaliGemma 3B remote sensing adaptation."""

    base_model_name: str = "google/paligemma-3b-pt-224"
    revision: str = "main"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    seed: int = 42
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.05
    max_seq_length: int = 256
    output_dir: str = "specialists/single_image/weights/satquery_paligemma_lora"

    def to_peft_config_dict(self) -> dict:
        """Convert to Hugging Face PEFT LoraConfig keyword arguments."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "task_type": "CAUSAL_LM",
        }
