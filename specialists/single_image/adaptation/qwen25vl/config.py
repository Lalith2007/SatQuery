"""Configuration management and hardware diagnostics for Qwen2.5-VL Remote-Sensing Adaptation.

Defines schemas, environment variable overrides, hardware inspection routines,
and guard checks to enforce Colab CUDA training topologies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from core.logging import get_logger

logger = get_logger("qwen25vl_config")


@dataclass
class ModelConfig:
    """Model identity and checkpoint paths."""
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    supported_models: List[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ]
    )
    revision: str = "main"
    trust_remote_code: bool = True


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit QLoRA configuration."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"  # "bfloat16" or "float16"


@dataclass
class LoraConfigQwen:
    """PEFT QLoRA configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Explicit regex targeting language decoder and visual merger, freezing vision blocks
    target_modules_regex: str = (
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|.*merger\.mlp\.[02]"
    )
    # Target module suffixes for inventory discovery
    target_module_suffixes: List[str] = field(
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


@dataclass
class ResolutionConfig:
    """Dynamic resolution boundaries for satellite imagery."""
    min_pixels: int = 256 * 28 * 28   # 200,704 pixels (~448x448)
    max_pixels: int = 1024 * 28 * 28  # 802,816 pixels (~896x896)
    high_res_mode: bool = False
    high_res_max_pixels: int = 1280 * 28 * 28  # 1,003,520 pixels


@dataclass
class TrainingConfig:
    """Hyperparameters and optimizer settings for TRL SFTTrainer."""
    output_dir: str = "specialists/single_image/weights/qwen25vl_lora"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    optimizer: str = "paged_adamw_8bit"
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    eval_strategy: str = "steps"
    eval_steps: int = 50
    max_length: Optional[int] = None
    seed: int = 42

    @property
    def optim(self) -> str:
        return self.optimizer

    @optim.setter
    def optim(self, val: str) -> None:
        self.optimizer = val



@dataclass
class Qwen25VLFullConfig:
    """Master configuration container for Qwen2.5-VL adaptation."""
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfigQwen = field(default_factory=LoraConfigQwen)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_env(cls) -> Qwen25VLFullConfig:
        """Construct configuration with environment variable overrides."""
        cfg = cls()
        model_env = os.getenv("MODEL_ID", "").strip()
        if model_env:
            if model_env not in cfg.model.supported_models:
                logger.warning(
                    f"Custom MODEL_ID '{model_env}' specified. Supported baselines: {cfg.model.supported_models}"
                )
            cfg.model.model_id = model_env

        min_px = os.getenv("MIN_PIXELS", "").strip()
        if min_px.isdigit():
            cfg.resolution.min_pixels = int(min_px)

        max_px = os.getenv("MAX_PIXELS", "").strip()
        if max_px.isdigit():
            cfg.resolution.max_pixels = int(max_px)

        out_dir = os.getenv("OUTPUT_DIR", "").strip()
        if out_dir:
            cfg.training.output_dir = out_dir

        return cfg

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Qwen25VLFullConfig:
        """Load configuration from a YAML file."""
        import yaml
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Config YAML not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "quantization" in raw:
            for k, v in raw["quantization"].items():
                if hasattr(cfg.quantization, k):
                    setattr(cfg.quantization, k, v)
        if "lora" in raw:
            for k, v in raw["lora"].items():
                if hasattr(cfg.lora, k):
                    setattr(cfg.lora, k, v)
        if "resolution" in raw:
            for k, v in raw["resolution"].items():
                if hasattr(cfg.resolution, k):
                    setattr(cfg.resolution, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(cfg.training, k):
                    setattr(cfg.training, k, v)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def inspect_hardware() -> Dict[str, Any]:
    """Inspect and report the runtime hardware and package versions."""
    import transformers
    import peft

    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    try:
        import trl
        info["trl_version"] = trl.__version__
    except ImportError:
        info["trl_version"] = "not_installed"

    try:
        import bitsandbytes as bnb
        info["bitsandbytes_version"] = bnb.__version__
    except ImportError:
        info["bitsandbytes_version"] = "not_installed"

    try:
        import qwen_vl_utils
        info["qwen_vl_utils_version"] = getattr(qwen_vl_utils, "__version__", "installed")
    except ImportError:
        info["qwen_vl_utils_version"] = "not_installed"

    try:
        import datasets
        info["datasets_version"] = datasets.__version__
    except ImportError:
        info["datasets_version"] = "not_installed"

    try:
        import accelerate
        info["accelerate_version"] = accelerate.__version__
    except ImportError:
        info["accelerate_version"] = "not_installed"

    try:
        import pydantic_settings
        info["pydantic_settings_version"] = getattr(pydantic_settings, "__version__", "installed")
    except ImportError:
        info["pydantic_settings_version"] = "not_installed"

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["fp16_supported"] = True
    else:
        info["gpu_name"] = "None (CPU/MPS)"
        info["total_vram_gb"] = 0.0
        info["cuda_capability"] = None
        info["bf16_supported"] = False
        info["fp16_supported"] = False

    return info


def verify_cuda_available(strict: bool = True) -> bool:
    """Verify that a genuine CUDA environment is available for QLoRA training.
    
    Raises:
        RuntimeError: If strict is True and CUDA is unavailable.
    """
    hw = inspect_hardware()
    if not hw["cuda_available"]:
        msg = (
            "CUDA_NOT_AVAILABLE: QLoRA training of Qwen2.5-VL requires a genuine NVIDIA CUDA GPU "
            "(e.g., Google Colab T4/L4/A100). Local Apple Silicon or CPU environments are restricted "
            "to code validation, tests, packaging, and mock/fallback execution. "
            "Please launch the Colab training pipeline: specialists/single_image/training/colab/colab_qwen25vl_training.ipynb"
        )
        if strict:
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.warning(msg)
            return False
    return True
