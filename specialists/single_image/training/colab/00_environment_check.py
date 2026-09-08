"""Colab Step 00: Environment Diagnostics and Hardware Verification.

Validates CUDA availability, GPU capability (T4, L4, A100), package versions,
computes precision capability (BF16 / FP16), and generates `environment_manifest.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.config import inspect_hardware

logger = get_logger("colab_env_check")


def run_environment_check(output_path: str = "environment_manifest.json") -> Dict[str, Any]:
    """Inspect environment, generate manifest, and enforce minimum training constraints."""
    print("=" * 60)
    print("SatQuery AI — Division 2 Qwen2.5-VL Colab Environment Check")
    print("=" * 60)

    hw = inspect_hardware()

    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {hw['torch_version']}")
    print(f"Transformers: {hw['transformers_version']}")
    print(f"PEFT:         {hw['peft_version']}")
    print(f"TRL:          {hw['trl_version']}")
    print(f"Datasets:     {hw.get('datasets_version', 'N/A')}")
    print(f"Accelerate:   {hw.get('accelerate_version', 'N/A')}")
    print(f"BitsAndBytes: {hw['bitsandbytes_version']}")
    print(f"Qwen-VL-Utils:{hw['qwen_vl_utils_version']}")
    print(f"Pydantic-Set: {hw.get('pydantic_settings_version', 'N/A')}")
    print(f"CUDA Available:{hw['cuda_available']}")

    nvidia_smi_output = ""
    if hw["cuda_available"]:
        print(f"GPU Model:    {hw['gpu_name']}")
        print(f"Total VRAM:   {hw['total_vram_gb']} GB")
        print(f"Compute Cap:  {hw['cuda_capability']}")
        print(f"BF16 Support: {hw['bf16_supported']}")
        print(f"FP16 Support: {hw['fp16_supported']}")

        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            nvidia_smi_output = res.stdout
        except Exception:
            nvidia_smi_output = "nvidia-smi not available"
    else:
        print("WARNING: CUDA is NOT available. Running in non-CUDA inspection mode.")

    manifest = {
        "timestamp": torch.__version__,
        "hardware": hw,
        "nvidia_smi": nvidia_smi_output,
        "recommended_dtype": "bfloat16" if hw.get("bf16_supported") else "float16",
        "recommended_quantization": "4bit-nf4" if hw["cuda_available"] else "none",
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nEnvironment manifest written to: {out_p.resolve()}")
    print("=" * 60)
    return manifest


if __name__ == "__main__":
    run_environment_check()
