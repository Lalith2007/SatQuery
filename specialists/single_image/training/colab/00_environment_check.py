"""Colab Step 00: Environment Diagnostics and Hardware Verification.

Validates CUDA availability, GPU capability (T4, L4, A100), package versions,
computes precision capability (BF16 / FP16), and generates `environment_manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

# Colab runtime compatibility: neutralize torchaudio if CUDA version mismatch causes RuntimeError
try:
    import torchaudio  # noqa: F401
except (RuntimeError, Exception):
    sys.modules["torchaudio"] = None

import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.config import inspect_hardware

logger = get_logger("colab_env_check")


def run_environment_check(
    output_path: str = "environment_manifest.json",
    require_cuda: bool = True,
    allow_non_cuda: bool = False,
) -> Dict[str, Any]:
    """Inspect environment, generate manifest, and enforce minimum training constraints."""
    print("=" * 65)
    print("SatQuery AI — Division 2 Qwen2.5-VL Colab Environment Diagnostics")
    print("=" * 65)

    hw = inspect_hardware()

    # Flash Attention detection
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
    except ImportError:
        flash_attn_available = False

    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {hw['torch_version']}")
    print(f"Transformers:       {hw['transformers_version']}")
    print(f"PEFT:               {hw['peft_version']}")
    print(f"TRL:                {hw['trl_version']}")
    print(f"Datasets:           {hw.get('datasets_version', 'N/A')}")
    print(f"Accelerate:         {hw.get('accelerate_version', 'N/A')}")
    print(f"BitsAndBytes:       {hw['bitsandbytes_version']}")
    print(f"Qwen-VL-Utils:      {hw['qwen_vl_utils_version']}")
    print(f"CUDA Available:     {hw['cuda_available']}")
    print(f"Flash-Attention:    {flash_attn_available}")

    nvidia_smi_output = ""
    if hw["cuda_available"]:
        print(f"CUDA Version:       {torch.version.cuda}")
        print(f"GPU Model:          {hw['gpu_name']}")
        print(f"Total VRAM:         {hw['total_vram_gb']} GB")
        print(f"Compute Cap:        {hw['cuda_capability']}")
        print(f"BF16 Support:       {hw['bf16_supported']}")
        print(f"FP16 Support:       {hw['fp16_supported']}")
        execution_mode = "REAL-CUDA"

        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            nvidia_smi_output = res.stdout
        except Exception:
            nvidia_smi_output = "nvidia-smi not available"
    else:
        if require_cuda and not allow_non_cuda:
            print("\n" + "!" * 65)
            print("CRITICAL STOP: CUDA is NOT available.")
            print("Training must run on Google Colab with an active NVIDIA GPU (T4, L4, A100).")
            print("Silently falling back to CPU, MPS, or mock training is STRICTLY FORBIDDEN.")
            print("!" * 65 + "\n")
            raise RuntimeError(
                "CRITICAL: CUDA is required for Qwen2.5-VL QLoRA training! "
                "No active CUDA device found. Stopping execution."
            )
        else:
            print("WARNING: Running in LOCAL-SMOKE-TEST mode (Non-CUDA environment).")
            execution_mode = "LOCAL-SMOKE-TEST"

    print(f"\n>>> EXECUTION_MODE={execution_mode} <<<\n")

    manifest = {
        "execution_mode": execution_mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "hardware": hw,
        "flash_attn_available": flash_attn_available,
        "nvidia_smi": nvidia_smi_output,
        "recommended_dtype": "bfloat16" if hw.get("bf16_supported") else "float16",
        "recommended_quantization": "4bit-nf4" if hw["cuda_available"] else "none",
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Environment manifest written to: {out_p.resolve()}")
    print("=" * 65)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Environment Diagnostics")
    parser.add_argument("--output_path", default=None, help="Output path for environment_manifest.json")
    parser.add_argument("--manifest_path", default=None, help="Alias for --output_path")
    parser.add_argument("--allow_non_cuda", action="store_true", help="Allow non-CUDA for local testing")
    parser.add_argument("--strict", action="store_true", help="Enforce strict CUDA hardware checks")
    args = parser.parse_args()

    target_out = args.manifest_path or args.output_path or "environment_manifest.json"
    req_cuda = not args.allow_non_cuda or args.strict
    allow_non_cuda = args.allow_non_cuda and not args.strict

    run_environment_check(
        output_path=target_out,
        require_cuda=req_cuda,
        allow_non_cuda=allow_non_cuda,
    )
