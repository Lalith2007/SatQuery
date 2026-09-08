"""Kaggle Step 00: Environment Diagnostics and Hardware Verification.

Validates Kaggle execution environment, CUDA availability, GPU capability (T4, P100),
working disk quotas, package versions, compute precision (BF16 / FP16),
and generates `environment_manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

try:
    import torchaudio  # noqa: F401
except (RuntimeError, Exception):
    sys.modules["torchaudio"] = None

import torch

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.config import inspect_hardware

logger = get_logger("kaggle_env_check")


def run_kaggle_environment_check(
    output_path: str = "environment_manifest.json",
    require_cuda: bool = True,
    allow_non_cuda: bool = False,
) -> Dict[str, Any]:
    """Inspect Kaggle environment, generate manifest, and enforce training constraints."""
    print("=" * 65)
    print("SatQuery AI — Division 2 Qwen2.5-VL Kaggle Environment Diagnostics")
    print("=" * 65)

    is_kaggle = Path("/kaggle").exists()
    print(f"Kaggle Environment: {is_kaggle}")

    # Inspect disk space on /kaggle/working
    working_dir = Path("/kaggle/working") if is_kaggle else Path.cwd()
    total, used, free = shutil.disk_usage(working_dir)
    print(f"Working Directory:  {working_dir.resolve()}")
    print(f"Working Disk Total: {total / (1024**3):.2f} GB")
    print(f"Working Disk Used:  {used / (1024**3):.2f} GB")
    print(f"Working Disk Free:  {free / (1024**3):.2f} GB")

    hw = inspect_hardware()

    flash_attn_available = False
    try:
        import flash_attn  # noqa: F401
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
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if hw["cuda_available"]:
        print(f"CUDA Version:       {torch.version.cuda}")
        print(f"GPU Count:          {gpu_count}")
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
            print("Training must run on Kaggle with an active GPU accelerator (T4 x 2 or P100).")
            print("Go to Kaggle Notebook Settings -> Accelerator -> select 'GPU T4 x 2' or 'GPU P100'.")
            print("Silently falling back to CPU, MPS, or mock training is STRICTLY FORBIDDEN.")
            print("!" * 65 + "\n")
            raise RuntimeError(
                "CRITICAL: CUDA is required for Qwen2.5-VL QLoRA training on Kaggle! "
                "No active CUDA device found. Stopping execution."
            )
        else:
            print("WARNING: Running in LOCAL-SMOKE-TEST mode (Non-CUDA environment).")
            execution_mode = "LOCAL-SMOKE-TEST"

    print(f"\n>>> EXECUTION_MODE={execution_mode} <<<\n")

    manifest = {
        "execution_mode": execution_mode,
        "environment": "Kaggle" if is_kaggle else "Local/Generic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "hardware": hw,
        "gpu_count": gpu_count,
        "working_disk": {
            "path": str(working_dir.resolve()),
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
        },
        "flash_attention_available": flash_attn_available,
        "nvidia_smi": nvidia_smi_output.splitlines()[:15] if nvidia_smi_output else [],
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Environment manifest saved to: {out_p.resolve()}")
    print("Hardware check completed successfully.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Kaggle Environment Diagnostic Check")
    parser.add_argument("--manifest_path", type=str, default="environment_manifest.json")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow_non_cuda", action="store_true", default=False)
    args = parser.parse_args()

    run_kaggle_environment_check(
        output_path=args.manifest_path,
        require_cuda=args.strict,
        allow_non_cuda=args.allow_non_cuda,
    )


if __name__ == "__main__":
    main()
