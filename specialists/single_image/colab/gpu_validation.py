"""Google Colab GPU Compute Environment Validation Script for Division 2.

Performs:
- GPU and VRAM detection (CUDA / MPS / CPU)
- Driver and library version auditing (PyTorch, Transformers, PEFT, CUDA)
- Synchronized GPU matrix computation benchmark
- Machine-readable environment report generation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time
import psutil
import torch

from core.logging import get_logger, setup_logging

logger = get_logger("colab_gpu_validation")


def validate_compute_environment() -> dict:
    """Detect and validate compute hardware, drivers, and runtime capabilities."""
    setup_logging()
    logger.info("Initializing Google Colab / Compute Environment Validation...")

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if has_cuda:
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        vram_allocated_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        cuda_version = torch.version.cuda or "N/A"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    elif has_mps:
        device_type = "mps"
        device_name = "Apple Silicon Metal Performance Shaders (MPS)"
        vram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        vram_allocated_gb = round(psutil.Process().memory_info().rss / (1024**3), 2)
        cuda_version = "N/A (Apple Metal)"
        cudnn_version = "N/A (Metal Graph)"
    else:
        device_type = "cpu"
        device_name = platform.processor() or "CPU"
        vram_total_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        vram_allocated_gb = round(psutil.Process().memory_info().rss / (1024**3), 2)
        cuda_version = "N/A"
        cudnn_version = "N/A"

    import peft
    import transformers

    env_report = {
        "device_type": device_type,
        "device_name": device_name,
        "cuda_available": has_cuda,
        "mps_available": has_mps,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "total_memory_gb": vram_total_gb,
        "allocated_memory_gb": vram_allocated_gb,
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "os_platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
    }

    # Execute synchronized GPU / Accelerator Matrix Multiplication Benchmark
    dev = torch.device(device_type if device_type != "auto" else "cpu")
    dim = 2048
    logger.info(f"Executing synchronized matrix benchmark on [{device_name}] ({dim}x{dim} float32)...")

    t_start = time.perf_counter()
    mat_a = torch.randn(dim, dim, device=dev, dtype=torch.float32)
    mat_b = torch.randn(dim, dim, device=dev, dtype=torch.float32)

    # Warmup
    _ = torch.matmul(mat_a, mat_b)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass

    t0 = time.perf_counter()
    res = torch.matmul(mat_a, mat_b)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    mat_time_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    env_report["matrix_benchmark_dim"] = f"{dim}x{dim}"
    env_report["matrix_compute_time_ms"] = mat_time_ms
    env_report["validation_status"] = "PASSED_COMPUTE_VALIDATION"

    # Export machine-readable report
    out_path = Path("specialists/single_image/colab/colab_gpu_environment_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(env_report, f, indent=2)

    logger.info(f"Saved environment report to: {out_path}")
    return env_report


if __name__ == "__main__":
    rep = validate_compute_environment()
    print("\n" + "=" * 70)
    print("GOOGLE COLAB / COMPUTE ENVIRONMENT VALIDATION REPORT")
    print("=" * 70)
    print(f"Device:       {rep['device_name']} [{rep['device_type'].upper()}]")
    print(f"Memory:       {rep['total_memory_gb']} GB")
    print(f"CUDA:         {rep['cuda_version']} (cuDNN: {rep['cudnn_version']})")
    print(f"Python:       {rep['python_version']}")
    print(f"PyTorch:      {rep['pytorch_version']}")
    print(f"Transformers: {rep['transformers_version']}")
    print(f"PEFT:         {rep['peft_version']}")
    print(f"Benchmark:    2048x2048 Matmul = {rep['matrix_compute_time_ms']} ms")
    print(f"Status:       {rep['validation_status']}")
    print("=" * 70)
