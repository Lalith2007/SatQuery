"""Colab Step 08: Trained Adapter Verification and Export Finalization.

Verifies the physical completeness of `specialists/single_image/weights/qwen25vl_lora/`,
computes the cryptographic SHA-256 of `adapter_model.safetensors`, and verifies PEFT configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from core.logging import get_logger

logger = get_logger("colab_export_adapter")


def verify_and_export_adapter(
    adapter_dir: str = "specialists/single_image/weights/qwen25vl_lora",
    manifest_path: str = "specialists/single_image/weights/qwen25vl_lora/adapter_verification.json",
) -> Dict[str, Any]:
    """Verify adapter weights file, compute checksum, and generate manifest."""
    dir_p = Path(adapter_dir)
    safetensors_p = dir_p / "adapter_model.safetensors"
    config_p = dir_p / "adapter_config.json"

    print("=" * 60)
    print("SatQuery AI — Division 2 Qwen2.5-VL Adapter Verification")
    print("=" * 60)

    if not dir_p.exists():
        raise FileNotFoundError(f"Adapter directory not found: {dir_p}")

    if not config_p.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {dir_p}")

    with open(config_p, "r") as f:
        adapter_cfg = json.load(f)

    file_size_mb = 0.0
    sha256_hash = "not_found"

    if safetensors_p.exists():
        file_size_mb = round(safetensors_p.stat().st_size / (1024 * 1024), 2)
        sha = hashlib.sha256()
        with open(safetensors_p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        sha256_hash = sha.hexdigest()
        print(f"adapter_model.safetensors: {file_size_mb} MB")
        print(f"Cryptographic SHA-256:     {sha256_hash}")
    else:
        print("WARNING: adapter_model.safetensors not present (smoke test or synthetic run).")

    manifest = {
        "adapter_dir": str(dir_p.resolve()),
        "safetensors_exists": safetensors_p.exists(),
        "file_size_mb": file_size_mb,
        "sha256": sha256_hash,
        "base_model": adapter_cfg.get("base_model_name_or_path"),
        "peft_type": adapter_cfg.get("peft_type", "LORA"),
        "r": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "target_modules": adapter_cfg.get("target_modules"),
        "verification_status": "VERIFIED" if safetensors_p.exists() else "CONFIG_ONLY",
    }

    out_p = Path(manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Verification manifest written to: {out_p.resolve()}")
    print("=" * 60)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="specialists/single_image/weights/qwen25vl_lora")
    args = parser.parse_args()
    verify_and_export_adapter(args.adapter_dir)
