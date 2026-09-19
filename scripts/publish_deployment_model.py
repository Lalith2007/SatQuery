"""Model Publishing CLI Utility for SatQuery AI.

Publishes an explicitly verified model revision (e.g. stage1-baseline) to the
Hugging Face Model Repository: Lalith47/SatQuery-Models.

Verification Steps:
1. Verifies presence of all model shards and configuration files
2. Computes SHA256 checksums of safetensors shards
3. Validates AutoProcessor instantiation
4. Audits parameter count (3,754,622,976 parameters)
5. Uploads to Hugging Face Hub under explicit revision tag
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


EXPECTED_PARAMS = 3754622976
EXPECTED_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash in 8MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_model_directory(model_dir: Path) -> Dict[str, Any]:
    """Perform rigorous preflight validation of model artifacts."""
    print(f"[*] Verifying model artifacts in: {model_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    missing = []
    file_details = {}
    for fn in EXPECTED_FILES:
        fp = model_dir / fn
        if not fp.exists() or fp.stat().st_size == 0:
            missing.append(fn)
        else:
            file_details[fn] = {"size_bytes": fp.stat().st_size}

    if missing:
        raise FileNotFoundError(f"Missing or empty required files: {missing}")

    print("[*] Computing shard SHA-256 hashes...")
    t0 = time.time()
    s1 = sha256_file(model_dir / "model-00001-of-00002.safetensors")
    file_details["model-00001-of-00002.safetensors"]["sha256"] = s1
    print(f"  - model-00001-of-00002.safetensors: {s1} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    s2 = sha256_file(model_dir / "model-00002-of-00002.safetensors")
    file_details["model-00002-of-00002.safetensors"]["sha256"] = s2
    print(f"  - model-00002-of-00002.safetensors: {s2} ({time.time() - t0:.1f}s)")

    print("[*] Verifying AutoProcessor instantiation...")
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    print("  Processor initialized successfully.")

    print("[*] Auditing model architecture & parameter count...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    if total_params != EXPECTED_PARAMS:
        raise ValueError(
            f"Parameter mismatch! Expected {EXPECTED_PARAMS:,}, got {total_params:,}."
        )

    return {
        "total_parameters": total_params,
        "files": file_details,
        "status": "VERIFIED",
    }


def publish_model(
    model_dir: Path,
    repo_id: str = "Lalith47/SatQuery-Models",
    revision: str = "stage1-baseline",
    token: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Publish verified model checkpoint to Hugging Face Hub under explicit revision."""
    verification = verify_model_directory(model_dir)

    print("=" * 75)
    print("MODEL PREFLIGHT AUDIT PASSED")
    print(f"Target Repository: {repo_id}")
    print(f"Target Revision:   {revision}")
    print(f"Parameters:        {verification['total_parameters']:,}")
    print("=" * 75)

    if dry_run:
        print("[INFO] Dry-run enabled. Skipping network upload to Hugging Face Hub.")
        return

    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        print("[WARNING] HF_TOKEN environment variable not found and --token not provided.")
        print("To publish to Hugging Face Hub, supply HF_TOKEN or --token.")
        print("Artifacts remain verified locally in: " + str(model_dir))
        return

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    print(f"[*] Creating or verifying repository '{repo_id}' on Hugging Face...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"[*] Uploading model artifacts to '{repo_id}' at revision '{revision}'...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=f"Publish verified baseline checkpoint: {revision}",
    )

    print("=" * 75)
    print("MODEL PUBLISH COMPLETED SUCCESSFULLY")
    print(f"Hugging Face Model: https://huggingface.co/{repo_id}/tree/{revision}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Publish verified SatQuery model to Hugging Face")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="merged_full",
        help="Path to verified standalone Qwen checkpoint directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Lalith47/SatQuery-Models",
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="stage1-baseline",
        help="Immutable revision tag (default: stage1-baseline)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify files and parameters without network upload",
    )
    args = parser.parse_args()

    publish_model(
        model_dir=Path(args.model_dir),
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
