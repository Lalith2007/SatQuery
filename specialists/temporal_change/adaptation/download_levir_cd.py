"""Automated downloader for official LEVIR-CD change detection benchmark dataset.

Downloads full LEVIR-CD repository directly from HuggingFace into target directory.
Supports optional HuggingFace Token for authenticated / high-bandwidth access.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional


def download_levir_cd(target_dir: str = "/content/LEVIR-CD", token: Optional[str] = None) -> Path:
    """Download official LEVIR-CD dataset via huggingface_hub."""
    t_dir = Path(target_dir)
    t_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset is already present
    if (t_dir / "train" / "A").exists() or (t_dir / "train" / "a").exists():
        print(f"LEVIR-CD dataset already present at: {t_dir}")
        return t_dir

    hf_token = token or os.getenv("HF_TOKEN")

    print(f"Downloading official LEVIR-CD dataset to {t_dir}...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="satellite-image-deep-learning/LEVIR-CD",
            repo_type="dataset",
            local_dir=str(t_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        print(f"Download complete! Dataset saved at: {t_dir}")
    except Exception as e:
        print(f"HuggingFace download encountered: {e}")
        if not hf_token:
            print("TIP: If rate-limited or access restricted, pass --token YOUR_HF_TOKEN or set HF_TOKEN env variable.")
        raise

    return t_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download official LEVIR-CD dataset.")
    parser.add_argument("--target-dir", type=str, default="/content/LEVIR-CD", help="Target directory for LEVIR-CD")
    parser.add_argument("--token", type=str, default=None, help="Optional HuggingFace access token")
    args = parser.parse_args()
    download_levir_cd(args.target_dir, token=args.token)
