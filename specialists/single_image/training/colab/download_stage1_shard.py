"""Colab & Local Step: Deterministic Stage-1 BigEarthNet.txt Shard Downloader & Packager.

Ensures BIFOLD-BigEarthNetv2-0/BigEarthNet.txt metadata is acquired and generates the
locked 8,000-pair / 16,000-example Stage-1 training shard with 100% split integrity.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
from huggingface_hub import hf_hub_download

from specialists.single_image.training.colab.generate_stage1_shard import generate_stage1_shard


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_and_generate_stage1(
    cache_dir: str = "data/cache_ben",
    manifest_output: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
    report_output: str = "bigearthnet_stage1_sampling_report.json",
    repo_id: str = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    filename: str = "BigEarthNet.txt.parquet",
) -> Path:
    """Download Parquet if missing, then generate the Stage-1 shard."""
    c_dir = Path(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = c_dir / filename

    if not parquet_path.exists():
        print(f"Downloading {filename} from {repo_id} to {parquet_path}...")
        t0 = time.perf_counter()
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(c_dir),
        )
        parquet_path = Path(downloaded)
        print(f"Download complete in {time.perf_counter() - t0:.2f}s.")
    else:
        print(f"Found cached Parquet at: {parquet_path}")

    # Compute Parquet Checksum
    pq_sha = compute_sha256(parquet_path)
    print(f"Source Parquet SHA-256: {pq_sha}")

    # Generate Shard
    print("Executing deterministic Stage-1 sampling...")
    report = generate_stage1_shard(
        parquet_path=str(parquet_path),
        output_manifest_path=manifest_output,
        output_report_path=report_output,
        target_pairs=8000,
        seed=42,
    )

    manifest_p = Path(manifest_output)
    manifest_sha = compute_sha256(manifest_p)
    print(f"Stage-1 Manifest generated: {manifest_p} ({manifest_p.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Manifest SHA-256: {manifest_sha}")

    return manifest_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and generate Stage-1 BigEarthNet shard")
    parser.add_argument("--cache_dir", default="data/cache_ben")
    parser.add_argument("--manifest_output", default="data/curated_mixture/bigearthnet_stage1_manifest.jsonl")
    parser.add_argument("--report_output", default="bigearthnet_stage1_sampling_report.json")
    args = parser.parse_args()

    download_and_generate_stage1(
        cache_dir=args.cache_dir,
        manifest_output=args.manifest_output,
        report_output=args.report_output,
    )
