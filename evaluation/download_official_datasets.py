"""SatQuery AI — Official Benchmark Dataset Downloader.

Automates high-speed, resumable acquisition and verification of all official public
benchmark datasets required for the SIH26167 evaluation:

1. RSVQA-LR (Official 2,000-sample validation partition with embedded rasters)
2. VRSBench (Official Captioning, Grounding, and VQA evaluation partitions + validation imagery)
3. BigEarthNet.txt (Official benchmark partition + Stage 1 held-out verification)
4. CDVQA (128-scene TinyCD visual evidence packages)

Ensures statistically meaningful sample sizes are available on disk with zero synthetic data.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dataset_downloader")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Authoritative dataset specifications
DATASET_SPECS = {
    "rsvqa": {
        "name": "RSVQA-LR Validation Partition",
        "url": "https://huggingface.co/datasets/dmarsili/RSVQA-LR-2k/resolve/main/data/validation-00000-of-00001.parquet",
        "target_file": "data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet",
        "expected_size": 174052999,
        "sample_count": 2000,
        "description": "2,000 real Sentinel-2 validation Q&A pairs with embedded image rasters",
    },
    "vrsbench_cap": {
        "name": "VRSBench Captioning Evaluation Partition",
        "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_Cap.json",
        "target_file": "data/benchmark_samples/vrsbench/VRSBench_EVAL_Cap.json",
        "expected_size": 4809521,
        "sample_count": 9350,
        "description": "Official evaluation partition for remote sensing captioning",
    },
    "vrsbench_grd": {
        "name": "VRSBench Referring Expression Grounding Partition",
        "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_referring.json",
        "target_file": "data/benchmark_samples/vrsbench/VRSBench_EVAL_referring.json",
        "expected_size": 10277680,
        "sample_count": 16159,
        "description": "Official evaluation partition for visual referring grounding",
    },
    "vrsbench_vqa": {
        "name": "VRSBench VQA Evaluation Partition",
        "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/VRSBench_EVAL_vqa.json",
        "target_file": "data/benchmark_samples/vrsbench/VRSBench_EVAL_vqa.json",
        "expected_size": 9358245,
        "sample_count": 37409,
        "description": "Official evaluation partition for complex remote sensing VQA",
    },
    "vrsbench_images": {
        "name": "VRSBench Validation Imagery Archive",
        "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip",
        "target_file": "data/benchmark_samples/vrsbench/Images_val.zip",
        "expected_size": 3976656690,
        "sample_count": 4854,
        "description": "High-resolution optical remote sensing validation scenes (3.79 GB)",
    },
    "bigearthnet": {
        "name": "BigEarthNet.txt Official Benchmark Partition",
        "url": "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet",
        "target_file": "data/benchmark_samples/bigearthnet/BigEarthNet.txt.parquet",
        "expected_size": 466819745,
        "sample_count": 1000,
        "description": "Official multimodal Sentinel-1 and Sentinel-2 benchmark partition",
    },
}


def download_url(url: str, dest: Path, expected_size: Optional[int] = None) -> bool:
    """Download a file with curl if available or streaming urllib with progress logging."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if already fully downloaded
    if dest.exists() and dest.stat().st_size > 1024:
        curr_size = dest.stat().st_size
        if expected_size is None or abs(curr_size - expected_size) < 1024 * 1024:
            logger.info(f"✓ Already downloaded: {dest.name} ({curr_size / (1024 ** 2):.1f} MB)")
            return True

    # 1. Prefer curl for high speed and resume capability
    curl_path = shutil.which("curl")
    if curl_path:
        logger.info(f"Starting download of {dest.name} via curl...")
        cmd = [
            curl_path,
            "-L",
            "--retry", "4",
            "--retry-delay", "3",
            "-C", "-",
            "-o", str(dest),
            url,
        ]
        try:
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0 and dest.exists() and dest.stat().st_size > 1024:
                logger.info(f"✓ Successfully downloaded {dest.name} ({dest.stat().st_size / (1024 ** 2):.1f} MB)")
                return True
            logger.warning(f"curl returned code {res.returncode}, falling back to Python urllib...")
        except Exception as e:
            logger.warning(f"curl invocation error: {e}, falling back to urllib...")

    # 2. Resilient Python urllib chunked streaming fallback
    temp_file = dest.with_suffix(".download.tmp")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SatQuery/1.0",
        "Accept": "*/*",
    }
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=headers)

    t_start = time.time()
    t_last_log = t_start
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=180) as resp, open(temp_file, "wb") as f:
            total_bytes = int(resp.headers.get("Content-Length", 0))
            logger.info(f"Downloading {dest.name} ({total_bytes / (1024 ** 2):.1f} MB)...")
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - t_last_log >= 6.0 or (total_bytes > 0 and downloaded == total_bytes):
                    pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0
                    speed_mb = (downloaded / (1024 ** 2)) / max(0.1, now - t_start)
                    logger.info(
                        f"  -> {dest.name}: {pct:5.1f}% | {downloaded / (1024 ** 2):6.1f} / {total_bytes / (1024 ** 2):6.1f} MB | {speed_mb:5.1f} MB/s"
                    )
                    t_last_log = now

        temp_file.replace(dest)
        logger.info(f"✓ Successfully downloaded {dest.name} ({dest.stat().st_size / (1024 ** 2):.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False


def extract_zip(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip archive with progress reporting."""
    if not zip_path.exists():
        logger.error(f"Zip archive not found: {zip_path}")
        return False

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {zip_path.name} ({zip_path.stat().st_size / (1024 ** 2):.1f} MB) into {extract_dir}...")

    unzip_cmd = shutil.which("unzip")
    if unzip_cmd:
        cmd = [unzip_cmd, "-q", "-o", str(zip_path), "-d", str(extract_dir)]
        try:
            res = subprocess.run(cmd, check=False)
            if res.returncode == 0:
                logger.info(f"✓ Unzip completed successfully via {unzip_cmd}.")
                return True
        except Exception as e:
            logger.warning(f"unzip command failed: {e}, using python zipfile...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            logger.info(f"Extracting {len(members)} files via zipfile...")
            zf.extractall(extract_dir)
        logger.info(f"✓ Extracted {len(members)} files into {extract_dir}.")
        return True
    except Exception as e:
        logger.error(f"Failed extracting {zip_path}: {e}")
        return False


def unpack_rsvqa_parquet_images(parquet_path: Path, output_dir: Path, max_samples: int = 2000) -> int:
    """Unpack raw images from RSVQA-LR parquet into local PNGs for offline access."""
    if not parquet_path.exists():
        return 0
    try:
        import pandas as pd
        from PIL import Image
    except ImportError:
        logger.info("pandas/PIL not imported, parquet will be read directly during evaluation.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_pngs = len(list(output_dir.glob("*.png")))
    if existing_pngs >= min(max_samples, 2000):
        logger.info(f"✓ RSVQA images already unpacked: {existing_pngs} PNGs in {output_dir.name}/.")
        return existing_pngs

    logger.info(f"Unpacking up to {max_samples} satellite images from {parquet_path.name}...")
    try:
        df = pd.read_parquet(parquet_path)
        count = 0
        records = []
        for idx, row in df.iloc[:max_samples].iterrows():
            img_data = row["image"]
            raw_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
            if raw_bytes:
                img_path = output_dir / f"rsvqa_{idx:05d}.png"
                if not img_path.exists():
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img.save(img_path, "PNG")
                records.append({
                    "index": idx,
                    "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                    "question": row["question"],
                    "answer": str(row["answer"]),
                })
                count += 1
                if count % 200 == 0:
                    logger.info(f"  Unpacked {count}/{min(max_samples, len(df))} images...")

        manifest_p = output_dir.parent / "unpacked_manifest.json"
        manifest_p.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info(f"✓ Successfully unpacked {count} RSVQA-LR satellite images to {output_dir}.")
        return count
    except Exception as e:
        logger.warning(f"Could not unpack parquet images ({e}); runner will stream them from parquet directly.")
        return 0


def download_rsvqa(unpack_images: bool = True) -> bool:
    """Download RSVQA-LR official validation partition."""
    spec = DATASET_SPECS["rsvqa"]
    target = PROJECT_ROOT / spec["target_file"]
    ok = download_url(spec["url"], target, spec["expected_size"])
    if ok and unpack_images:
        images_dir = target.parent / "images"
        unpack_rsvqa_parquet_images(target, images_dir, max_samples=spec["sample_count"])
    return ok


def download_vrsbench(include_images: bool = True) -> bool:
    """Download VRSBench partitions and optionally validation imagery."""
    vrs_dir = PROJECT_ROOT / "data/benchmark_samples/vrsbench"
    vrs_dir.mkdir(parents=True, exist_ok=True)

    for key in ["vrsbench_cap", "vrsbench_grd", "vrsbench_vqa"]:
        spec = DATASET_SPECS[key]
        target = PROJECT_ROOT / spec["target_file"]
        ok = download_url(spec["url"], target, spec["expected_size"])
        if not ok:
            logger.error(f"Failed downloading {spec['name']}")
            return False

    if include_images:
        img_dir = vrs_dir / "Images_val"
        existing_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        if existing_imgs >= 100:
            logger.info(f"✓ VRSBench validation images already extracted: {existing_imgs} scenes in {img_dir.name}/.")
            return True

        spec_img = DATASET_SPECS["vrsbench_images"]
        zip_target = PROJECT_ROOT / spec_img["target_file"]
        ok = download_url(spec_img["url"], zip_target, spec_img["expected_size"])
        if ok and zip_target.exists():
            extract_zip(zip_target, vrs_dir)
            final_count = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            logger.info(f"✓ Extracted {final_count} VRSBench validation images.")
    return True


def download_bigearthnet() -> bool:
    """Download BigEarthNet.txt official benchmark partition."""
    spec = DATASET_SPECS["bigearthnet"]
    target = PROJECT_ROOT / spec["target_file"]
    return download_url(spec["url"], target, spec["expected_size"])


def check_dataset_status() -> Dict[str, Any]:
    """Check and display local availability of all official benchmark datasets."""
    logger.info("=" * 75)
    logger.info("SATQUERY AI — OFFICIAL BENCHMARK DATASET INTEGRITY CHECK")
    logger.info("=" * 75)

    status_report = {}
    for key, spec in DATASET_SPECS.items():
        p = PROJECT_ROOT / spec["target_file"]
        exists = p.exists()
        sz_mb = round(p.stat().st_size / (1024 ** 2), 2) if exists else 0.0
        pct = round((p.stat().st_size / spec["expected_size"]) * 100, 1) if exists and spec.get("expected_size") else 0.0
        status = "READY" if (exists and (spec.get("expected_size") is None or pct >= 98.0)) else "MISSING"

        status_report[key] = {
            "name": spec["name"],
            "target": str(p.relative_to(PROJECT_ROOT)),
            "size_mb": sz_mb,
            "status": status,
            "sample_count": spec["sample_count"],
        }
        symbol = "✓" if status == "READY" else "✗"
        logger.info(f"  [{symbol}] {spec['name']:<42} | {sz_mb:7.1f} MB | {status}")

    vrs_imgs = PROJECT_ROOT / "data/benchmark_samples/vrsbench/Images_val"
    vrs_count = len(list(vrs_imgs.glob("*.png"))) if vrs_imgs.exists() else 0
    logger.info(f"  [{'✓' if vrs_count > 0 else '✗'}] VRSBench Extracted Images (Images_val/)    | {vrs_count:5d} files | {'READY' if vrs_count > 0 else 'PENDING EXTRACT'}")

    ben_test = PROJECT_ROOT / "data/qwen_dataset/test.jsonl"
    ben_count = sum(1 for _ in open(ben_test, "r", encoding="utf-8")) if ben_test.exists() else 0
    logger.info(f"  [{'✓' if ben_count >= 850 else '✗'}] BigEarthNet Stage-1 Held-Out Split        | {ben_count:5d} lines | {'READY' if ben_count >= 850 else 'MISSING'}")

    cdvqa_manifest = PROJECT_ROOT / "reports/final_sih_evaluation/qwen/cdvqa/cdvqa_evidence_manifest.jsonl"
    cdvqa_count = sum(1 for _ in open(cdvqa_manifest, "r", encoding="utf-8")) if cdvqa_manifest.exists() else 0
    logger.info(f"  [{'✓' if cdvqa_count >= 128 else '✗'}] CDVQA TinyCD Visual Evidence Packages     | {cdvqa_count:5d} scenes| {'READY' if cdvqa_count >= 128 else 'MISSING'}")

    logger.info("=" * 75)
    return status_report


def main():
    parser = argparse.ArgumentParser(description="Official Benchmark Dataset Downloader")
    parser.add_argument("--all", action="store_true", help="Download all official benchmark datasets")
    parser.add_argument("--rsvqa", action="store_true", help="Download RSVQA-LR official validation partition")
    parser.add_argument("--vrsbench", action="store_true", help="Download VRSBench partitions and imagery")
    parser.add_argument("--vrsbench-no-images", action="store_true", help="Download VRSBench annotations only (skip 3.79 GB zip)")
    parser.add_argument("--bigearthnet", action="store_true", help="Download BigEarthNet.txt official benchmark partition")
    parser.add_argument("--check-only", action="store_true", help="Audit local datasets without downloading")
    args = parser.parse_args()

    if args.check_only:
        check_dataset_status()
        return

    if not (args.rsvqa or args.vrsbench or args.bigearthnet or args.all):
        args.all = True

    logger.info("=" * 75)
    logger.info("STARTING OFFICIAL BENCHMARK DATASET DOWNLOAD WORKFLOW")
    logger.info("=" * 75)

    if args.all or args.rsvqa:
        logger.info(">>> [1/3] Downloading RSVQA-LR Dataset...")
        download_rsvqa(unpack_images=True)

    if args.all or args.vrsbench:
        logger.info(">>> [2/3] Downloading VRSBench Dataset...")
        include_imgs = not args.vrsbench_no_images
        download_vrsbench(include_images=include_imgs)

    if args.all or args.bigearthnet:
        logger.info(">>> [3/3] Downloading BigEarthNet.txt Dataset...")
        download_bigearthnet()

    logger.info("Download workflow completed. Verifying on-disk availability:")
    check_dataset_status()


if __name__ == "__main__":
    main()
