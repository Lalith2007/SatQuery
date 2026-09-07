"""Automated Downloader & Tiler for Official WHU-OPT-SAR Dataset.

Downloads official co-registered 5556x3704 Optical, SAR, and Label rasters
from the official Wuhan University Google Drive repository using the verified
file ID manifest, then deterministically tiles them into the 70/15/15
image-level split (zero scene overlap) ready for model training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("download_official_dataset")

# Manifest path relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SCRIPT_DIR / "whu_opt_sar_drive_manifest.json"


def download_single_file(file_id: str, output_path: Path, max_retries: int = 5) -> bool:
    """Download single file from Google Drive using direct streaming requests with resume check."""
    import re
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume / existence check
    if output_path.exists() and output_path.is_file() and output_path.stat().st_size > 10_000:
        return True

    temp_path = output_path.with_suffix(".tmp")
    if temp_path.exists():
        temp_path.unlink()

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    url = "https://drive.google.com/uc?export=download"

    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            resp = session.get(url, params={"id": file_id}, headers=headers, stream=True, timeout=30)
            token = None
            for key, val in resp.cookies.items():
                if key.startswith("download_warning"):
                    token = val
                    break
            if not token and "text/html" in resp.headers.get("Content-Type", ""):
                m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
                if m:
                    token = m.group(1)
                else:
                    token = "t"

            if token:
                resp = session.get(
                    url,
                    params={"id": file_id, "confirm": token},
                    headers=headers,
                    stream=True,
                    timeout=30,
                )

            if resp.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                if temp_path.exists() and temp_path.stat().st_size > 10_000:
                    temp_path.replace(output_path)
                    return True

            raise RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{max_retries} failed for {output_path.name} (id={file_id}): {exc}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return False


def download_and_tile_official_dataset(
    raw_dir: Path = Path("data/raw_whu_opt_sar"),
    output_tiled_dir: Path = Path("data/official_whu_opt_sar"),
    max_pairs: int = 100,
    workers: int = 8,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    zip_backup_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Automated download and tiling of the WHU-OPT-SAR dataset.

    Args:
        raw_dir: Directory to store downloaded raw rasters (optical, sar, labels).
        output_tiled_dir: Directory where tiled 256x256 splits will be saved.
        max_pairs: Maximum number of image pairs to download (default: 100, full dataset).
        workers: Concurrency worker threads for downloading.
        train_ratio: Fraction of scenes for training split (default: 0.70).
        val_ratio: Fraction of scenes for validation split (default: 0.15).
        zip_backup_path: Optional path to save a zipped archive of the tiled dataset.

    Returns:
        Dictionary mapping split name ('train', 'val', 'test') to tile count.
    """
    raw_dir = Path(raw_dir)
    output_tiled_dir = Path(output_tiled_dir)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Drive manifest not found at '{MANIFEST_PATH}'")

    with open(MANIFEST_PATH, "r") as f:
        manifest_data = json.load(f)

    all_pairs = manifest_data.get("pairs", [])
    if not all_pairs:
        raise ValueError("Manifest contains no image pairs.")

    selected_pairs = all_pairs[:max_pairs]
    logger.info(f"Selected {len(selected_pairs)} / {len(all_pairs)} image pairs for download.")

    opt_raw_dir = raw_dir / "optical"
    sar_raw_dir = raw_dir / "sar"
    lbl_raw_dir = raw_dir / "labels"

    for d in [opt_raw_dir, sar_raw_dir, lbl_raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Queue download tasks
    download_tasks = []
    for p in selected_pairs:
        pid = p["pair_id"]
        # Optical
        download_tasks.append((p["optical"]["id"], opt_raw_dir / f"{pid}.tif"))
        # SAR
        download_tasks.append((p["sar"]["id"], sar_raw_dir / f"{pid}.tif"))
        # Labels
        download_tasks.append((p["lbl"]["id"], lbl_raw_dir / f"{pid}.tif"))

    total_files = len(download_tasks)
    logger.info(f"Starting concurrent download of {total_files} files using {workers} workers...")

    completed_files = 0
    failed_files = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(download_single_file, fid, out_p): (fid, out_p)
            for fid, out_p in download_tasks
        }

        for future in as_completed(future_to_task):
            fid, out_p = future_to_task[future]
            try:
                success = future.result()
                if success:
                    completed_files += 1
                else:
                    failed_files.append(out_p.name)
            except Exception as exc:
                failed_files.append(out_p.name)
                logger.error(f"Error downloading {out_p.name}: {exc}")

            if completed_files % 15 == 0 or completed_files == total_files:
                elapsed = time.time() - t0
                pct = (completed_files / total_files) * 100
                logger.info(f"Progress: {completed_files}/{total_files} files ({pct:.1f}%) in {elapsed:.1f}s")

    if failed_files:
        raise RuntimeError(f"Failed to download {len(failed_files)} files: {failed_files[:10]}...")

    logger.info(f"All {completed_files} raw files successfully downloaded to '{raw_dir}'!")

    # Deterministic tiling
    logger.info(f"Starting deterministic tiling into '{output_tiled_dir}' (70/15/15 image-level split)...")
    from specialists.optical_sar.tile_official_dataset import tile_whu_dataset_directory
    tile_counts = tile_whu_dataset_directory(
        raw_dataset_dir=raw_dir,
        output_dir=output_tiled_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        tile_size=256,
    )

    logger.info(f"Tiling complete! Tile counts: {tile_counts}")

    # Optional zip backup creation
    if zip_backup_path:
        zip_backup_path = Path(zip_backup_path)
        zip_backup_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating zip backup archive at '{zip_backup_path}'...")
        temp_zip = zip_backup_path.with_suffix(".tmp.zip")
        if temp_zip.exists():
            temp_zip.unlink()

        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in output_tiled_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path("official_whu_opt_sar") / file_path.relative_to(output_tiled_dir)
                    zf.write(file_path, arcname)

        temp_zip.replace(zip_backup_path)
        logger.info(f"Backup archive created successfully at '{zip_backup_path}' ({zip_backup_path.stat().st_size / (1024**2):.1f} MB)")

    return tile_counts


def main():
    parser = argparse.ArgumentParser(description="Download and tile official WHU-OPT-SAR dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw_whu_opt_sar", help="Path for raw rasters.")
    parser.add_argument("--output-dir", type=str, default="data/official_whu_opt_sar", help="Path for tiled dataset.")
    parser.add_argument("--max-pairs", type=int, default=100, help="Maximum number of pairs to download (default: 100).")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio.")
    parser.add_argument("--zip-backup", type=str, default=None, help="Optional path to create a zip backup.")

    args = parser.parse_args()

    download_and_tile_official_dataset(
        raw_dir=Path(args.raw_dir),
        output_tiled_dir=Path(args.output_dir),
        max_pairs=args.max_pairs,
        workers=args.workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        zip_backup_path=Path(args.zip_backup) if args.zip_backup else None,
    )


if __name__ == "__main__":
    main()
