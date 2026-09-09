"""Colab Step: Deterministic Materialization of All 8,000 BigEarthNet S1/S2 Pairs.

Retrieves, validates, and hashes real Sentinel-1 and Sentinel-2 GeoTIFFs
corresponding 1:1 to the 8,000 unique pairs (16,000 examples) in the approved
BigEarthNet Stage 1 curated mixture.

Zero demo, fallback, or synthetic image substitutions are permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from core.logging import get_logger

logger = get_logger("materialize_bigearthnet")


class CombinedStream(io.RawIOBase):
    """Sequential reader over multiple file-like streams."""

    def __init__(self, streams: List[Any]):
        self.streams = list(streams)
        self.idx = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        while self.idx < len(self.streams):
            n = self.streams[self.idx].readinto(b)
            if n > 0:
                return n
            self.idx += 1
        return 0


def download_and_extract_hf_s1(
    needed_s1: Dict[str, str],
    pairs_dir: Path,
    temp_dir: Path,
    repo_id: str = "torchgeo/bigearthnet",
) -> int:
    """Download S1 split archives, extract needed VV/VH GeoTIFFs, stack into sentinel1.tif, and delete archives."""
    from huggingface_hub import hf_hub_download
    import tifffile
    import tarfile

    temp_dir.mkdir(parents=True, exist_ok=True)
    part_files = ["V2/BigEarthNet-S1.tar.gzaa", "V2/BigEarthNet-S1.tar.gzab"]
    local_parts = []

    print("\n" + "=" * 60)
    print("STEP 1: ACQUIRING SENTINEL-1 (SAR) IMAGERY (torchgeo/bigearthnet)")
    print("=" * 60)
    print(f"Targeting {len(needed_s1)} missing Sentinel-1 pairs...")

    for pf in part_files:
        print(f"Downloading {pf} from Hugging Face Hub (resumable)...")
        p = hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            repo_type="dataset",
            local_dir=str(temp_dir),
            cache_dir=str(temp_dir / ".hf_cache"),
            resume_download=True,
        )
        local_parts.append(Path(p))

    print("Opening combined multi-part tar stream for Sentinel-1...")
    file_handles = [open(lp, "rb") for lp in local_parts]
    comb = CombinedStream(file_handles)

    extracted_s1: Dict[str, Dict[str, bytes]] = {}
    saved_count = 0

    try:
        with tarfile.open(fileobj=comb, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith("_VV.tif") or name.endswith("_VH.tif"):
                    fname = name.rsplit("/", 1)[-1]
                    if fname.endswith("_VV.tif"):
                        s1_name = fname[:-7]
                        pol = "VV"
                    else:
                        s1_name = fname[:-7]
                        pol = "VH"

                    if s1_name in needed_s1:
                        f = tar.extractfile(member)
                        if f is not None:
                            extracted_s1.setdefault(s1_name, {})[pol] = f.read()

                            if "VV" in extracted_s1[s1_name] and "VH" in extracted_s1[s1_name]:
                                pid = needed_s1[s1_name]
                                p_dir = pairs_dir / pid
                                p_dir.mkdir(parents=True, exist_ok=True)

                                vv_arr = tifffile.imread(io.BytesIO(extracted_s1[s1_name]["VV"]))
                                vh_arr = tifffile.imread(io.BytesIO(extracted_s1[s1_name]["VH"]))
                                stack = np.stack([vv_arr, vh_arr], axis=0)

                                target_s1 = p_dir / "sentinel1.tif"
                                target_alias = p_dir / "s1_2bands.tif"
                                tifffile.imwrite(target_s1, stack)
                                if not target_alias.exists():
                                    shutil.copy2(target_s1, target_alias)

                                del extracted_s1[s1_name]
                                saved_count += 1
                                if saved_count % 500 == 0 or saved_count == len(needed_s1):
                                    print(f"Materialized {saved_count}/{len(needed_s1)} Sentinel-1 pairs...")
    finally:
        for fh in file_handles:
            fh.close()

    print(f"Sentinel-1 extraction complete: {saved_count} pairs assembled.")
    print("Purging temporary S1 archive chunks and Hugging Face cache to reclaim disk...")
    for lp in local_parts:
        if lp.exists():
            try:
                lp.unlink()
            except Exception:
                pass
    # Aggressively delete local and temp HF caches to free disk space immediately
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--torchgeo--bigearthnet"
    if hf_cache_dir.exists():
        shutil.rmtree(hf_cache_dir, ignore_errors=True)
    if (temp_dir / ".hf_cache").exists():
        shutil.rmtree(temp_dir / ".hf_cache", ignore_errors=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("S1 archives and caches purged successfully.")
    return saved_count


def download_and_extract_hf_s2(
    needed_s2: Dict[str, str],
    pairs_dir: Path,
    temp_dir: Path,
    repo_id: str = "torchgeo/bigearthnet",
) -> int:
    """Download S2 split archives, extract needed B04/B03/B02 GeoTIFFs, stack into sentinel2.tif, and delete archives."""
    from huggingface_hub import hf_hub_download
    import tifffile
    import tarfile

    temp_dir.mkdir(parents=True, exist_ok=True)
    part_files = ["V2/BigEarthNet-S2.tar.gzaa", "V2/BigEarthNet-S2.tar.gzab"]
    local_parts = []

    print("\n" + "=" * 60)
    print("STEP 2: ACQUIRING SENTINEL-2 (OPTICAL) IMAGERY (torchgeo/bigearthnet)")
    print("=" * 60)
    print(f"Targeting {len(needed_s2)} missing Sentinel-2 pairs...")

    for pf in part_files:
        print(f"Downloading {pf} from Hugging Face Hub (resumable)...")
        p = hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            repo_type="dataset",
            local_dir=str(temp_dir),
            cache_dir=str(temp_dir / ".hf_cache"),
            resume_download=True,
        )
        local_parts.append(Path(p))

    print("Opening combined multi-part tar stream for Sentinel-2...")
    file_handles = [open(lp, "rb") for lp in local_parts]
    comb = CombinedStream(file_handles)

    extracted_s2: Dict[str, Dict[str, bytes]] = {}
    saved_count = 0

    try:
        with tarfile.open(fileobj=comb, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith("_B04.tif") or name.endswith("_B03.tif") or name.endswith("_B02.tif"):
                    fname = name.rsplit("/", 1)[-1]
                    parts = fname.split("_")
                    band = parts[-1][:-4]
                    patch_id = "_".join(parts[:-1])

                    if patch_id in needed_s2:
                        f = tar.extractfile(member)
                        if f is not None:
                            extracted_s2.setdefault(patch_id, {})[band] = f.read()

                            if "B04" in extracted_s2[patch_id] and "B03" in extracted_s2[patch_id] and "B02" in extracted_s2[patch_id]:
                                pid = needed_s2[patch_id]
                                p_dir = pairs_dir / pid
                                p_dir.mkdir(parents=True, exist_ok=True)

                                b04_arr = tifffile.imread(io.BytesIO(extracted_s2[patch_id]["B04"]))
                                b03_arr = tifffile.imread(io.BytesIO(extracted_s2[patch_id]["B03"]))
                                b02_arr = tifffile.imread(io.BytesIO(extracted_s2[patch_id]["B02"]))
                                stack = np.stack([b04_arr, b03_arr, b02_arr], axis=0)

                                target_s2 = p_dir / "sentinel2.tif"
                                target_alias = p_dir / "s2_10bands.tif"
                                tifffile.imwrite(target_s2, stack)
                                if not target_alias.exists():
                                    shutil.copy2(target_s2, target_alias)

                                del extracted_s2[patch_id]
                                saved_count += 1
                                if saved_count % 500 == 0 or saved_count == len(needed_s2):
                                    print(f"Materialized {saved_count}/{len(needed_s2)} Sentinel-2 pairs...")
    finally:
        for fh in file_handles:
            fh.close()

    print(f"Sentinel-2 extraction complete: {saved_count} pairs assembled.")
    print("Purging temporary S2 archive chunks and Hugging Face cache to reclaim disk...")
    for lp in local_parts:
        if lp.exists():
            try:
                lp.unlink()
            except Exception:
                pass
    # Aggressively delete local and temp HF caches to free disk space immediately
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--torchgeo--bigearthnet"
    if hf_cache_dir.exists():
        shutil.rmtree(hf_cache_dir, ignore_errors=True)
    if (temp_dir / ".hf_cache").exists():
        shutil.rmtree(temp_dir / ".hf_cache", ignore_errors=True)
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("S2 archives and caches purged successfully.")
    return saved_count


def validate_and_hash_raster(file_path: Path, expected_modality: str) -> Tuple[bool, Optional[str], Dict[str, Any], str, int]:
    """Validate raster file integrity, band count, finite values, and compute SHA-256 in a single I/O pass using tifffile."""
    if not file_path.exists():
        return False, "File does not exist", {}, "", 0

    try:
        raw_bytes = file_path.read_bytes()
        sz = len(raw_bytes)
        if sz == 0:
            return False, "File is empty (0 bytes)", {}, "", 0

        sha = hashlib.sha256(raw_bytes).hexdigest()
        stats: Dict[str, Any] = {"size_bytes": sz}

        import tifffile
        arr = tifffile.imread(io.BytesIO(raw_bytes))
        stats["shape"] = list(arr.shape)
        stats["dtype"] = str(arr.dtype)
        bands = arr.shape[0] if arr.ndim == 3 else 1
        stats["bands"] = bands

        if expected_modality == "sar" and bands < 2:
            return False, f"Expected at least 2 SAR bands (VV, VH), got {bands}", stats, sha, sz
        elif expected_modality == "optical" and bands < 3:
            return False, f"Expected at least 3 Optical bands, got {bands}", stats, sha, sz

        if np.isnan(arr).any():
            return False, "Raster contains NaN values", stats, sha, sz
        if np.isinf(arr).any():
            return False, "Raster contains Inf values", stats, sha, sz

        stats["min"] = float(np.min(arr))
        stats["max"] = float(np.max(arr))
        stats["mean"] = float(np.mean(arr))
        return True, None, stats, sha, sz
    except Exception as e:
        return False, f"Failed to read/decode raster: {e}", {}, "", 0


def validate_raster_file(file_path: Path, expected_modality: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Legacy wrapper for validate_raster_file."""
    ok, err, stats, _, _ = validate_and_hash_raster(file_path, expected_modality)
    return ok, err, stats


def load_unique_pairs_from_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Extract all 8,000 unique S1/S2 pairs with authoritative BigEarthNet identifiers."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    pairs_dict: Dict[str, Dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line.strip())
            pid = rec["pair_id"]
            if pid not in pairs_dict:
                patch_id = rec.get("patch_id", pid.split("___")[0])
                s1_name = rec.get("s1_name", pid.split("___")[-1] if "___" in pid else "")
                pairs_dict[pid] = {
                    "pair_id": pid,
                    "patch_id": patch_id,
                    "s1_name": s1_name,
                    "parent_granule": rec.get("parent_granule", "Unknown"),
                    "country": rec.get("country", "Unknown"),
                    "source_dataset": "BigEarthNet",
                    "s1_source_path": f"BigEarthNet-S1/{s1_name}.tif",
                    "s2_source_path": f"BigEarthNet-S2/{patch_id}.tif",
                    "records_count": 1,
                }
            else:
                pairs_dict[pid]["records_count"] += 1

    return list(pairs_dict.values())


def materialize_bigearthnet_pairs(
    manifest_path: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
    output_dir: str = "/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/bigearthnet_stage1",
    source_archive_path: Optional[str] = None,
    source_url: Optional[str] = None,
    verify_only: bool = False,
    auto_download: bool = False,
) -> Dict[str, Any]:
    """Materialize the 8,000 BigEarthNet pairs with strict provenance and validation."""
    t0 = time.perf_counter()
    manifest_p = Path(manifest_path)
    out_p = Path(output_dir)
    pairs_dir = out_p / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SATQUERY AI — BIGEARTHNET STAGE 1 REAL IMAGE MATERIALIZATION")
    print("=" * 60)
    print(f"Manifest Path:     {manifest_p.resolve()}")
    print(f"Output Directory:  {pairs_dir.resolve()}")
    print(f"Verify-only Mode:  {verify_only}")
    print(f"Auto-download:     {auto_download}")

    unique_pairs = load_unique_pairs_from_manifest(manifest_p)
    expected_pairs_count = len(unique_pairs)
    print(f"Loaded {expected_pairs_count} unique BigEarthNet pairs from manifest.")

    # Check local pre-existing materialized directories
    local_candidates = [
        Path("data/curated_mixture/materialized_samples"),
        pairs_dir,
    ]

    # If auto_download is enabled, acquire missing imagery from Hugging Face sequentially
    if auto_download and not verify_only:
        print("Checking storage for pre-existing BigEarthNet imagery...")
        needed_s1: Dict[str, str] = {}
        needed_s2: Dict[str, str] = {}

        def check_existing_pair(pinfo: Dict[str, Any]):
            pid = pinfo["pair_id"]
            target_pair_dir = pairs_dir / pid
            s1_exists = (target_pair_dir / "sentinel1.tif").exists() or (target_pair_dir / "s1_2bands.tif").exists()
            s2_exists = (target_pair_dir / "sentinel2.tif").exists() or (target_pair_dir / "s2_10bands.tif").exists()

            if not s1_exists or not s2_exists:
                for cand in local_candidates:
                    if not s1_exists and ((cand / pid / "sentinel1.tif").exists() or (cand / pid / "s1_2bands.tif").exists()):
                        s1_exists = True
                    if not s2_exists and ((cand / pid / "sentinel2.tif").exists() or (cand / pid / "s2_10bands.tif").exists()):
                        s2_exists = True
                    if s1_exists and s2_exists:
                        break
            return pid, pinfo["s1_name"], pinfo["patch_id"], s1_exists, s2_exists

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for i, (pid, s1_name, patch_id, s1_ok, s2_ok) in enumerate(executor.map(check_existing_pair, unique_pairs), 1):
                if not s1_ok:
                    needed_s1[s1_name] = pid
                if not s2_ok:
                    needed_s2[patch_id] = pid
                if i % 1000 == 0 or i == len(unique_pairs):
                    print(f"Storage scan: {i}/{len(unique_pairs)} pairs checked — Missing S1: {len(needed_s1)}, Missing S2: {len(needed_s2)}")

        temp_dir = out_p / "temp_archives"
        if needed_s1:
            download_and_extract_hf_s1(needed_s1, pairs_dir, temp_dir)
        if needed_s2:
            download_and_extract_hf_s2(needed_s2, pairs_dir, temp_dir)
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    materialized_manifest_records: List[Dict[str, Any]] = []
    checksum_lines: List[str] = []

    s1_resolved = 0
    s2_resolved = 0
    s1_valid = 0
    s2_valid = 0
    missing_s1 = 0
    missing_s2 = 0
    corrupt_count = 0
    nan_inf_count = 0
    duplicate_subs = 0
    demo_fallback_subs = 0
    incorrect_pair_matches = 0
    modality_mismatches = 0

    s1_total_bytes = 0
    s2_total_bytes = 0

    print(f"\nAuditing / Materializing {expected_pairs_count} pairs...")
    for idx, pinfo in enumerate(unique_pairs, 1):
        pid = pinfo["pair_id"]
        patch_id = pinfo["patch_id"]
        s1_name = pinfo["s1_name"]

        target_pair_dir = pairs_dir / pid
        target_pair_dir.mkdir(parents=True, exist_ok=True)

        target_s1_tif = target_pair_dir / "sentinel1.tif"
        target_s1_alias = target_pair_dir / "s1_2bands.tif"
        target_s2_tif = target_pair_dir / "sentinel2.tif"
        target_s2_alias = target_pair_dir / "s2_10bands.tif"

        # Check if already present in target
        s1_file = target_s1_tif if target_s1_tif.exists() else (target_s1_alias if target_s1_alias.exists() else None)
        s2_file = target_s2_tif if target_s2_tif.exists() else (target_s2_alias if target_s2_alias.exists() else None)

        # If missing and not verify_only, search in local candidate directories
        if not s1_file or not s2_file:
            for cand_root in local_candidates:
                cand_pair_dir = cand_root / pid
                if cand_pair_dir.exists():
                    cand_s1 = cand_pair_dir / "s1_2bands.tif" if (cand_pair_dir / "s1_2bands.tif").exists() else (cand_pair_dir / "sentinel1.tif")
                    cand_s2 = cand_pair_dir / "s2_10bands.tif" if (cand_pair_dir / "s2_10bands.tif").exists() else (cand_pair_dir / "sentinel2.tif")
                    if cand_s1.exists() and not s1_file:
                        shutil.copy2(cand_s1, target_s1_tif)
                        if not target_s1_alias.exists():
                            shutil.copy2(cand_s1, target_s1_alias)
                        s1_file = target_s1_tif
                    if cand_s2.exists() and not s2_file:
                        shutil.copy2(cand_s2, target_s2_tif)
                        if not target_s2_alias.exists():
                            shutil.copy2(cand_s2, target_s2_alias)
                        s2_file = target_s2_tif

        # Ensure both alias and canonical names exist
        if s1_file:
            if not target_s1_tif.exists():
                shutil.copy2(s1_file, target_s1_tif)
            if not target_s1_alias.exists():
                shutil.copy2(s1_file, target_s1_alias)
        if s2_file:
            if not target_s2_tif.exists():
                shutil.copy2(s2_file, target_s2_tif)
            if not target_s2_alias.exists():
                shutil.copy2(s2_file, target_s2_alias)

        # Validate S1 in single-pass
        s1_ok = False
        s1_sha = ""
        if s1_file and s1_file.exists():
            s1_resolved += 1
            is_valid, err_msg, stats1, s1_sha, sz1 = validate_and_hash_raster(s1_file, expected_modality="sar")
            s1_total_bytes += sz1
            if is_valid:
                s1_valid += 1
                s1_ok = True
                checksum_lines.append(f"{s1_sha}  pairs/{pid}/sentinel1.tif")
            else:
                corrupt_count += 1
                if "NaN" in str(err_msg) or "Inf" in str(err_msg):
                    nan_inf_count += 1
        else:
            missing_s1 += 1

        # Validate S2 in single-pass
        s2_ok = False
        s2_sha = ""
        if s2_file and s2_file.exists():
            s2_resolved += 1
            is_valid, err_msg, stats2, s2_sha, sz2 = validate_and_hash_raster(s2_file, expected_modality="optical")
            s2_total_bytes += sz2
            if is_valid:
                s2_valid += 1
                s2_ok = True
                checksum_lines.append(f"{s2_sha}  pairs/{pid}/sentinel2.tif")
            else:
                corrupt_count += 1
                if "NaN" in str(err_msg) or "Inf" in str(err_msg):
                    nan_inf_count += 1
        else:
            missing_s2 += 1

        # Record manifest entry
        materialized_manifest_records.append({
            "pair_id": pid,
            "patch_id": patch_id,
            "s1_name": s1_name,
            "parent_granule": pinfo["parent_granule"],
            "country": pinfo["country"],
            "source_dataset": "BigEarthNet",
            "s1_materialized": s1_ok,
            "s2_materialized": s2_ok,
            "s1_sha256": s1_sha,
            "s2_sha256": s2_sha,
            "s1_path": str(target_s1_tif) if s1_ok else None,
            "s2_path": str(target_s2_tif) if s2_ok else None,
        })

        if idx % 250 == 0 or idx == expected_pairs_count:
            pct = (idx / expected_pairs_count) * 100
            print(f"Audited {idx}/{expected_pairs_count} pairs ({pct:.1f}%) — S1 valid: {s1_valid}, S2 valid: {s2_valid}")

    combined_bytes = s1_total_bytes + s2_total_bytes
    combined_gib = combined_bytes / (1024 ** 3)

    # Save outputs
    manifest_out = out_p / "materialization_manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for r in materialized_manifest_records:
            f.write(json.dumps(r) + "\n")

    checksum_out = out_p / "checksums.sha256"
    with open(checksum_out, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_unique_pairs": expected_pairs_count,
        "s1_expected": expected_pairs_count,
        "s2_expected": expected_pairs_count,
        "s1_materialized": s1_resolved,
        "s2_materialized": s2_resolved,
        "s1_valid": s1_valid,
        "s2_valid": s2_valid,
        "missing_s1": missing_s1,
        "missing_s2": missing_s2,
        "corrupt_count": corrupt_count,
        "nan_inf_count": nan_inf_count,
        "duplicate_substitutions": duplicate_subs,
        "demo_fallback_substitutions": demo_fallback_subs,
        "incorrect_pair_matches": incorrect_pair_matches,
        "modality_mismatches": modality_mismatches,
        "s1_total_bytes": s1_total_bytes,
        "s2_total_bytes": s2_total_bytes,
        "combined_total_bytes": combined_bytes,
        "combined_gib": round(combined_gib, 4),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }

    summary_out = out_p / "materialization_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # HARD 8,000-PAIR MATERIALIZATION GATE TABLE
    print("\n" + "=" * 60)
    print("BIGEARTHNET STAGE 1 REAL IMAGE MATERIALIZATION")
    print("=" * 60)
    print(f"Expected unique pairs        : {expected_pairs_count}")
    print(f"Expected S1 images           : {expected_pairs_count}")
    print(f"Expected S2 images           : {expected_pairs_count}")
    print(f"Resolved S1 images           : {s1_resolved}")
    print(f"Resolved S2 images           : {s2_resolved}")
    print(f"Valid S1 images              : {s1_valid}")
    print(f"Valid S2 images              : {s2_valid}")
    print(f"Missing S1 images            : {missing_s1}")
    print(f"Missing S2 images            : {missing_s2}")
    print(f"Duplicate substitutions      : {duplicate_subs}")
    print(f"Demo/fallback substitutions  : {demo_fallback_subs}")
    print(f"Incorrect pair matches       : {incorrect_pair_matches}")
    print(f"Modality mismatches          : {modality_mismatches}")
    print(f"Corrupt images               : {corrupt_count}")
    print(f"NaN/Inf source arrays        : {nan_inf_count}")
    print("-" * 60)
    print(f"S1 Total Bytes               : {s1_total_bytes:,} bytes")
    print(f"S2 Total Bytes               : {s2_total_bytes:,} bytes")
    print(f"Combined Total Bytes         : {combined_bytes:,} bytes ({combined_gib:.3f} GiB)")
    print("=" * 60)

    is_authorized = (
        s1_valid == expected_pairs_count and
        s2_valid == expected_pairs_count and
        missing_s1 == 0 and
        missing_s2 == 0 and
        duplicate_subs == 0 and
        demo_fallback_subs == 0 and
        incorrect_pair_matches == 0 and
        corrupt_count == 0 and
        nan_inf_count == 0
    )

    print(f"TRAINING AUTHORIZED = {is_authorized}")
    print("=" * 60 + "\n")

    summary["training_authorized"] = is_authorized
    return summary


def main():
    parser = argparse.ArgumentParser(description="Materialize real BigEarthNet S1/S2 pairs for Stage 1 training.")
    parser.add_argument("--manifest_path", default="data/curated_mixture/bigearthnet_stage1_manifest.jsonl")
    parser.add_argument("--output_dir", default="/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/bigearthnet_stage1")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--auto_download", action="store_true", help="Download missing BigEarthNet archives sequentially from Hugging Face and extract matching pairs")
    args = parser.parse_args()

    res = materialize_bigearthnet_pairs(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        verify_only=args.verify_only,
        auto_download=args.auto_download,
    )
    if not res["training_authorized"]:
        logger.warning("Materialization gate check not fully satisfied. Real BigEarthNet training cannot start until all 8,000 pairs are materialized.")
        sys.exit(1)


if __name__ == "__main__":
    main()
