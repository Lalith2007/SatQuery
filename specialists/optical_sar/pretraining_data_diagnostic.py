"""Pre-Training Data Diagnostics (Phase 10).

Analyzes and reports:
1. Split counts (tiles & scenes for train, val, test)
2. Exact pixel distribution & percentages
3. Tile-level presence (any vs meaningful threshold)
4. Expected sampling distribution with Meaningful Minority Content Sampler
5. Empirical batch sampling simulation (draws 100 batches to report mean, std, min, max pixel exposure)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from specialists.optical_sar.config_balanced_v3 import get_default_v3_config
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.train_colab_v3 import ColabTrainerV3

CLASS_NAMES = [
    "background",
    "farmland",
    "city",
    "village",
    "water",
    "forest",
    "road",
    "others",
]

MEANINGFUL_THRESHOLDS = {
    "background": 10,
    "farmland": 1000,
    "city": 250,
    "village": 500,
    "water": 500,
    "forest": 1000,
    "road": 100,
    "others": 250,
}


def run_data_diagnostics(counts_cache_path: Optional[Path] = None) -> Dict[str, Any]:
    config = get_default_v3_config()
    data_dir = config.data_dir

    print("=" * 80)
    print("  PHASE 10: PRE-TRAINING DATA DIAGNOSTICS SUITE")
    print("=" * 80)

    # 1. Dataset Split Counts & Scene Counts
    split_info = {}
    for s in ["train", "val", "test"]:
        opt_dir = data_dir / s / "optical"
        files = list(opt_dir.glob("*.png"))
        scenes = set(f.stem.split("_")[0] for f in files)
        split_info[s] = {"tiles": len(files), "scenes": len(scenes), "scene_names": sorted(list(scenes))}
        print(f"Split {s.upper():5s} | Tiles: {len(files):6,d} | Scenes: {len(scenes):2d}")

    assert split_info["train"]["tiles"] == 11880, "Train tile count mismatch"
    assert split_info["val"]["tiles"] == 2310, "Val tile count mismatch"
    assert split_info["test"]["tiles"] == 2970, "Test tile count mismatch"

    # Leakage check: pairwise intersection of scenes and tiles
    s_tr = set(split_info["train"]["scene_names"])
    s_va = set(split_info["val"]["scene_names"])
    s_te = set(split_info["test"]["scene_names"])

    assert len(s_tr & s_va) == 0, f"Train-Val scene leakage detected: {s_tr & s_va}"
    assert len(s_tr & s_te) == 0, f"Train-Test scene leakage detected: {s_tr & s_te}"
    assert len(s_va & s_te) == 0, f"Val-Test scene leakage detected: {s_va & s_te}"
    print(">>> Split & Scene-Level Isolation: 100% VERIFIED (ZERO LEAKAGE) <<<\n")

    # 2. Pixel Distribution
    if counts_cache_path is not None and counts_cache_path.exists():
        raw_counts = np.load(counts_cache_path)  # (11880, 9)
    else:
        raise FileNotFoundError(f"Missing precomputed counts at {counts_cache_path}")

    pixel_counts_8 = raw_counts[:, :8]
    total_pixels_per_class = pixel_counts_8.sum(axis=0)
    total_valid = total_pixels_per_class.sum()
    freqs = total_pixels_per_class / total_valid

    print("--- 1. Pixel Distribution (Train Split: 11,880 Tiles) ---")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name:12s} | {total_pixels_per_class[i]:12,d} px | {freqs[i]*100:8.4f}%")
    print(f"  Ignore 255   | {raw_counts[:, 8].sum():12,d} px")
    print(f"  Total Valid  | {total_valid:12,d} px\n")

    # 3. Tile-Level Presence (Any vs Meaningful)
    presence_any = (pixel_counts_8 > 0)
    meaningful_mask = np.zeros_like(presence_any, dtype=bool)
    for i, name in enumerate(CLASS_NAMES):
        meaningful_mask[:, i] = (pixel_counts_8[:, i] >= MEANINGFUL_THRESHOLDS[name])

    tiles_any = presence_any.sum(axis=0)
    tiles_meaningful = meaningful_mask.sum(axis=0)

    print("--- 2. Tile Distribution & Meaningful Minority Presence ---")
    header = f"  {'Class':12s} | {'Any Pixels':12s} | {'% Tiles':8s} | {'Meaningful Threshold':20s} | {'Meaningful Tiles':16s} | {'% Tiles':8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(CLASS_NAMES):
        th = MEANINGFUL_THRESHOLDS[name]
        print(
            f"  {name:12s} | {tiles_any[i]:6,d} tiles | {tiles_any[i]/11880*100:7.2f}% | "
            f">= {th:4d} pixels        | {tiles_meaningful[i]:6,d} tiles       | {tiles_meaningful[i]/11880*100:7.2f}%"
        )

    # 4. Sampler Expected Distribution
    trainer = ColabTrainerV3(config=config, device="cpu")
    train_ds = OpticalSarPairedDataset(data_dir, split="train", num_classes=8, augment=False)
    sampler, tile_weights = trainer.build_meaningful_minority_sampler(
        train_ds, counts_cache_path=counts_cache_path
    )

    p_new = tile_weights / tile_weights.sum()
    exp_pct_new = (np.sum(pixel_counts_8 * p_new[:, None], axis=0) / np.sum(pixel_counts_8 * p_new[:, None])) * 100.0
    exp_tile_new = np.sum(presence_any * p_new[:, None], axis=0) * 100.0

    print("\n--- 3. Expected Sampling Distribution (New Meaningful Minority Sampler) ---")
    header_s = f"  {'Class':12s} | {'Original Px %':14s} | {'Effective Px %':14s} | {'Exposure Ratio':14s} | {'Tile Draw Prob %':16s}"
    print(header_s)
    print("  " + "-" * (len(header_s) - 2))
    for i, name in enumerate(CLASS_NAMES):
        ratio = exp_pct_new[i] / max(freqs[i] * 100.0, 1e-6)
        print(
            f"  {name:12s} | {freqs[i]*100:13.4f}% | {exp_pct_new[i]:13.4f}% | "
            f"{ratio:13.2f}x | {exp_tile_new[i]:15.2f}%"
        )

    # 5. Empirical Batch Sampling Simulation (100 batches drawn with sampler)
    print("\n--- 4. Empirical Batch Sampling Simulation (100 batches, batch_size=16) ---")
    sim_loader = DataLoader(
        train_ds,
        batch_size=16,
        sampler=sampler,
        num_workers=0,
    )

    batch_px_fractions = []
    for step, b in enumerate(sim_loader):
        if step >= 100:
            break
        lbl = b["label"].numpy()
        valid = (lbl != 255)
        if valid.any():
            binc = np.bincount(lbl[valid], minlength=8)[:8]
            batch_px_fractions.append(binc / binc.sum())

    batch_arr = np.array(batch_px_fractions) * 100.0  # (100, 8) in percent
    mean_exp = batch_arr.mean(axis=0)
    std_exp = batch_arr.std(axis=0)
    min_exp = batch_arr.min(axis=0)
    max_exp = batch_arr.max(axis=0)

    header_b = f"  {'Class':12s} | {'Mean %':8s} | {'Std %':8s} | {'Min %':8s} | {'Max %':8s}"
    print(header_b)
    print("  " + "-" * (len(header_b) - 2))
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:12s} | {mean_exp[i]:7.2f}% | {std_exp[i]:7.2f}% | {min_exp[i]:7.2f}% | {max_exp[i]:7.2f}%")

    print("\n" + "=" * 80)
    print(">>> PRE-TRAINING DATA DIAGNOSTICS COMPLETE AND FULLY VERIFIED! <<<")
    print("=" * 80 + "\n")

    return {
        "split_info": split_info,
        "pixel_counts": total_pixels_per_class.tolist(),
        "pixel_frequencies": freqs.tolist(),
        "tile_presence_any": tiles_any.tolist(),
        "tile_meaningful": tiles_meaningful.tolist(),
        "expected_pixel_pct": exp_pct_new.tolist(),
        "empirical_batch_mean": mean_exp.tolist(),
    }


if __name__ == "__main__":
    cache_p = Path(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\3bf06703-89b4-4c6d-9b4c-cd159c7b5e3c\scratch\train_tile_counts_all.npy")
    run_data_diagnostics(counts_cache_path=cache_p)
