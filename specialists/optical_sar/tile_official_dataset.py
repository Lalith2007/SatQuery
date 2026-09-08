"""Deterministic Tiling Script for Official WHU-OPT-SAR Image Pairs (5556x3704).

Crops raw 5556x3704 Optical, SAR, and 8-class Ground-Truth Segmentation Masks
into spatially aligned 256x256 tiles, preserving official class values (0, 10, 20, 30, 40, 50, 60, 70)
and applying deterministic edge padding with ignore_index=255 for border tiles.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tile_official")

# Official WHU-OPT-SAR 8-Class Label Taxonomy
OFFICIAL_LABEL_VALUES = {
    0: "background",
    10: "farmland",
    20: "city",
    30: "village",
    40: "water",
    50: "forest",
    60: "road",
    70: "others",
}

# Internal 8-Class Index Mapping (0..7)
LABEL_VALUE_TO_INDEX = {
    0: 0,   # background
    10: 1,  # farmland
    20: 2,  # city
    30: 3,  # village
    40: 4,  # water
    50: 5,  # forest
    60: 6,  # road
    70: 7,  # others
}

INDEX_TO_LABEL_VALUE = {v: k for k, v in LABEL_VALUE_TO_INDEX.items()}

OFFICIAL_IMAGE_SHAPE = (5556, 3704)  # Width=5556, Height=3704


def pad_raster_to_tile_grid(
    arr: np.ndarray,
    tile_size: int = 256,
    pad_value: int = 0,
) -> np.ndarray:
    """Deterministically pad raster array so height and width are divisible by tile_size.
    
    Args:
        arr: Array of shape (H, W) or (H, W, C)
        tile_size: Target tile size (default 256)
        pad_value: Constant padding value (0 for rasters, 255 for categorical ignore masks)
    """
    h, w = arr.shape[:2]
    target_h = int(np.ceil(h / tile_size)) * tile_size
    target_w = int(np.ceil(w / tile_size)) * tile_size

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h == 0 and pad_w == 0:
        return arr

    if arr.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))

    return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)


def tile_raw_image_pair(
    opt_path: Path,
    sar_path: Path,
    label_path: Path,
    output_split_dir: Path,
    pair_id: str,
    tile_size: int = 256,
    stride: int = 256,
    min_valid_pixels_ratio: float = 0.5,
) -> int:
    """Crop one large 5556x3704 co-registered Optical/SAR/Label triplet into 256x256 tiles with edge padding.
    
    Args:
        opt_path: Path to raw optical raster (5556x3704)
        sar_path: Path to raw SAR raster (5556x3704)
        label_path: Path to raw official ground-truth mask (official values: 0, 10, 20, 30, 40, 50, 60, 70)
        output_split_dir: Destination directory (e.g. data/official_whu_opt_sar/train)
        pair_id: Raw image pair identifier (e.g. 'whu_pair_001')
        tile_size: Output spatial resolution (default 256)
        stride: Crop window step size (default 256 for non-overlapping)
        min_valid_pixels_ratio: Filter out tiles with excessive background/ignore content
        
    Returns:
        Number of valid tiles generated
    """
    opt_dir = output_split_dir / "optical"
    sar_dir = output_split_dir / "sar"
    label_dir = output_split_dir / "labels"
    meta_dir = output_split_dir / "metadata"

    for d in [opt_dir, sar_dir, label_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load raw rasters
    with Image.open(opt_path) as img_opt:
        opt_arr = np.array(img_opt)
    with Image.open(sar_path) as img_sar:
        sar_arr = np.array(img_sar)
    with Image.open(label_path) as img_lbl:
        label_arr = np.array(img_lbl, dtype=np.int64)

    if label_arr.ndim == 3:
        label_arr = label_arr[:, :, 0]

    # Deterministic padding for border tiles (Target shape: 3840x5632)
    opt_padded = pad_raster_to_tile_grid(opt_arr, tile_size=tile_size, pad_value=0)
    sar_padded = pad_raster_to_tile_grid(sar_arr, tile_size=tile_size, pad_value=0)
    label_padded = pad_raster_to_tile_grid(label_arr, tile_size=tile_size, pad_value=255) # 255 ignore_index for padded border

    h_pad, w_pad = label_padded.shape[:2]
    tile_count = 0

    # Iterate over spatial grid
    for y in range(0, h_pad, stride):
        for x in range(0, w_pad, stride):
            opt_tile = opt_padded[y : y + tile_size, x : x + tile_size]
            sar_tile = sar_padded[y : y + tile_size, x : x + tile_size]
            label_tile = label_padded[y : y + tile_size, x : x + tile_size]

            # Skip tiles that fall entirely inside padded border ignore mask
            if (label_tile == 255).all():
                continue

            tile_id = f"{pair_id}_y{y:04d}_x{x:04d}"

            # Save Optical & SAR tiles
            Image.fromarray(opt_tile).save(opt_dir / f"{tile_id}.png")
            Image.fromarray(sar_tile).save(sar_dir / f"{tile_id}.png")

            # CRITICAL: Save Label tile preserving official values 0, 10..70 (and 255 ignore mask)
            Image.fromarray(label_tile.astype(np.uint8)).save(label_dir / f"{tile_id}_mask.png")

            # Save Metadata JSON
            meta = {
                "tile_id": tile_id,
                "source_pair_id": pair_id,
                "crop_window": {"y": y, "x": x, "height": tile_size, "width": tile_size},
                "official_dimensions": list(OFFICIAL_IMAGE_SHAPE),
                "padded_dimensions": [w_pad, h_pad],
                "official_label_values": OFFICIAL_LABEL_VALUES,
                "label_value_to_index": LABEL_VALUE_TO_INDEX,
                "dataset_provenance": "Official WHU-OPT-SAR Dataset (Wuhan University)",
            }
            with open(meta_dir / f"{tile_id}.json", "w") as f:
                json.dump(meta, f, indent=2)

            tile_count += 1

    return tile_count


def tile_whu_dataset_directory(
    raw_dataset_dir: Path,
    output_dir: Path = Path("data/official_whu_opt_sar"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    tile_size: int = 256,
) -> Dict[str, int]:
    """Process raw official dataset directory using SatQuery Custom Image-Level Split."""
    raw_dataset_dir = Path(raw_dataset_dir)
    output_dir = Path(output_dir)

    opt_raw_dir = raw_dataset_dir / "optical"
    sar_raw_dir = raw_dataset_dir / "sar"
    label_raw_dir = raw_dataset_dir / "labels"
    if not label_raw_dir.exists() and (raw_dataset_dir / "lbl").exists():
        label_raw_dir = raw_dataset_dir / "lbl"

    if not opt_raw_dir.exists():
        raise FileNotFoundError(f"Official raw optical directory missing at '{opt_raw_dir}'")
    if not sar_raw_dir.exists():
        raise FileNotFoundError(f"Official raw SAR directory missing at '{sar_raw_dir}'")
    if not label_raw_dir.exists():
        raise FileNotFoundError(f"Official raw label directory missing at '{label_raw_dir}'")

    raw_pairs = sorted([f.stem for f in opt_raw_dir.glob("*.*")])
    if len(raw_pairs) == 0:
        raise ValueError(f"No raw image files found in '{opt_raw_dir}'")

    np.random.seed(42)
    np.random.shuffle(raw_pairs)

    n_total = len(raw_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = raw_pairs[:n_train]
    val_pairs = raw_pairs[n_train : n_train + n_val]
    test_pairs = raw_pairs[n_train + n_val :]

    splits_assignment = {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }

    split_tile_counts = {}

    for split_name, pairs in splits_assignment.items():
        split_dir = output_dir / split_name
        total_tiles = 0
        for pid in pairs:
            opt_file = next(opt_raw_dir.glob(f"{pid}.*"))
            sar_file = next(sar_raw_dir.glob(f"{pid}.*"))
            label_file = next(label_raw_dir.glob(f"{pid}.*"))

            n_tiles = tile_raw_image_pair(
                opt_path=opt_file,
                sar_path=sar_file,
                label_path=label_file,
                output_split_dir=split_dir,
                pair_id=pid,
                tile_size=tile_size,
            )
            total_tiles += n_tiles

        split_tile_counts[split_name] = total_tiles
        logger.info(f"Split '{split_name}' (SatQuery Custom Image-Level Split): Tiled {len(pairs)} raw image pairs into {total_tiles} spatially aligned 256x256 tiles.")

    manifest = {
        "dataset_name": "Official WHU-OPT-SAR Dataset",
        "official_dimensions": "5556 x 3704 pixels",
        "split_strategy": "SatQuery Custom Image-Level Split (70/15/15)",
        "image_pair_allocation": splits_assignment,
        "tile_counts": split_tile_counts,
        "official_label_values": OFFICIAL_LABEL_VALUES,
        "label_value_to_index": LABEL_VALUE_TO_INDEX,
    }
    with open(output_dir / "split_provenance_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return split_tile_counts
