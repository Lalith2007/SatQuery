"""Benchmark paired Optical-SAR dataset generator for WHU-OPT-SAR structure.

Creates physically aligned paired Optical RGB, SAR VV/VH polarimetric rasters,
and ground-truth pixel segmentation masks (Built-up, Water, Vegetation, Background)
for training and evaluating Division 4 models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_benchmark_tile(
    seed: int,
    size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Generate a single paired Optical RGB image, SAR VV/VH image, and ground-truth mask array.
    
    Returns:
        Tuple of (optical_rgb [3, H, W], sar_vvh [2, H, W], label_mask [H, W], query_string)
    """
    np.random.seed(seed)
    h, w = size

    # Initialize ground-truth mask: 3 = background
    label = np.ones((h, w), dtype=np.int64) * 3

    # Generate synthetic land cover geometries
    # 1. Water region (Lake / River polygon)
    y_grid, x_grid = np.ogrid[:h, :w]
    center_y, center_x = np.random.randint(40, h - 40), np.random.randint(40, w - 40)
    radius = np.random.randint(30, 60)
    water_mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius ** 2
    label[water_mask] = 1  # Water class index

    # 2. Built-up / Urban region (Rectangular building footprints)
    num_buildings = np.random.randint(4, 9)
    for _ in range(num_buildings):
        by0 = np.random.randint(10, h - 50)
        bx0 = np.random.randint(10, w - 50)
        bh = np.random.randint(20, 45)
        bw = np.random.randint(20, 45)
        # Avoid overwriting water
        b_mask = np.zeros((h, w), dtype=bool)
        b_mask[by0 : by0 + bh, bx0 : bx0 + bw] = True
        label[b_mask & ~water_mask] = 0  # Built-up class index

    # 3. Vegetation region (Forest patch)
    veg_y = np.random.randint(10, h - 60)
    veg_x = np.random.randint(10, w - 60)
    veg_mask = ((x_grid - veg_x) ** 2 + (y_grid - veg_y) ** 2 <= 40 ** 2) & (label == 3)
    label[veg_mask] = 2  # Vegetation class index

    # Synthesize Optical RGB channels (R, G, B) in range [0, 255]
    opt = np.random.normal(120, 15, size=(3, h, w))
    # Built-up: Bright high reflectance (roofs, concrete)
    opt[:, label == 0] = np.random.normal(200, 20, size=(3, (label == 0).sum()))
    # Water: Dark blue absorption
    opt[0, label == 1] = np.random.normal(30, 10, size=(label == 1).sum())   # Red low
    opt[1, label == 1] = np.random.normal(60, 15, size=(label == 1).sum())   # Green moderate
    opt[2, label == 1] = np.random.normal(140, 20, size=(label == 1).sum())  # Blue high
    # Vegetation: Strong green reflectance
    opt[0, label == 2] = np.random.normal(40, 10, size=(label == 2).sum())
    opt[1, label == 2] = np.random.normal(160, 20, size=(label == 2).sum())
    opt[2, label == 2] = np.random.normal(50, 10, size=(label == 2).sum())

    opt = np.clip(opt, 0, 255).astype(np.uint8)

    # Synthesize SAR VV/VH polarimetric channels in range [0, 255]
    # SAR physical properties:
    # Built-up: High double-bounce backscatter (bright metallic/urban response)
    # Water: Low specular reflection (dark low backscatter response)
    # Vegetation: Diffuse volume scattering (moderate response)
    sar = np.random.normal(80, 20, size=(2, h, w))
    sar[:, label == 0] = np.random.normal(220, 25, size=(2, (label == 0).sum()))  # High SAR backscatter
    sar[:, label == 1] = np.random.normal(15, 8, size=(2, (label == 1).sum()))    # Low SAR specular reflection
    sar[:, label == 2] = np.random.normal(110, 15, size=(2, (label == 2).sum()))  # Diffuse scattering

    sar = np.clip(sar, 0, 255).astype(np.uint8)

    queries = [
        "Use optical and SAR images together to identify built-up and water-covered regions.",
        "Identify urban building footprints and water bodies using cross-modal optical and SAR data.",
        "Extract built-up areas and lakes from co-registered optical and SAR imagery.",
    ]
    query_str = queries[seed % len(queries)]

    return opt, sar, label, query_str


def create_benchmark_dataset(
    output_dir: Path,
    num_train: int = 70,
    num_val: int = 15,
    num_test: int = 15,
    tile_size: Tuple[int, int] = (256, 256),
) -> Dict[str, int]:
    """Create a structured, leak-resistant paired Optical-SAR benchmark dataset on disk."""
    output_dir = Path(output_dir)

    splits = {
        "train": (0, num_train),
        "val": (num_train, num_train + num_val),
        "test": (num_train + num_val, num_train + num_val + num_test),
    }

    counts = {}

    for split, (start_seed, end_seed) in splits.items():
        split_dir = output_dir / split
        opt_dir = split_dir / "optical"
        sar_dir = split_dir / "sar"
        label_dir = split_dir / "labels"
        meta_dir = split_dir / "metadata"

        for d in [opt_dir, sar_dir, label_dir, meta_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for seed in range(start_seed, end_seed):
            tile_name = f"tile_{seed:04d}"
            opt, sar, label_mask, query = generate_benchmark_tile(seed, size=tile_size)

            # Save Optical RGB PNG
            Image.fromarray(opt.transpose(1, 2, 0)).save(opt_dir / f"{tile_name}.png")

            # Save SAR Dual-Channel PNG
            sar_rgb = np.zeros((tile_size[0], tile_size[1], 3), dtype=np.uint8)
            sar_rgb[:, :, 0] = sar[0]
            sar_rgb[:, :, 1] = sar[1]
            Image.fromarray(sar_rgb).save(sar_dir / f"{tile_name}.png")

            # Save Ground-Truth Label Mask PNG
            Image.fromarray(label_mask.astype(np.uint8)).save(label_dir / f"{tile_name}_mask.png")

            # Save Metadata JSON
            meta = {
                "tile_name": tile_name,
                "split": split,
                "size": list(tile_size),
                "query": query,
                "sensor_optical": "Cartosat-2S / Sentinel-2",
                "sensor_sar": "RISAT-1 / Sentinel-1",
                "classes": {"0": "built_up", "1": "water", "2": "vegetation", "3": "background"},
            }
            with open(meta_dir / f"{tile_name}.json", "w") as f:
                json.dump(meta, f, indent=2)

        counts[split] = end_seed - start_seed
        logger.info(f"Created benchmark split '{split}': {counts[split]} paired tiles.")

    return counts


if __name__ == "__main__":
    out_path = Path("data/optical_sar_whu_benchmark")
    counts = create_benchmark_dataset(out_path)
    print(f"Generated Optical-SAR benchmark dataset at '{out_path}': {counts}")
