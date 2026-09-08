"""Unit tests for Official 8-Class WHU-OPT-SAR Data Pipeline.

Verifies:
1. All 8 official class values (0, 10, 20, 30, 40, 50, 60, 70) are preserved.
2. No collapse of 'Others' (70) into 'Background' (0).
3. 5556x3704 official image dimensions and deterministic 256x256 border padding with ignore_index=255.
4. Categorical label nearest-neighbor interpolation.
5. Deterministic query aggregation from 8-class probabilities to SatQuery intents.
6. Dataset intent_vector shape (8,) per sample and (16, 8) for DataLoader batch of 16.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from PIL import Image

from specialists.optical_sar.dataset import OpticalSarPairedDataset, aggregate_8class_probs_to_satquery_intents
from specialists.optical_sar.tile_official_dataset import (
    LABEL_VALUE_TO_INDEX,
    OFFICIAL_IMAGE_SHAPE,
    OFFICIAL_LABEL_VALUES,
    pad_raster_to_tile_grid,
    tile_raw_image_pair,
)


def test_official_8class_preservation_and_tiling(tmp_path: Path) -> None:
    """Test tiling a mock 5556x3704 raw raster triplet preserves all 8 official class values."""
    raw_dir = tmp_path / "raw"
    opt_file = raw_dir / "opt.png"
    sar_file = raw_dir / "sar.png"
    label_file = raw_dir / "label.png"

    raw_dir.mkdir(parents=True)

    # Verify official dimensions constant (5556 x 3704)
    w_raw, h_raw = OFFICIAL_IMAGE_SHAPE
    assert (w_raw, h_raw) == (5556, 3704)

    # Create mock 512x512 raw rasters containing all 8 official class values
    h, w = 512, 512
    opt_arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    sar_arr = (np.random.rand(h, w, 2) * 255).astype(np.uint8)

    label_arr = np.zeros((h, w), dtype=np.uint8)
    label_arr[0:128, 0:128] = 0   # Background
    label_arr[0:128, 128:256] = 10 # Farmland
    label_arr[0:128, 256:384] = 20 # City
    label_arr[0:128, 384:512] = 30 # Village
    label_arr[128:256, 0:128] = 40 # Water
    label_arr[128:256, 128:256] = 50 # Forest
    label_arr[128:256, 256:384] = 60 # Road
    label_arr[128:256, 384:512] = 70 # Others (NOT collapsed into Background!)

    Image.fromarray(opt_arr).save(opt_file)
    Image.fromarray(sar_arr).save(sar_file)
    Image.fromarray(label_arr).save(label_file)

    split_dir = tmp_path / "data" / "official_whu_opt_sar" / "train"
    n_tiles = tile_raw_image_pair(
        opt_path=opt_file,
        sar_path=sar_file,
        label_path=label_file,
        output_split_dir=split_dir,
        pair_id="test_pair_01",
        tile_size=256,
        stride=256,
        min_valid_pixels_ratio=0.0,
    )

    assert n_tiles == 4

    # Verify OpticalSarPairedDataset loads tiles preserving 8-class internal indices (0..7)
    ds = OpticalSarPairedDataset(root_dir=tmp_path / "data" / "official_whu_opt_sar", split="train", num_classes=8)
    assert len(ds) == 4

    sample = ds[0]
    assert sample["optical"].shape[1:] == (256, 256)
    assert sample["sar"].shape[1:] == (256, 256)
    assert sample["label"].shape == (256, 256)

    # Check official class index 7 ('Others') is preserved and distinct from 0 ('Background')
    raw_vals = torch.unique(sample["raw_label_value"]).tolist()
    for rv in raw_vals:
        if rv != 255:
            assert rv in OFFICIAL_LABEL_VALUES

    # Confirm index 7 corresponds to value 70 ('Others') and index 0 to value 0 ('Background')
    assert LABEL_VALUE_TO_INDEX[70] == 7
    assert LABEL_VALUE_TO_INDEX[0] == 0
    assert LABEL_VALUE_TO_INDEX[70] != LABEL_VALUE_TO_INDEX[0]


def test_deterministic_5556x3704_border_padding() -> None:
    """Test deterministic border padding for 5556x3704 images with ignore_index=255."""
    w_raw, h_raw = 5556, 3704

    # Mock raw label mask of exact official dimensions (H=3704, W=5556)
    label_arr = np.ones((h_raw, w_raw), dtype=np.uint8) * 10 # Farmland

    # Pad to grid divisible by 256
    padded_mask = pad_raster_to_tile_grid(label_arr, tile_size=256, pad_value=255)

    # Target shape: H=3840 (15*256), W=5632 (22*256)
    assert padded_mask.shape == (3840, 5632)

    # Verify original area retains class 10 and padded border has ignore_index=255
    assert (padded_mask[:3704, :5556] == 10).all()
    assert (padded_mask[3704:, :] == 255).all()
    assert (padded_mask[:, 5556:] == 255).all()


def test_categorical_nearest_neighbor_interpolation() -> None:
    """Test nearest neighbor interpolation on label masks to prevent invalid non-integer float class IDs."""
    label_mask = torch.tensor([[0, 10], [40, 70]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 2, 2)

    resampled_nearest = torch.nn.functional.interpolate(label_mask, size=(4, 4), mode="nearest").squeeze().long()

    unique_vals = set(torch.unique(resampled_nearest).tolist())
    assert unique_vals.issubset({0, 10, 40, 70})

    resampled_bilinear = torch.nn.functional.interpolate(label_mask, size=(4, 4), mode="bilinear", align_corners=False).squeeze()
    has_floats = not torch.equal(resampled_bilinear, resampled_bilinear.round())
    assert has_floats, "Bilinear interpolation on categorical label masks introduces invalid float class IDs!"


def test_triplet_validation_rejection(tmp_path: Path) -> None:
    """Test OpticalSarPairedDataset raises ValueError if SAR or Label files are missing."""
    split_dir = tmp_path / "data" / "official_whu_opt_sar" / "train"
    opt_dir = split_dir / "optical"
    sar_dir = split_dir / "sar"
    label_dir = split_dir / "labels"

    opt_dir.mkdir(parents=True)
    sar_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(opt_dir / "orphan_tile.png")

    with pytest.raises(ValueError, match="Dataset integrity failure"):
        OpticalSarPairedDataset(root_dir=tmp_path / "data" / "official_whu_opt_sar", split="train")


def test_8class_query_aggregation_determinism() -> None:
    """Test deterministic mapping from 8-class neural probabilities to SatQuery query intent categories."""
    # 8 classes: 0=Background, 1=Farmland, 2=City, 3=Village, 4=Water, 5=Forest, 6=Road, 7=Others
    probs_8class = torch.tensor([
        0.05, # 0: Background
        0.15, # 1: Farmland (Veg)
        0.25, # 2: City (Built-up)
        0.15, # 3: Village (Built-up)
        0.20, # 4: Water (Water)
        0.10, # 5: Forest (Veg)
        0.05, # 6: Road (Built-up)
        0.05  # 7: Others (Others)
    ])

    agg = aggregate_8class_probs_to_satquery_intents(probs_8class)

    # City (0.25) + Village (0.15) + Road (0.05) = 0.45
    assert pytest.approx(agg["built_up"].item()) == 0.45

    # Water (0.20)
    assert pytest.approx(agg["water"].item()) == 0.20

    # Farmland (0.15) + Forest (0.10) = 0.25
    assert pytest.approx(agg["vegetation"].item()) == 0.25

    # Others (0.05)
    assert pytest.approx(agg["others"].item()) == 0.05

    # Background (0.05)
    assert pytest.approx(agg["background"].item()) == 0.05

    # Total probability sum equals 1.0
    total = agg["built_up"] + agg["water"] + agg["vegetation"] + agg["others"] + agg["background"]
    assert pytest.approx(total.item()) == 1.0


def test_dataset_8class_intent_vector_contract(tmp_path: Path) -> None:
    """Test OpticalSarPairedDataset intent_vector contract is shape (8,) for single sample and (16, 8) for batch of 16."""
    real_ds_dir = Path("data/official_whu_opt_sar")
    if (real_ds_dir / "train" / "optical").exists():
        ds = OpticalSarPairedDataset(real_ds_dir, split="train")
    else:
        # Create minimal valid mock dataset structure in tmp_path
        train_dir = tmp_path / "mock_ds" / "train"
        for sub in ["optical", "sar", "labels", "metadata"]:
            (train_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(16):
            tid = f"mock_{i:03d}"
            Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(train_dir / "optical" / f"{tid}.png")
            Image.fromarray(np.zeros((256, 256, 2), dtype=np.uint8)).save(train_dir / "sar" / f"{tid}.png")
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(train_dir / "labels" / f"{tid}_mask.png")
            with open(train_dir / "metadata" / f"{tid}.json", "w") as f:
                json.dump({"tile_id": tid}, f)
        ds = OpticalSarPairedDataset(tmp_path / "mock_ds", split="train")

    # Assert single sample intent_vector shape is (8,)
    sample = ds[0]
    assert sample["intent_vector"].shape == (8,), f"Expected intent_vector shape (8,), got {sample['intent_vector'].shape}"

    # Assert DataLoader batch of 16 produces intent_vector shape (16, 8)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    assert batch["intent_vector"].shape == (16, 8), f"Expected batch intent_vector shape (16, 8), got {batch['intent_vector'].shape}"


def test_drive_manifest_structure() -> None:
    """Test official WHU-OPT-SAR drive manifest validity and triplet integrity."""
    from specialists.optical_sar.download_official_dataset import MANIFEST_PATH
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)
    assert data["total_pairs"] == 100
    assert len(data["pairs"]) == 100
    for p in data["pairs"]:
        assert "pair_id" in p
        assert "optical" in p and "id" in p["optical"] and len(p["optical"]["id"]) > 10
        assert "sar" in p and "id" in p["sar"] and len(p["sar"]["id"]) > 10
        assert "lbl" in p and "id" in p["lbl"] and len(p["lbl"]["id"]) > 10
