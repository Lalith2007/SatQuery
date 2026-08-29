"""PyTorch Paired Optical-SAR Dataset loader supporting official 8-class WHU-OPT-SAR annotations.

Loads co-registered Optical rasters, SAR rasters, official 8-class ground-truth segmentation masks
(values 0, 10, 20, 30, 40, 50, 60, 70), mapping them to internal indices (0..7).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from specialists.optical_sar.config import PreprocessingConfig
from specialists.optical_sar.preprocessing import OpticalSarPreprocessor
from specialists.optical_sar.query_intent import QueryIntentInterpreter
from specialists.optical_sar.tile_official_dataset import LABEL_VALUE_TO_INDEX, OFFICIAL_LABEL_VALUES

logger = logging.getLogger(__name__)


def aggregate_8class_probs_to_satquery_intents(
    probs_8class: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Deterministically aggregate 8-class neural probabilities into SatQuery intent targets.
    
    Args:
        probs_8class: Tensor of shape (B, 8, H, W) or (8, H, W)
        
    Returns:
        Dict containing aggregated intent tensors (BUILT_UP, WATER, VEGETATION, OTHERS, BACKGROUND)
    """
    dim = 1 if probs_8class.ndim == 4 else 0

    p_background = probs_8class.select(dim, 0) # Label 0: Background
    p_farmland   = probs_8class.select(dim, 1) # Label 10: Farmland
    p_city       = probs_8class.select(dim, 2) # Label 20: City
    p_village    = probs_8class.select(dim, 3) # Label 30: Village
    p_water      = probs_8class.select(dim, 4) # Label 40: Water
    p_forest     = probs_8class.select(dim, 5) # Label 50: Forest
    p_road       = probs_8class.select(dim, 6) # Label 60: Road
    p_others     = probs_8class.select(dim, 7) # Label 70: Others

    # SatQuery Intent Aggregation Rules
    built_up   = p_city + p_village + p_road
    water      = p_water
    vegetation = p_farmland + p_forest
    others     = p_others
    background = p_background

    return {
        "built_up": built_up,
        "water": water,
        "vegetation": vegetation,
        "others": others,
        "background": background,
    }


class OpticalSarPairedDataset(Dataset):
    """PyTorch Dataset loading spatially co-registered Optical, SAR, Official 8-Class Label Mask, and Query Intent."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        num_classes: int = 8,
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ) -> None:
        self.root_dir = Path(root_dir) / split
        self.num_classes = num_classes
        self.preprocessor = OpticalSarPreprocessor(preprocessing_config or PreprocessingConfig())
        self.query_interpreter = QueryIntentInterpreter()

        self.opt_dir = self.root_dir / "optical"
        self.sar_dir = self.root_dir / "sar"
        self.label_dir = self.root_dir / "labels"
        self.meta_dir = self.root_dir / "metadata"

        if not self.opt_dir.exists():
            raise FileNotFoundError(f"Optical directory not found at '{self.opt_dir}'")

        # Discover paired tile filenames
        self.samples: List[str] = sorted([
            f.name for f in self.opt_dir.glob("*.*")
            if f.suffix.lower() in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]
        ])

        if len(self.samples) == 0:
            raise ValueError(f"No valid image files found in '{self.opt_dir}'")

        self._verify_triplets()
        logger.info(f"Loaded OpticalSarPairedDataset split='{split}' ({self.num_classes}-class mode): {len(self.samples)} verified paired tiles.")

    def _verify_triplets(self) -> None:
        """Verify that every optical sample has a corresponding SAR and label file."""
        missing: List[str] = []
        for filename in self.samples:
            stem = Path(filename).stem
            sar_file = self.sar_dir / filename if (self.sar_dir / filename).exists() else self.sar_dir / f"{stem}.tif"
            if not sar_file.exists():
                sar_file = self.sar_dir / f"{stem}.png"

            label_file = self.label_dir / f"{stem}_mask.png"
            if not label_file.exists():
                label_file = self.label_dir / f"{stem}.png"

            if not sar_file.exists() or not label_file.exists():
                missing.append(stem)

        if len(missing) > 0:
            raise ValueError(f"Dataset integrity failure: {len(missing)} samples missing SAR or label files: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.samples[idx]
        stem = Path(filename).stem

        opt_file = self.opt_dir / filename
        sar_file = self.sar_dir / filename if (self.sar_dir / filename).exists() else self.sar_dir / f"{stem}.tif"
        if not sar_file.exists():
            sar_file = self.sar_dir / f"{stem}.png"

        label_file = self.label_dir / f"{stem}_mask.png"
        if not label_file.exists():
            label_file = self.label_dir / f"{stem}.png"

        # 1. Preprocess Optical & SAR rasters
        opt_data = self.preprocessor.preprocess_optical(opt_file)
        sar_data = self.preprocessor.preprocess_sar(sar_file)

        opt_aligned, sar_aligned = self.preprocessor.align_spatial_dimensions(
            opt_data.tensor, sar_data.tensor
        )

        # 2. Load Official 8-Class Ground-Truth Mask (Values: 0, 10, 20, 30, 40, 50, 60, 70, 255)
        with Image.open(label_file) as img:
            label_arr = np.array(img, dtype=np.int64)
            if label_arr.ndim == 3:
                label_arr = label_arr[:, :, 0]

        # Spatial resolution verification
        opt_h, opt_w = opt_aligned.shape[1], opt_aligned.shape[2]
        label_tensor = torch.from_numpy(label_arr).long()
        if label_tensor.shape != (opt_h, opt_w):
            label_tensor = torch.nn.functional.interpolate(
                label_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(opt_h, opt_w),
                mode="nearest"  # CRITICAL: Categorical masks MUST use nearest neighbor interpolation
            ).squeeze(0).squeeze(0).long()

        # Map official label values (0, 10..70) to internal target indices (0..7, 255 ignore)
        target_index_mask = torch.ones_like(label_tensor) * 255
        for val, idx_val in LABEL_VALUE_TO_INDEX.items():
            target_index_mask[label_tensor == val] = idx_val

        # Preserve border ignore_index = 255
        target_index_mask[label_tensor == 255] = 255

        # 3. Load Query Intent
        query_str = "Use optical and SAR images together to identify built-up and water-covered regions."
        meta_file = self.meta_dir / f"{stem}.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta_data = json.load(f)
                    query_str = meta_data.get("query", query_str)
            except Exception:
                pass

        intent_vec = self.query_interpreter.get_intent_vector(query_str)
        intent_dict = self.query_interpreter.parse_query_intent(query_str)

        return {
            "optical": opt_aligned,                 # Tensor (3, H, W)
            "sar": sar_aligned,                     # Tensor (2, H, W)
            "label": target_index_mask,             # LongTensor (H, W) with target indices 0..7 (and 255 ignore)
            "raw_label_value": label_tensor,        # Raw official values (0, 10, 20..70)
            "intent_vector": intent_vec,            # FloatTensor (8,)
            "intent_dict": intent_dict,
            "query": query_str,
            "sample_id": stem,
            "filename": filename,
        }
