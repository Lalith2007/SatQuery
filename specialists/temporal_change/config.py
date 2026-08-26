"""Configuration for Division 3: Bi-Temporal Change Intelligence.

All configurable parameters are centralized here.
No hard-coded model paths, thresholds, or device settings elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class TemporalChangeConfig:
    """Configuration for the bi-temporal change specialist.

    Reads from environment variables with SATQUERY_TC_ prefix,
    falling back to sensible defaults.
    """

    def __init__(self) -> None:
        # --- Model configuration ---
        self.model_checkpoint_path: str = os.environ.get(
            "SATQUERY_TC_MODEL_CHECKPOINT", ""
        )
        self.model_architecture: str = os.environ.get(
            "SATQUERY_TC_MODEL_ARCH", "changeformer"
        )
        self.device: str = os.environ.get("SATQUERY_TC_DEVICE", "auto")

        # --- Thresholds (all configurable) ---
        self.change_threshold: float = float(
            os.environ.get("SATQUERY_TC_CHANGE_THRESHOLD", "0.5")
        )
        self.min_region_area_pixels: int = int(
            os.environ.get("SATQUERY_TC_MIN_REGION_AREA", "100")
        )
        self.morphology_kernel_size: int = int(
            os.environ.get("SATQUERY_TC_MORPH_KERNEL", "3")
        )
        self.max_regions_reported: int = int(
            os.environ.get("SATQUERY_TC_MAX_REGIONS", "20")
        )

        # --- Spatial alignment policy ---
        self.allow_geospatial_reprojection: bool = os.environ.get(
            "SATQUERY_TC_ALLOW_REPROJECTION", "false"
        ).lower() in ("true", "1", "yes")
        self.reprojection_interpolation: str = os.environ.get(
            "SATQUERY_TC_REPROJECT_INTERP", "nearest"
        )

        # --- Preprocessing ---
        self.model_input_size: int = int(
            os.environ.get("SATQUERY_TC_INPUT_SIZE", "256")
        )
        self.normalize_to_float: bool = True

        # --- Output and storage ---
        self.artifact_output_dir: Path = Path(
            os.environ.get("SATQUERY_TC_ARTIFACT_DIR", "artifacts_storage")
        )

        # --- Mock/production mode ---
        self.use_mock_model: bool = os.environ.get(
            "SATQUERY_TC_USE_MOCK", "true"
        ).lower() in ("true", "1", "yes")

    def resolve_device(self) -> str:
        """Resolve the actual compute device, with graceful CPU fallback."""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return self.device

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.artifact_output_dir.mkdir(parents=True, exist_ok=True)
