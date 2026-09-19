"""SAR Image Preprocessing and 3-Channel Synthesis for Multimodal Vision-Language Models.

Transforms single-band or dual-polarimetric (VV, VH) SAR rasters into normalized,
3-channel RGB-compatible images suitable for Qwen2.5-VL vision input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from core.logging import get_logger

logger = get_logger("sar_preprocessor")


class SARPreprocessor:
    """Standardized preprocessor for remote-sensing Synthetic Aperture Radar (SAR) imagery."""

    DEFAULT_EPSILON = 1e-6
    DEFAULT_LOWER_PERCENTILE = 1.0
    DEFAULT_UPPER_PERCENTILE = 99.0

    @classmethod
    def load_sar_raster(cls, file_path: Union[str, Path]) -> np.ndarray:
        """Load a SAR raster from disk, handling GeoTIFF, TIFF, or common image formats."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"SAR file not found: {file_path}")

        ext = p.suffix.lower()
        if ext in {".tif", ".tiff"}:
            try:
                import tifffile
                data = tifffile.imread(str(p))
            except Exception:
                # Fallback to PIL
                with Image.open(p) as img:
                    data = np.array(img)
        else:
            with Image.open(p) as img:
                data = np.array(img)

        return data.astype(np.float32)

    @classmethod
    def sanitize_array(cls, data: np.ndarray) -> np.ndarray:
        """Replace NaNs and infinities with median/safe boundary values."""
        if not np.all(np.isfinite(data)):
            finite_mask = np.isfinite(data)
            median_val = float(np.median(data[finite_mask])) if np.any(finite_mask) else 0.0
            data = np.nan_to_num(data, nan=median_val, posinf=median_val, neginf=median_val)
        return data

    @classmethod
    def normalize_band(
        cls,
        band: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> np.ndarray:
        """Apply percentile contrast stretching and scale to [0, 255] uint8."""
        band = cls.sanitize_array(band)
        p_low = np.percentile(band, lower_percentile)
        p_high = np.percentile(band, upper_percentile)

        if p_high <= p_low:
            p_high = p_low + 1e-3

        clipped = np.clip(band, p_low, p_high)
        norm = (clipped - p_low) / (p_high - p_low)
        return (norm * 255.0).astype(np.uint8)

    @classmethod
    def construct_3channel_sar(
        cls,
        data: np.ndarray,
        mode: str = "auto",
        epsilon: float = DEFAULT_EPSILON,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Synthesize a 3-channel RGB image from raw SAR array data.
        
        Supported configurations:
        1. 2-Band [H, W, 2] or [2, H, W]: (VV, VH) -> Channels: [VV, VH, log(VV/VH)]
        2. 1-Band [H, W]: (Single polarization) -> Channels: [Grayscale, Grayscale, Grayscale]
        3. 3-Band [H, W, 3]: Already 3 channels -> Validated and percentile-normalized
        """
        data = cls.sanitize_array(data)

        # Normalize channel dimension layout to (H, W, C) or (H, W)
        if data.ndim == 3 and data.shape[0] in {1, 2, 3} and data.shape[0] < data.shape[1]:
            data = np.transpose(data, (1, 2, 0))

        if data.ndim == 2 or (data.ndim == 3 and data.shape[-1] == 1):
            # Single band intensity: replicate across 3 channels
            band = data[..., 0] if data.ndim == 3 else data
            norm_band = cls.normalize_band(band)
            rgb_arr = np.stack([norm_band, norm_band, norm_band], axis=-1)
            metadata = {
                "modality": "sar",
                "representation": "3x_single_band_replicated",
                "channels": ["intensity", "intensity", "intensity"],
                "source_shape": list(data.shape),
                "is_multipolarized": False,
            }

        elif data.ndim == 3 and data.shape[-1] == 2:
            # Dual-polarimetric SAR: VV and VH
            vv = data[..., 0]
            vh = data[..., 1]

            # Compute numerically stable logarithmic ratio: log((VV + eps) / (VH + eps))
            log_ratio = np.log((np.abs(vv) + epsilon) / (np.abs(vh) + epsilon))

            norm_vv = cls.normalize_band(vv)
            norm_vh = cls.normalize_band(vh)
            norm_ratio = cls.normalize_band(log_ratio)

            rgb_arr = np.stack([norm_vv, norm_vh, norm_ratio], axis=-1)
            metadata = {
                "modality": "sar",
                "representation": "dual_pol_ratio",
                "channels": ["VV", "VH", "log(VV/VH)"],
                "source_shape": list(data.shape),
                "is_multipolarized": True,
                "epsilon": epsilon,
            }

        elif data.ndim == 3 and data.shape[-1] >= 3:
            # 3 or more bands: normalize first 3 bands
            b0 = cls.normalize_band(data[..., 0])
            b1 = cls.normalize_band(data[..., 1])
            b2 = cls.normalize_band(data[..., 2])
            rgb_arr = np.stack([b0, b1, b2], axis=-1)
            metadata = {
                "modality": "sar",
                "representation": "3_band_direct",
                "channels": ["band_0", "band_1", "band_2"],
                "source_shape": list(data.shape),
                "is_multipolarized": True,
            }

        else:
            raise ValueError(f"Unsupported SAR data shape for preprocessing: {data.shape}")

        pil_img = Image.fromarray(rgb_arr, mode="RGB")
        return pil_img, metadata

    @classmethod
    def process_file(
        cls,
        file_path: Union[str, Path],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Convenience method to load and process a SAR raster file directly."""
        data = cls.load_sar_raster(file_path)
        return cls.construct_3channel_sar(data)
