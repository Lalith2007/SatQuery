"""Explicit Deterministic Sensor Representation Converters for Qwen2.5-VL.

Converts multi-channel Sentinel-1 SAR and 12-band Sentinel-2 MSI data into
canonical 3-channel visual representations compatible with Qwen2.5-VL while
strictly documenting and preserving spectral/sensor metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image


@dataclass
class TransformedImageMetadata:
    """Preserves full sensor provenance alongside the transformed 3-channel visual tensor."""

    source_record_id: str
    sensor: str  # "Sentinel-1" | "Sentinel-2"
    scene_id: str  # e.g. "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
    patch_id: str  # e.g. "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    original_bands: List[str]  # e.g. ["B01"..."B12"] or ["VV", "VH"]
    channels_rendered: List[str]  # e.g. ["B04", "B03", "B02"] or ["VV", "VH", "log(VV/VH)"]
    conversion_method: str
    spectral_information_loss_doc: str
    geolocation: Dict[str, Union[float, str]] = field(default_factory=dict)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class Sentinel1SARConverter:
    """Deterministic converter for Sentinel-1 dual-polarization (VV/VH) SAR.

    Maps 2-channel microwave backscatter into a 3-channel representation:
    - Channel 1 (R): Calibrated linear / dB VV backscatter
    - Channel 2 (G): Calibrated linear / dB VH backscatter
    - Channel 3 (B): Logarithmic cross-polarization ratio log((VV + eps) / (VH + eps)) = VV_dB - VH_dB
    """

    VV_DEFAULT_MIN = -25.0  # dB
    VV_DEFAULT_MAX = 0.0    # dB
    VH_DEFAULT_MIN = -32.0  # dB
    VH_DEFAULT_MAX = -5.0   # dB
    EPSILON = 1e-6

    DOCUMENTED_LOSS = (
        "Phase information and the full 2x2 complex covariance matrix [C2] (including "
        "polarimetric off-diagonal cross-correlation terms) are collapsed into scalar backscatter "
        "intensities (VV, VH) and the cross-polarization power ratio."
    )

    @classmethod
    def convert_s1_to_rgb(
        cls,
        vv_array: np.ndarray,
        vh_array: np.ndarray,
        record_id: str,
        scene_id: str,
        patch_id: str,
        geolocation: Optional[Dict[str, Any]] = None,
        is_db: bool = True,
    ) -> Tuple[Image.Image, TransformedImageMetadata]:
        """Convert VV and VH arrays into a 3-channel PIL Image and sidecar metadata."""
        vv = np.asarray(vv_array, dtype=np.float32)
        vh = np.asarray(vh_array, dtype=np.float32)

        if not is_db:
            # Convert linear amplitude to decibels
            vv = 10.0 * np.log10(np.maximum(vv, cls.EPSILON))
            vh = 10.0 * np.log10(np.maximum(vh, cls.EPSILON))

        # Channel 1: VV normalized to [0, 1]
        ch_r = np.clip((vv - cls.VV_DEFAULT_MIN) / (cls.VV_DEFAULT_MAX - cls.VV_DEFAULT_MIN), 0.0, 1.0)
        # Channel 2: VH normalized to [0, 1]
        ch_g = np.clip((vh - cls.VH_DEFAULT_MIN) / (cls.VH_DEFAULT_MAX - cls.VH_DEFAULT_MIN), 0.0, 1.0)
        # Channel 3: Cross-ratio log(VV / VH) = VV_dB - VH_dB normalized from [-5, 20] dB
        ratio_db = vv - vh
        ch_b = np.clip((ratio_db - (-5.0)) / (20.0 - (-5.0)), 0.0, 1.0)

        rgb_stack = np.stack([ch_r, ch_g, ch_b], axis=-1)
        rgb_uint8 = (rgb_stack * 255.0).astype(np.uint8)
        image = Image.fromarray(rgb_uint8)

        metadata = TransformedImageMetadata(
            source_record_id=record_id,
            sensor="Sentinel-1 SAR",
            scene_id=scene_id,
            patch_id=patch_id,
            original_bands=["VV", "VH"],
            channels_rendered=["VV_normalized", "VH_normalized", "log(VV/VH)_ratio"],
            conversion_method="deterministic_dual_pol_ratio",
            spectral_information_loss_doc=cls.DOCUMENTED_LOSS,
            geolocation=geolocation or {},
            extra_metadata={"is_db": True, "clip_ranges": {"vv": [cls.VV_DEFAULT_MIN, cls.VV_DEFAULT_MAX], "vh": [cls.VH_DEFAULT_MIN, cls.VH_DEFAULT_MAX]}},
        )
        return image, metadata


class Sentinel2MultispectralConverter:
    """Deterministic converter for Sentinel-2 12-band MSI imagery.

    Selects the canonical True-Color Composite (TCC) for natural human-vision compatibility:
    - Channel 1 (R): Band 04 (Red, 665 nm)
    - Channel 2 (G): Band 03 (Green, 560 nm)
    - Channel 3 (B): Band 02 (Blue, 490 nm)

    Documents explicit loss of Red-Edge (B05-B07), NIR (B08, B8A), and SWIR (B11, B12).
    """

    DOCUMENTED_LOSS = (
        "9 out of 12 spectral bands are omitted: Red-Edge (B05, B06, B07 sensitive to chlorophyll), "
        "Near-Infrared (B08, B8A sensitive to vegetative cellular structure), Short-Wave Infrared "
        "(SWIR1 B11, SWIR2 B12 sensitive to soil moisture and burn scars), and atmospheric bands "
        "(B01 coastal aerosol, B09 water vapor). Only the visible spectrum (B04 Red, B03 Green, B02 Blue) is retained."
    )

    REFLECTANCE_SCALE = 2000.0  # BigEarthNet typical DN normalization scale

    @classmethod
    def convert_s2_to_rgb(
        cls,
        band_dict: Dict[str, np.ndarray],
        record_id: str,
        scene_id: str,
        patch_id: str,
        geolocation: Optional[Dict[str, Any]] = None,
        clip_max: float = 2000.0,
    ) -> Tuple[Image.Image, TransformedImageMetadata]:
        """Convert Sentinel-2 12-band dictionary into True Color Composite (B04, B03, B02)."""
        b04 = np.asarray(band_dict["B04"], dtype=np.float32)
        b03 = np.asarray(band_dict["B03"], dtype=np.float32)
        b02 = np.asarray(band_dict["B02"], dtype=np.float32)

        ch_r = np.clip(b04 / clip_max, 0.0, 1.0)
        ch_g = np.clip(b03 / clip_max, 0.0, 1.0)
        ch_b = np.clip(b02 / clip_max, 0.0, 1.0)

        rgb_stack = np.stack([ch_r, ch_g, ch_b], axis=-1)
        rgb_uint8 = (rgb_stack * 255.0).astype(np.uint8)
        image = Image.fromarray(rgb_uint8)

        metadata = TransformedImageMetadata(
            source_record_id=record_id,
            sensor="Sentinel-2 MSI",
            scene_id=scene_id,
            patch_id=patch_id,
            original_bands=sorted(list(band_dict.keys())),
            channels_rendered=["B04 (Red)", "B03 (Green)", "B02 (Blue)"],
            conversion_method="true_color_composite_tcc",
            spectral_information_loss_doc=cls.DOCUMENTED_LOSS,
            geolocation=geolocation or {},
            extra_metadata={"reflectance_clip_max": clip_max, "omitted_bands": ["B01", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]},
        )
        return image, metadata
