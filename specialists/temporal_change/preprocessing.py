"""Image preprocessing for Division 3: Bi-Temporal Change Intelligence.

Handles loading, normalization, and tensor conversion for T0/T1 image pairs.
Controlled geospatial reprojection is performed only when explicitly justified
and authorized via configuration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from core.logging import get_logger

logger = get_logger("temporal_change.preprocessing")


def load_image_as_array(path: str) -> np.ndarray:
    """Load an image file to a numpy array (H, W, C) in uint8 RGB.

    Supports PNG, JPEG, TIFF via Pillow.
    For multi-band GeoTIFFs, falls back to tifffile if available.

    Returns:
        np.ndarray of shape (H, W, C) with dtype uint8.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    # Try tifffile for .tif/.tiff first (handles multi-band better)
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(str(p))
            # Handle various shapes
            if arr.ndim == 2:
                # Grayscale -> expand to 3 channels
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3:
                # Could be (C, H, W) or (H, W, C)
                if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
                    arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                # Take first 3 channels if more than 3
                if arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                elif arr.shape[2] == 1:
                    arr = np.concatenate([arr, arr, arr], axis=-1)
            # Normalize to uint8
            if arr.dtype != np.uint8:
                if arr.dtype in (np.float32, np.float64):
                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                elif arr.dtype == np.uint16:
                    arr = (arr / 256).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            return arr
        except ImportError:
            pass  # Fall through to PIL

    # General case: use PIL
    img = Image.open(path)
    img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def resize_image(
    arr: np.ndarray,
    target_size: int,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Resize an image array to (target_size, target_size) for model input.

    This is model-input preprocessing, NOT spatial alignment.
    Uses PIL for consistent cross-platform behavior.

    Args:
        arr: Input array (H, W, C) uint8.
        target_size: Target square dimension.
        interpolation: 'nearest', 'bilinear', or 'bicubic'.

    Returns:
        Resized array (target_size, target_size, C) uint8.
    """
    if arr.shape[0] == target_size and arr.shape[1] == target_size:
        return arr

    interp_map = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }
    pil_interp = interp_map.get(interpolation, Image.BILINEAR)
    img = Image.fromarray(arr)
    img = img.resize((target_size, target_size), resample=pil_interp)
    return np.array(img, dtype=np.uint8)


def normalize_to_float32(arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 image to float32 [0.0, 1.0]."""
    return arr.astype(np.float32) / 255.0


def preprocess_pair(
    t0_path: str,
    t1_path: str,
    model_input_size: int = 256,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and preprocess a T0/T1 image pair for model inference.

    This performs model-input preprocessing (resize for model consumption),
    NOT spatial alignment. Spatial alignment is handled separately in the
    alignment policy stage.

    Args:
        t0_path: Path to T0 image.
        t1_path: Path to T1 image.
        model_input_size: Target square dimension for model input.
        normalize: Whether to normalize to float32 [0, 1].

    Returns:
        Tuple of (t0_array, t1_array, preprocessing_metadata).
    """
    start = time.perf_counter()

    t0_raw = load_image_as_array(t0_path)
    t1_raw = load_image_as_array(t1_path)

    t0_original_shape = t0_raw.shape
    t1_original_shape = t1_raw.shape

    # Resize for model input
    t0_resized = resize_image(t0_raw, model_input_size)
    t1_resized = resize_image(t1_raw, model_input_size)

    if normalize:
        t0_out = normalize_to_float32(t0_resized)
        t1_out = normalize_to_float32(t1_resized)
    else:
        t0_out = t0_resized
        t1_out = t1_resized

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    meta = {
        "t0_original_shape": list(t0_original_shape),
        "t1_original_shape": list(t1_original_shape),
        "model_input_size": model_input_size,
        "normalized": normalize,
        "preprocessing_time_ms": round(elapsed_ms, 2),
    }

    return t0_out, t1_out, meta
