"""Postprocessing for Division 3: Bi-Temporal Change Intelligence.

Configurable thresholding, morphological filtering, connected-component
extraction, bounding-box generation, and area statistics.

Pixel-area statistics are kept distinct from geographic-area statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from specialists.temporal_change.interfaces import ChangedRegion


@dataclass
class PostprocessingResult:
    """Result of postprocessing a change detection output."""
    binary_map: np.ndarray  # H x W uint8 {0, 1}
    regions: List[ChangedRegion]
    changed_pixel_count: int
    total_pixel_count: int
    changed_pixel_ratio: float
    parameters: Dict[str, Any] = field(default_factory=dict)


def threshold_probability_map(
    prob_map: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply threshold to a probability map to produce a binary change mask.

    Args:
        prob_map: H x W float array with values in [0, 1].
        threshold: Decision boundary.

    Returns:
        Binary mask H x W uint8 {0, 1}.
    """
    return (prob_map >= threshold).astype(np.uint8)


def morphological_cleanup(
    binary_map: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply conservative morphological opening to remove noise.

    Uses a small kernel to avoid destroying genuine small changes.

    Args:
        binary_map: H x W uint8 binary map.
        kernel_size: Size of the morphological kernel.

    Returns:
        Cleaned binary map.
    """
    if kernel_size <= 0:
        return binary_map

    try:
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
        return cleaned
    except ImportError:
        # Fallback: simple erosion + dilation using scipy
        try:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            eroded = ndimage.binary_erosion(binary_map.astype(bool), structure=struct)
            dilated = ndimage.binary_dilation(eroded, structure=struct)
            return dilated.astype(np.uint8)
        except ImportError:
            # No morphology available — return as-is
            return binary_map


def extract_connected_regions(
    binary_map: np.ndarray,
    min_area_pixels: int = 100,
    max_regions: int = 20,
    prob_map: Optional[np.ndarray] = None,
) -> List[ChangedRegion]:
    """Extract connected components from a binary change map.

    Args:
        binary_map: H x W uint8 binary map.
        min_area_pixels: Minimum region area to include.
        max_regions: Maximum number of regions to return (sorted by area).
        prob_map: Optional probability map for computing mean confidence per region.

    Returns:
        List of ChangedRegion sorted by area (largest first).
    """
    regions: List[ChangedRegion] = []

    try:
        from scipy import ndimage

        labeled, num_features = ndimage.label(binary_map)

        for region_id in range(1, num_features + 1):
            mask = labeled == region_id
            area = int(mask.sum())

            if area < min_area_pixels:
                continue

            ys, xs = np.where(mask)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
            centroid = (float(ys.mean()), float(xs.mean()))

            # Mean confidence from probability map if available
            if prob_map is not None:
                mean_conf = float(prob_map[mask].mean())
            else:
                mean_conf = 1.0

            regions.append(ChangedRegion(
                region_id=region_id,
                bbox=bbox,
                area_pixels=area,
                centroid=centroid,
                mean_confidence=mean_conf,
            ))

    except ImportError:
        # Minimal fallback: treat entire changed area as one region
        changed_pixels = binary_map.sum()
        if changed_pixels >= min_area_pixels:
            ys, xs = np.where(binary_map > 0)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
            centroid = (float(ys.mean()), float(xs.mean()))
            mean_conf = float(prob_map[binary_map > 0].mean()) if prob_map is not None else 1.0
            regions.append(ChangedRegion(
                region_id=1,
                bbox=bbox,
                area_pixels=int(changed_pixels),
                centroid=centroid,
                mean_confidence=mean_conf,
            ))

    # Sort by area (largest first) and limit
    regions.sort(key=lambda r: r.area_pixels, reverse=True)
    return regions[:max_regions]


def postprocess_change_map(
    prob_map: Optional[np.ndarray] = None,
    binary_map: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_region_area: int = 100,
    max_regions: int = 20,
    morphology_kernel: int = 3,
) -> PostprocessingResult:
    """Full postprocessing pipeline.

    Args:
        prob_map: Optional probability map (H x W float [0, 1]).
        binary_map: Optional pre-thresholded binary map (H x W uint8).
        threshold: Threshold for probability map.
        min_region_area: Minimum connected component area.
        max_regions: Maximum regions to extract.
        morphology_kernel: Morphological kernel size (0 to disable).

    Returns:
        PostprocessingResult with binary map, regions, and statistics.
    """
    if binary_map is None and prob_map is None:
        raise ValueError("At least one of prob_map or binary_map must be provided.")

    # Create binary map from probability if not provided
    if binary_map is None:
        binary_map = threshold_probability_map(prob_map, threshold)

    # Apply morphological cleanup
    cleaned = morphological_cleanup(binary_map, morphology_kernel)

    # Extract connected regions
    regions = extract_connected_regions(
        cleaned,
        min_area_pixels=min_region_area,
        max_regions=max_regions,
        prob_map=prob_map,
    )

    total_pixels = int(cleaned.shape[0] * cleaned.shape[1])
    changed_pixels = int(cleaned.sum())
    ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0

    return PostprocessingResult(
        binary_map=cleaned,
        regions=regions,
        changed_pixel_count=changed_pixels,
        total_pixel_count=total_pixels,
        changed_pixel_ratio=ratio,
        parameters={
            "threshold": threshold,
            "min_region_area": min_region_area,
            "max_regions": max_regions,
            "morphology_kernel": morphology_kernel,
        },
    )
