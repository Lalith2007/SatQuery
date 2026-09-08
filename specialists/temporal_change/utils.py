"""Geospatial metadata extraction and formatting utilities.

Division 3 — Bi-Temporal Change Intelligence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.schemas import ImageInput


def extract_geospatial_summary(img: ImageInput) -> Dict[str, Any]:
    """Extract a safe summary of geospatial metadata from an ImageInput."""
    info: Dict[str, Any] = {
        "image_id": img.image_id,
        "width": img.width,
        "height": img.height,
        "channels": img.channel_count,
        "dtype": img.dtype,
        "format": img.format.value if img.format else None,
        "modality": img.modality.value if img.modality else None,
    }

    if img.geospatial:
        geo = img.geospatial
        info["crs"] = geo.crs
        info["geo_bounds"] = geo.geo_bounds
        info["resolution"] = geo.resolution
        info["sensor"] = geo.sensor
        info["nodata_value"] = geo.nodata_value
        if geo.acquisition_timestamp:
            info["acquisition_timestamp"] = geo.acquisition_timestamp.isoformat()

    return info


def format_pixel_ratio_as_text(ratio: float) -> str:
    """Format a changed pixel ratio as human-readable percentage text."""
    pct = round(ratio * 100, 2)
    if pct < 0.01:
        return "less than 0.01%"
    return f"{pct}%"
