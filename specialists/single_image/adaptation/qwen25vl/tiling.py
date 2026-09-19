"""High-Resolution Satellite Tiling and Global Coordinate Translation Engine.

Enables processing of large-scale satellite scenes without resizing them to small thumbnails.
Provides tile extraction, local-to-global coordinate translation, and NMS box merging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.bbox_codec import BoxCodec
from specialists.single_image.adaptation.qwen25vl.grounding import QwenGroundingParser

logger = get_logger("satellite_tiling")


class SatelliteTilingEngine:
    """Slices large satellite rasters, coordinates distributed inference, and maps results to global space."""

    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 128,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.nms_iou_threshold = nms_iou_threshold

    def generate_tiles(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Slice PIL Image into overlapping tiles with explicit global bounding offsets."""
        width, height = image.size
        tiles: List[Dict[str, Any]] = []

        y_points = list(range(0, height, self.stride))
        x_points = list(range(0, width, self.stride))

        for tile_idx_y, y in enumerate(y_points):
            for tile_idx_x, x in enumerate(x_points):
                # Clamp window to image boundaries
                x_end = min(x + self.tile_size, width)
                y_end = min(y + self.tile_size, height)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)

                crop = image.crop((x_start, y_start, x_end, y_end))

                tiles.append({
                    "tile_index": len(tiles),
                    "grid_coords": (tile_idx_x, tile_idx_y),
                    "image": crop,
                    "x_offset": x_start,
                    "y_offset": y_start,
                    "width": x_end - x_start,
                    "height": y_end - y_start,
                })

        return tiles

    @classmethod
    def translate_local_to_global_bbox(
        cls,
        local_bbox_xyxy: List[float],
        x_offset: int,
        y_offset: int,
    ) -> List[float]:
        """Translate a local tile pixel bbox [x1, y1, x2, y2] to global image pixel coordinates."""
        x1, y1, x2, y2 = local_bbox_xyxy
        return [
            round(x1 + x_offset, 2),
            round(y1 + y_offset, 2),
            round(x2 + x_offset, 2),
            round(y2 + y_offset, 2),
        ]

    def merge_detections_nms(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression (NMS) on global bounding boxes."""
        thresh = iou_threshold if iou_threshold is not None else self.nms_iou_threshold
        if not detections:
            return []

        # Sort by confidence descending if available, else keep order
        sorted_dets = sorted(
            detections,
            key=lambda d: d.get("confidence") or 1.0,
            reverse=True,
        )

        keep: List[Dict[str, Any]] = []
        for det in sorted_dets:
            box = det["pixel_bbox"]
            is_suppressed = False
            for kept in keep:
                kept_box = kept["pixel_bbox"]
                # Only suppress if same or generic label
                if det.get("label") == kept.get("label"):
                    iou = QwenGroundingParser.calculate_iou(box, kept_box, format_name="pixel_xyxy")
                    if iou >= thresh:
                        is_suppressed = True
                        break
            if not is_suppressed:
                keep.append(det)

        return keep
