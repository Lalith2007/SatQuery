"""Bounding Box Codec for Remote Sensing Visual Grounding.

Provides reversible conversions between:
1. Canonical pixel coordinates: [x1, y1, x2, y2]
2. Normalized coordinates [0.0 - 1.0]: [xmin, ymin, xmax, ymax]
3. Division 2/5 standard normalized coordinates [0.0 - 1.0]: [ymin, xmin, ymax, xmax]
4. Qwen native 1000-scale coordinate tokens: <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logging import get_logger

logger = get_logger("bbox_codec")

# Regex pattern for Qwen native box format: (ymin,xmin),(ymax,xmax)
QWEN_BOX_PATTERN = re.compile(
    r"<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>"
)
# Regex pattern for object reference spans
QWEN_OBJECT_REF_PATTERN = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>"
)
# Combined object ref + box pattern
QWEN_REF_AND_BOX_PATTERN = re.compile(
    r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>"
)


class BoxCodec:
    """Canonical bounding box encoding, decoding, and coordinate transformation engine."""

    @staticmethod
    def validate_bbox(
        bbox: Union[List[float], Tuple[float, ...]],
        format_name: str = "pixel_xyxy",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> bool:
        """Validate bounding box length, numerical validity, and ordering."""
        if len(bbox) != 4:
            return False

        try:
            c0, c1, c2, c3 = [float(x) for x in bbox]
        except (ValueError, TypeError):
            return False

        if format_name in {"pixel_xyxy", "pixel_xyxy"}:
            x1, y1, x2, y2 = c0, c1, c2, c3
            if x2 <= x1 or y2 <= y1:
                return False
            if x1 < 0 or y1 < 0:
                return False
            if width is not None and (x1 > width or x2 > width):
                return False
            if height is not None and (y1 > height or y2 > height):
                return False
            return True

        elif format_name == "normalized_0_1_xyxy":
            xmin, ymin, xmax, ymax = c0, c1, c2, c3
            if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
                return False
            return True

        elif format_name == "normalized_0_1_ymin_xmin":
            ymin, xmin, ymax, xmax = c0, c1, c2, c3
            if not (0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0):
                return False
            return True

        elif format_name == "qwen_1000_ymin_xmin":
            ymin, xmin, ymax, xmax = c0, c1, c2, c3
            if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
                return False
            return True

        return False

    @staticmethod
    def clip_bbox(
        bbox: List[float],
        width: int,
        height: int,
    ) -> List[float]:
        """Clip canonical pixel coordinates [x1, y1, x2, y2] to image boundaries."""
        x1 = max(0.0, min(float(bbox[0]), float(width)))
        y1 = max(0.0, min(float(bbox[1]), float(height)))
        x2 = max(x1 + 1.0, min(float(bbox[2]), float(width)))
        y2 = max(y1 + 1.0, min(float(bbox[3]), float(height)))
        return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]

    @classmethod
    def normalize_bbox(
        cls,
        bbox: List[float],
        width: int,
        height: int,
        target_format: str = "normalized_0_1_ymin_xmin",
    ) -> List[float]:
        """Convert canonical pixel coordinates [x1, y1, x2, y2] to normalized target format."""
        clipped = cls.clip_bbox(bbox, width, height)
        x1, y1, x2, y2 = clipped

        xmin_norm = x1 / width
        ymin_norm = y1 / height
        xmax_norm = x2 / width
        ymax_norm = y2 / height

        if target_format == "normalized_0_1_ymin_xmin":
            return [round(ymin_norm, 4), round(xmin_norm, 4), round(ymax_norm, 4), round(xmax_norm, 4)]
        elif target_format == "normalized_0_1_xyxy":
            return [round(xmin_norm, 4), round(ymin_norm, 4), round(xmax_norm, 4), round(ymax_norm, 4)]
        elif target_format == "qwen_1000_ymin_xmin":
            return [
                int(round(ymin_norm * 1000)),
                int(round(xmin_norm * 1000)),
                int(round(ymax_norm * 1000)),
                int(round(xmax_norm * 1000)),
            ]
        else:
            raise ValueError(f"Unknown target normalization format: {target_format}")

    @classmethod
    def denormalize_bbox(
        cls,
        bbox: List[float],
        width: int,
        height: int,
        source_format: str = "normalized_0_1_ymin_xmin",
    ) -> List[float]:
        """Convert normalized coordinates back to canonical pixel coordinates [x1, y1, x2, y2]."""
        if source_format == "normalized_0_1_ymin_xmin":
            ymin, xmin, ymax, xmax = bbox
            x1 = xmin * width
            y1 = ymin * height
            x2 = xmax * width
            y2 = ymax * height
        elif source_format == "normalized_0_1_xyxy":
            xmin, ymin, xmax, ymax = bbox
            x1 = xmin * width
            y1 = ymin * height
            x2 = xmax * width
            y2 = ymax * height
        elif source_format == "qwen_1000_ymin_xmin":
            ymin, xmin, ymax, xmax = bbox
            x1 = (xmin / 1000.0) * width
            y1 = (ymin / 1000.0) * height
            x2 = (xmax / 1000.0) * width
            y2 = (ymax / 1000.0) * height
        else:
            raise ValueError(f"Unknown source format: {source_format}")

        return cls.clip_bbox([x1, y1, x2, y2], width, height)

    @classmethod
    def encode_bbox(
        cls,
        bbox: List[float],
        width: int,
        height: int,
        label: Optional[str] = None,
        source_format: str = "pixel_xyxy",
    ) -> str:
        """Encode bounding box into Qwen native token format.
        
        Output format:
        <|object_ref_start|>label<|object_ref_end|><|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
        or without label:
        <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
        """
        if source_format == "pixel_xyxy":
            qwen_coords = cls.normalize_bbox(bbox, width, height, target_format="qwen_1000_ymin_xmin")
        elif source_format == "normalized_0_1_ymin_xmin":
            qwen_coords = [int(round(float(c) * 1000)) for c in bbox]
        elif source_format == "qwen_1000_ymin_xmin":
            qwen_coords = [int(c) for c in bbox]
        else:
            raise ValueError(f"Unsupported source format for encoding: {source_format}")

        ymin, xmin, ymax, xmax = qwen_coords
        # Clamp to [0, 1000]
        ymin = max(0, min(ymin, 1000))
        xmin = max(0, min(xmin, 1000))
        ymax = max(ymin + 1, min(ymax, 1000))
        xmax = max(xmin + 1, min(xmax, 1000))

        box_str = f"<|box_start|>({ymin},{xmin}),({ymax},{xmax})<|box_end|>"
        if label and label.strip():
            clean_label = label.strip()
            return f"<|object_ref_start|>{clean_label}<|object_ref_end|>{box_str}"
        return box_str

    @classmethod
    def decode_bbox(
        cls,
        token_str: str,
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        """Decode Qwen token strings into structured objects with canonical pixel coordinates."""
        results: List[Dict[str, Any]] = []

        # 1. First search for paired object_ref + box
        ref_matches = list(QWEN_REF_AND_BOX_PATTERN.finditer(token_str))
        consumed_spans = []

        for m in ref_matches:
            label = m.group(1).strip()
            ymin = int(m.group(2))
            xmin = int(m.group(3))
            ymax = int(m.group(4))
            xmax = int(m.group(5))

            canonical_xyxy = cls.denormalize_bbox(
                [ymin, xmin, ymax, xmax], width, height, source_format="qwen_1000_ymin_xmin"
            )
            norm_div2 = cls.normalize_bbox(
                canonical_xyxy, width, height, target_format="normalized_0_1_ymin_xmin"
            )

            results.append({
                "label": label if label else "detected_feature",
                "pixel_bbox": canonical_xyxy,
                "normalized_bbox": norm_div2,
                "qwen_coords": [ymin, xmin, ymax, xmax],
            })
            consumed_spans.append(m.span())

        # 2. Search for standalone boxes not captured by paired pattern
        box_matches = list(QWEN_BOX_PATTERN.finditer(token_str))
        for m in box_matches:
            # Check if this box was part of an already consumed span
            start, end = m.span()
            if any(c_start <= start and end <= c_end for c_start, c_end in consumed_spans):
                continue

            ymin = int(m.group(1))
            xmin = int(m.group(2))
            ymax = int(m.group(3))
            xmax = int(m.group(4))

            canonical_xyxy = cls.denormalize_bbox(
                [ymin, xmin, ymax, xmax], width, height, source_format="qwen_1000_ymin_xmin"
            )
            norm_div2 = cls.normalize_bbox(
                canonical_xyxy, width, height, target_format="normalized_0_1_ymin_xmin"
            )

            results.append({
                "label": "detected_feature",
                "pixel_bbox": canonical_xyxy,
                "normalized_bbox": norm_div2,
                "qwen_coords": [ymin, xmin, ymax, xmax],
            })

        return results
