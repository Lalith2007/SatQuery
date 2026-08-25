"""Remote Sensing Visual Grounding and Location Token Parser for PaliGemma.

Parses location coordinate tokens (<loc0000>-<loc1023>) and normalized bounding boxes,
validates coordinate ranges, computes intersection-over-union (IoU),
and maps detected regions into canonical Division 1 Evidence objects.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger
from core.schemas import Evidence, EvidenceType

logger = get_logger("single_image_grounding")

# Regex pattern matching PaliGemma location tokens: <loc0000> to <loc1023>
LOC_TOKEN_PATTERN = re.compile(r"<loc(\d{4})>")

# Regex pattern matching raw coordinate formats e.g. [0.25, 0.30, 0.65, 0.80] or [250, 300, 650, 800]
RAW_BBOX_PATTERN = re.compile(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]")


class GroundingCoordinateParser:
    """Parses, validates, and normalizes visual grounding outputs for remote sensing."""

    @staticmethod
    def parse_location_tokens(
        raw_text: str,
        label: str = "detected_feature",
        image_id: Optional[str] = None,
        confidence: float = 0.90,
    ) -> List[Evidence]:
        """Parse PaliGemma <locXXXX> tokens into standardized bounding box Evidence."""
        matches = LOC_TOKEN_PATTERN.findall(raw_text)
        evidence_list: List[Evidence] = []

        # Location tokens come in 4-tuples: [ymin, xmin, ymax, xmax] scaled 0..1023
        if len(matches) >= 4:
            for i in range(0, len(matches) - 3, 4):
                try:
                    ymin_raw = int(matches[i])
                    xmin_raw = int(matches[i + 1])
                    ymax_raw = int(matches[i + 2])
                    xmax_raw = int(matches[i + 3])

                    ymin = round(min(max(ymin_raw / 1024.0, 0.0), 1.0), 4)
                    xmin = round(min(max(xmin_raw / 1024.0, 0.0), 1.0), 4)
                    ymax = round(min(max(ymax_raw / 1024.0, 0.0), 1.0), 4)
                    xmax = round(min(max(xmax_raw / 1024.0, 0.0), 1.0), 4)

                    # Ensure valid box dimensions
                    if ymax <= ymin:
                        ymax = min(ymin + 0.05, 1.0)
                    if xmax <= xmin:
                        xmax = min(xmin + 0.05, 1.0)

                    bbox = [ymin, xmin, ymax, xmax]
                    evidence = Evidence(
                        type=EvidenceType.BOUNDING_BOX,
                        label=label,
                        confidence=confidence,
                        data={
                            "bbox": bbox,
                            "format": "[ymin, xmin, ymax, xmax]",
                            "coordinate_system": "normalized_image_coordinates (0.0 - 1.0)",
                            "raw_tokens": [f"<loc{matches[i]}>", f"<loc{matches[i+1]}>", f"<loc{matches[i+2]}>", f"<loc{matches[i+3]}>"],
                        },
                        image_id=image_id,
                    )
                    evidence_list.append(evidence)
                except Exception as e:
                    logger.warning(f"Failed to parse location token quartet: {e}")

        # Fallback: check for raw bracketed bounding boxes e.g. [0.2, 0.3, 0.6, 0.8]
        if not evidence_list:
            raw_matches = RAW_BBOX_PATTERN.findall(raw_text)
            for m in raw_matches:
                try:
                    nums = [float(val) for val in m]
                    # Check if coordinates are 0..1000 or 0..1
                    if any(n > 1.0 for n in nums):
                        scaled = [round(n / 1000.0, 4) for n in nums]
                    else:
                        scaled = [round(n, 4) for n in nums]

                    evidence = Evidence(
                        type=EvidenceType.BOUNDING_BOX,
                        label=label,
                        confidence=confidence,
                        data={
                            "bbox": scaled,
                            "format": "[ymin, xmin, ymax, xmax]",
                            "coordinate_system": "normalized_image_coordinates (0.0 - 1.0)",
                        },
                        image_id=image_id,
                    )
                    evidence_list.append(evidence)
                except Exception as e:
                    logger.warning(f"Failed to parse raw bbox pattern: {e}")

        return evidence_list

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two [ymin, xmin, ymax, xmax] boxes."""
        ymin1, xmin1, ymax1, xmax1 = box1
        ymin2, xmin2, ymax2, xmax2 = box2

        # Determine intersection rectangle
        inter_ymin = max(ymin1, ymin2)
        inter_xmin = max(xmin1, xmin2)
        inter_ymax = min(ymax1, ymax2)
        inter_xmax = min(xmax1, xmax2)

        if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
            return 0.0

        inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
        box1_area = (ymax1 - ymin1) * (xmax1 - xmin1)
        box2_area = (ymax2 - ymin2) * (xmax2 - xmin2)

        union_area = box1_area + box2_area - inter_area
        if union_area <= 0:
            return 0.0

        return round(inter_area / union_area, 4)
