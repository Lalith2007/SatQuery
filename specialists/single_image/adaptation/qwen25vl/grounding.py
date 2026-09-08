"""Grounding parser and natural-language cleaner for Qwen2.5-VL Remote-Sensing outputs.

Extracts structured bounding boxes, removes raw special tokens from conversational text,
computes IoU, and packages parsed detections into canonical SatQuery Evidence objects.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger
from core.schemas import Evidence, EvidenceType
from specialists.single_image.adaptation.qwen25vl.bbox_codec import BoxCodec

logger = get_logger("qwen25vl_grounding")

# Regex pattern for cleaning special tokens from natural-language text
SPECIAL_TOKEN_CLEAN_PATTERN = re.compile(
    r"<\|object_ref_start\|>|<\|object_ref_end\|>|<\|box_start\|>|<\|box_end\|>|<\|vision_start\|>|<\|vision_end\|>|<\|image_pad\|>"
)
# Coordinate block pattern in natural text: (ymin,xmin),(ymax,xmax)
COORD_BLOCK_PATTERN = re.compile(r"\(\d+\s*,\s*\d+\)\s*,\s*\(\d+\s*,\s*\d+\)")


class QwenGroundingParser:
    """Parses and renders Qwen2.5-VL multimodal grounding responses."""

    @classmethod
    def clean_text_response(cls, raw_text: str) -> str:
        """Strip raw grounding syntax tokens while preserving natural conversational flow."""
        # Replace object_ref and box markers with clean inline references
        cleaned = COORD_BLOCK_PATTERN.sub("", raw_text)
        cleaned = SPECIAL_TOKEN_CLEAN_PATTERN.sub("", cleaned)
        # Clean extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Clean stray punctuation like "near the center , ."
        cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
        return cleaned

    @classmethod
    def parse_grounding_response(
        cls,
        raw_text: str,
        image_width: int,
        image_height: int,
        query: Optional[str] = None,
        image_id: Optional[str] = None,
        default_confidence: Optional[float] = None,
    ) -> Tuple[str, List[Evidence], Dict[str, Any]]:
        """Parse raw model output into clean conversational answer and structured Evidence list.
        
        Returns:
            Tuple of:
            - clean_text: User-facing conversational explanation
            - evidence_list: List of Evidence objects
            - metadata: Detailed parse telemetry (raw tokens, counts, parse status)
        """
        parsed_boxes = BoxCodec.decode_bbox(raw_text, width=image_width, height=image_height)
        evidence_list: List[Evidence] = []

        fallback_label = query if query and query.strip() else "detected_feature"

        for idx, item in enumerate(parsed_boxes):
            label = item["label"] if item["label"] != "detected_feature" else fallback_label
            # [ymin, xmin, ymax, xmax] normalized (0.0 to 1.0)
            norm_bbox = item["normalized_bbox"]
            pixel_bbox = item["pixel_bbox"]

            ev = Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label=label,
                confidence=default_confidence,  # None unless calibrated / measured
                data={
                    "bbox": norm_bbox,
                    "canonical_bbox": pixel_bbox,
                    "format": "[ymin, xmin, ymax, xmax]",
                    "coordinate_system": "normalized_image_coordinates (0.0 - 1.0)",
                    "pixel_coordinates": {
                        "x1": pixel_bbox[0],
                        "y1": pixel_bbox[1],
                        "x2": pixel_bbox[2],
                        "y2": pixel_bbox[3],
                        "width": image_width,
                        "height": image_height,
                    },
                    "qwen_coords": item["qwen_coords"],
                    "detection_index": idx,
                },
                image_id=image_id,
            )
            evidence_list.append(ev)

        clean_text = cls.clean_text_response(raw_text)
        if not clean_text:
            if evidence_list:
                labels_str = ", ".join([e.label for e in evidence_list])
                clean_text = f"Identified {labels_str} at the highlighted coordinates."
            else:
                clean_text = "No distinct features could be localized for the specified query."

        metadata = {
            "raw_text_length": len(raw_text),
            "parsed_boxes_count": len(evidence_list),
            "parse_status": "SUCCESS" if evidence_list else ("NO_BOXES_FOUND" if "<|box_start|>" not in raw_text else "MALFORMED_BOX_SYNTAX"),
            "has_native_box_tokens": "<|box_start|>" in raw_text,
        }

        return clean_text, evidence_list, metadata

    @staticmethod
    def calculate_iou(
        box1: List[float],
        box2: List[float],
        format_name: str = "pixel_xyxy",
    ) -> float:
        """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
        if format_name in {"pixel_xyxy", "normalized_0_1_xyxy"}:
            x1_a, y1_a, x2_a, y2_a = box1
            x1_b, y1_b, x2_b, y2_b = box2
        elif format_name == "normalized_0_1_ymin_xmin":
            y1_a, x1_a, y2_a, x2_a = box1
            y1_b, x1_b, y2_b, x2_b = box2
        else:
            raise ValueError(f"Unsupported format for IoU calculation: {format_name}")

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        return round(inter_area / union_area, 4)
