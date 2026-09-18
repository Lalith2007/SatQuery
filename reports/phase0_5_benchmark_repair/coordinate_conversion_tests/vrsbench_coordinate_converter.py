"""VRSBench & Qwen Coordinate Conversion Pipeline for SatQuery AI.

Provides mathematically verified bidirectional coordinate transformations between:
1. VRSBench Official Annotation: [x1, y1, x2, y2] in [0, 100] scale ({<x1><y1><x2><y2>})
2. VRSBench 8-Corner Polygons: [x1, y1, x2, y2, x3, y3, x4, y4] in [0, 1.0] scale
3. Canonical Pixel Bounding Box: [xmin, ymin, xmax, ymax] in absolute image pixels
4. Qwen2.5-VL Model Output: [ymin, xmin, ymax, xmax] in [0, 1000] scale

Includes exact reference IoU from the official VRSBench repository (lx709/VRSBench).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class BoundingBox:
    """Canonical bounding box representation in [xmin, ymin, xmax, ymax]."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_list(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


class CoordinateConverter:
    """Production coordinate converter between VRSBench, Qwen, and Canonical spaces."""

    @staticmethod
    def clip_and_order_box(
        c1: float, c2: float, c3: float, c4: float,
        min_val: float = 0.0, max_val: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Order coordinates so min <= max for [xmin, ymin, xmax, ymax], and clip to bounds."""
        xmin = min(c1, c3)
        xmax = max(c1, c3)
        ymin = min(c2, c4)
        ymax = max(c2, c4)

        if min_val is not None:
            xmin = max(min_val, xmin)
            ymin = max(min_val, ymin)
            xmax = max(min_val, xmax)
            ymax = max(min_val, ymax)

        if max_val is not None:
            xmin = min(max_val, xmin)
            ymin = min(max_val, ymin)
            xmax = min(max_val, xmax)
            ymax = min(max_val, ymax)

        return (xmin, ymin, xmax, ymax)

    @classmethod
    def parse_vrsbench_ground_truth(
        cls, gt_val: Union[str, List[float], Tuple[float, ...]]
    ) -> Tuple[float, float, float, float]:
        """Parse VRSBench ground truth from '{<x1><y1><x2><y2>}' or list into [x1, y1, x2, y2] (0-100)."""
        if isinstance(gt_val, (list, tuple)):
            if len(gt_val) >= 4:
                return cls.clip_and_order_box(
                    float(gt_val[0]), float(gt_val[1]), float(gt_val[2]), float(gt_val[3]),
                    min_val=0.0, max_val=100.0,
                )
            raise ValueError(f"List must contain at least 4 coordinates, got {gt_val}")

        text = str(gt_val).strip()

        # Check for VRSBench official tag format: {<x1><y1><x2><y2>}
        tag_match = re.findall(r"<(\d+)>", text)
        if len(tag_match) >= 4:
            c1, c2, c3, c4 = [float(x) for x in tag_match[:4]]
            return cls.clip_and_order_box(c1, c2, c3, c4, min_val=0.0, max_val=100.0)

        # Check for floats first
        floats = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
        if len(floats) >= 4:
            c1, c2, c3, c4 = [float(x) for x in floats[:4]]
            # If coordinates are in [0, 1], scale to [0, 100]
            if max(c1, c2, c3, c4) <= 1.5:
                c1 *= 100.0
                c2 *= 100.0
                c3 *= 100.0
                c4 *= 100.0
            return cls.clip_and_order_box(c1, c2, c3, c4, min_val=0.0, max_val=100.0)

        raise ValueError(f"Could not parse valid VRSBench ground truth from: {gt_val}")

    @classmethod
    def parse_vrsbench_corners(
        cls, corners: List[float], scale_to_100: bool = True
    ) -> Tuple[float, float, float, float]:
        """Parse VRSBench 8-corner polygon [x1, y1, x2, y2, x3, y3, x4, y4] to [x1, y1, x2, y2].

        Even indices (0, 2, 4, 6) are x coordinates.
        Odd indices (1, 3, 5, 7) are y coordinates.
        """
        if len(corners) < 8:
            raise ValueError(f"Expected at least 8 corner values, got {len(corners)}")

        xs = [float(corners[i]) for i in range(0, 8, 2)]
        ys = [float(corners[i]) for i in range(1, 8, 2)]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        if scale_to_100 and max(xmax, ymax) <= 1.5:
            xmin *= 100.0
            xmax *= 100.0
            ymin *= 100.0
            ymax *= 100.0

        max_bound = 100.0 if scale_to_100 else None
        return cls.clip_and_order_box(xmin, ymin, xmax, ymax, min_val=0.0, max_val=max_bound)

    @classmethod
    def parse_qwen_output(
        cls, pred_text: str
    ) -> Tuple[float, float, float, float]:
        """Extract [ymin, xmin, ymax, xmax] in [0, 1000] scale from Qwen raw output string."""
        text = str(pred_text).strip()

        # Check for tags <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
        tag_match = re.search(r"<\|box_start\|>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>", text)
        if tag_match:
            y1, x1, y2, x2 = [float(x) for x in tag_match.groups()]
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            return (
                max(0.0, min(1000.0, ymin)),
                max(0.0, min(1000.0, xmin)),
                max(0.0, min(1000.0, ymax)),
                max(0.0, min(1000.0, xmax)),
            )

        # Look for [ymin, xmin, ymax, xmax] pattern
        bracket_match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        if bracket_match:
            y1, x1, y2, x2 = [float(x) for x in bracket_match.groups()]
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            return (
                max(0.0, min(1000.0, ymin)),
                max(0.0, min(1000.0, xmin)),
                max(0.0, min(1000.0, ymax)),
                max(0.0, min(1000.0, xmax)),
            )

        # Fallback to float search
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
        if len(numbers) >= 4:
            y1, x1, y2, x2 = [float(x) for x in numbers[:4]]
            # If coordinates are in [0, 1], scale to [0, 1000]
            if max(y1, x1, y2, x2) <= 1.5:
                y1 *= 1000.0
                x1 *= 1000.0
                y2 *= 1000.0
                x2 *= 1000.0
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            return (
                max(0.0, min(1000.0, ymin)),
                max(0.0, min(1000.0, xmin)),
                max(0.0, min(1000.0, ymax)),
                max(0.0, min(1000.0, xmax)),
            )

        raise ValueError(f"Could not extract 4 bounding box coordinates from Qwen output: '{pred_text}'")

    @classmethod
    def qwen_to_canonical(
        cls, qwen_box: Union[Tuple[float, float, float, float], List[float]],
        image_width: float, image_height: float
    ) -> BoundingBox:
        """Convert Qwen [ymin, xmin, ymax, xmax] in [0, 1000] to Canonical Pixel Box [xmin, ymin, xmax, ymax]."""
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = qwen_box[:4]

        xmin = (xmin_norm / 1000.0) * image_width
        xmax = (xmax_norm / 1000.0) * image_width
        ymin = (ymin_norm / 1000.0) * image_height
        ymax = (ymax_norm / 1000.0) * image_height

        c_xmin, c_ymin, c_xmax, c_ymax = cls.clip_and_order_box(
            xmin, ymin, xmax, ymax, min_val=0.0,
        )
        return BoundingBox(
            xmin=min(image_width, c_xmin),
            ymin=min(image_height, c_ymin),
            xmax=min(image_width, c_xmax),
            ymax=min(image_height, c_ymax),
        )

    @classmethod
    def canonical_to_qwen(
        cls, canonical_box: Union[BoundingBox, Tuple[float, float, float, float], List[float]],
        image_width: float, image_height: float
    ) -> Tuple[int, int, int, int]:
        """Convert Canonical Pixel Box [xmin, ymin, xmax, ymax] to Qwen [ymin, xmin, ymax, xmax] in [0, 1000]."""
        if isinstance(canonical_box, BoundingBox):
            xmin, ymin, xmax, ymax = canonical_box.to_tuple()
        else:
            xmin, ymin, xmax, ymax = canonical_box[:4]

        xmin_norm = int(round(max(0.0, min(1.0, xmin / max(1.0, image_width))) * 1000.0))
        xmax_norm = int(round(max(0.0, min(1.0, xmax / max(1.0, image_width))) * 1000.0))
        ymin_norm = int(round(max(0.0, min(1.0, ymin / max(1.0, image_height))) * 1000.0))
        ymax_norm = int(round(max(0.0, min(1.0, ymax / max(1.0, image_height))) * 1000.0))

        return (
            min(ymin_norm, ymax_norm),
            min(xmin_norm, xmax_norm),
            max(ymin_norm, ymax_norm),
            max(xmin_norm, xmax_norm),
        )

    @classmethod
    def canonical_to_vrsbench(
        cls, canonical_box: Union[BoundingBox, Tuple[float, float, float, float], List[float]],
        image_width: float, image_height: float
    ) -> Tuple[int, int, int, int]:
        """Convert Canonical Pixel Box to VRSBench [x1, y1, x2, y2] in [0, 100] integer scale."""
        if isinstance(canonical_box, BoundingBox):
            xmin, ymin, xmax, ymax = canonical_box.to_tuple()
        else:
            xmin, ymin, xmax, ymax = canonical_box[:4]

        x1 = int(round(max(0.0, min(1.0, xmin / max(1.0, image_width))) * 100.0))
        x2 = int(round(max(0.0, min(1.0, xmax / max(1.0, image_width))) * 100.0))
        y1 = int(round(max(0.0, min(1.0, ymin / max(1.0, image_height))) * 100.0))
        y2 = int(round(max(0.0, min(1.0, ymax / max(1.0, image_height))) * 100.0))

        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @classmethod
    def vrsbench_to_canonical(
        cls, vrs_box: Union[Tuple[float, float, float, float], List[float]],
        image_width: float, image_height: float
    ) -> BoundingBox:
        """Convert VRSBench [x1, y1, x2, y2] (in 0-100 scale) to Canonical Pixel Box."""
        x1, y1, x2, y2 = vrs_box[:4]

        xmin = (x1 / 100.0) * image_width
        xmax = (x2 / 100.0) * image_width
        ymin = (y1 / 100.0) * image_height
        ymax = (y2 / 100.0) * image_height

        c_xmin, c_ymin, c_xmax, c_ymax = cls.clip_and_order_box(xmin, ymin, xmax, ymax, min_val=0.0)
        return BoundingBox(
            xmin=min(image_width, c_xmin),
            ymin=min(image_height, c_ymin),
            xmax=min(image_width, c_xmax),
            ymax=min(image_height, c_ymax),
        )

    @classmethod
    def qwen_to_vrsbench(
        cls, qwen_box: Union[Tuple[float, float, float, float], List[float]]
    ) -> Tuple[int, int, int, int]:
        """Direct conversion from Qwen [ymin, xmin, ymax, xmax] (0-1000) to VRSBench [x1, y1, x2, y2] (0-100)."""
        ymin, xmin, ymax, xmax = qwen_box[:4]

        x1 = int(round((xmin / 1000.0) * 100.0))
        y1 = int(round((ymin / 1000.0) * 100.0))
        x2 = int(round((xmax / 1000.0) * 100.0))
        y2 = int(round((ymax / 1000.0) * 100.0))

        x1 = max(0, min(100, x1))
        y1 = max(0, min(100, y1))
        x2 = max(0, min(100, x2))
        y2 = max(0, min(100, y2))

        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @staticmethod
    def compute_official_vrsbench_iou(
        bbox1: Union[Tuple[float, ...], List[float]],
        bbox2: Union[Tuple[float, ...], List[float]],
    ) -> float:
        """Exact reference computeIoU implementation from lx709/VRSBench/eval_fianl/eval_utils.py.

        Operates on [x1, y1, x2, y2] boxes with discrete coordinate expansion (+ 1.0).
        """
        x1, y1, x2, y2 = bbox1[:4]
        x3, y3, x4, y4 = bbox2[:4]

        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        intersection_area = max(0.0, intersection_x2 - intersection_x1 + 1.0) * max(0.0, intersection_y2 - intersection_y1 + 1.0)
        bbox1_area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        bbox2_area = (x4 - x3 + 1.0) * (y4 - y3 + 1.0)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area <= 0.0:
            return 0.0
        return float(intersection_area / union_area)

    @staticmethod
    def compute_standard_box_iou(
        box_a: Union[Tuple[float, ...], List[float]],
        box_b: Union[Tuple[float, ...], List[float]],
    ) -> float:
        """Continuous standard IoU on [xmin, ymin, xmax, ymax] without +1 pixel discrete bias."""
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]

        inter_xmin = max(ax1, bx1)
        inter_ymin = max(ay1, by1)
        inter_xmax = min(ax2, bx2)
        inter_ymax = min(ay2, by2)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)
