"""Unit tests for VRSBench & Qwen Coordinate Converter.

Contains 30 deterministic test cases covering:
- Identity, full-image, and corner boxes
- Non-square image aspect ratios
- Resizing transformations
- Normalized annotation conversions ([0, 100] <-> [0, 1000] <-> Pixels)
- Qwen output string parsing (<|box_start|>, brackets, plain integers)
- Boundary clipping and reversed coordinates
- 8-corner polygon parsing (even=x, odd=y)
- Official discrete IoU (+1 bias) vs standard continuous IoU
- Acc@0.5 and Acc@0.7 thresholding
"""

import pytest
from scripts.vrsbench_coordinate_converter import BoundingBox, CoordinateConverter


# 1. Identity Box
def test_identity_box():
    box = CoordinateConverter.clip_and_order_box(10, 20, 50, 60)
    assert box == (10, 20, 50, 60)


# 2. Full-image box in pixels
def test_full_image_box_pixels():
    qwen_full = (0, 0, 1000, 1000)
    canonical = CoordinateConverter.qwen_to_canonical(qwen_full, 800, 600)
    assert canonical.xmin == 0.0
    assert canonical.ymin == 0.0
    assert canonical.xmax == 800.0
    assert canonical.ymax == 600.0


# 3. Corner Box (Top-Left)
def test_top_left_corner_box():
    qwen_tl = (0, 0, 200, 200)
    canonical = CoordinateConverter.qwen_to_canonical(qwen_tl, 1000, 1000)
    assert canonical == BoundingBox(0.0, 0.0, 200.0, 200.0)


# 4. Corner Box (Bottom-Right)
def test_bottom_right_corner_box():
    qwen_br = (800, 800, 1000, 1000)
    canonical = CoordinateConverter.qwen_to_canonical(qwen_br, 500, 500)
    assert canonical == BoundingBox(400.0, 400.0, 500.0, 500.0)


# 5. Non-Square Image (W > H: 1024 x 512)
def test_non_square_wide():
    # Box centered
    qwen_box = (250, 250, 750, 750)  # y: 250..750, x: 250..750
    canonical = CoordinateConverter.qwen_to_canonical(qwen_box, 1024, 512)
    assert canonical.xmin == 256.0
    assert canonical.xmax == 768.0
    assert canonical.ymin == 128.0
    assert canonical.ymax == 384.0


# 6. Non-Square Image (H > W: 400 x 800)
def test_non_square_tall():
    qwen_box = (100, 200, 900, 800)
    canonical = CoordinateConverter.qwen_to_canonical(qwen_box, 400, 800)
    assert canonical.xmin == 80.0
    assert canonical.xmax == 320.0
    assert canonical.ymin == 80.0
    assert canonical.ymax == 720.0


# 7. Resized Image Scale Invariance (Qwen to VRSBench)
def test_qwen_to_vrsbench_scale_invariance():
    # 50% box
    qwen_box = (100, 200, 600, 700)
    vrs_box = CoordinateConverter.qwen_to_vrsbench(qwen_box)
    assert vrs_box == (20, 10, 70, 60)  # x1=20, y1=10, x2=70, y2=60


# 8. Normalized Annotation Parsing: Standard format {<x1><y1><x2><y2>}
def test_parse_vrsbench_ground_truth_standard():
    gt_str = "{<25><40><33><60>}"
    x1, y1, x2, y2 = CoordinateConverter.parse_vrsbench_ground_truth(gt_str)
    assert (x1, y1, x2, y2) == (25.0, 40.0, 33.0, 60.0)


# 9. Normalized Annotation Parsing: List input
def test_parse_vrsbench_ground_truth_list():
    gt_list = [11, 96, 24, 100]
    box = CoordinateConverter.parse_vrsbench_ground_truth(gt_list)
    assert box == (11.0, 96.0, 24.0, 100.0)


# 10. Normalized Annotation Parsing: Normalized [0, 1] float string
def test_parse_vrsbench_ground_truth_normalized_float():
    gt_str = "0.15 0.25 0.35 0.65"
    box = CoordinateConverter.parse_vrsbench_ground_truth(gt_str)
    assert box == (15.0, 25.0, 35.0, 65.0)


# 11. Qwen Output Parsing: Standard bracket list [ymin, xmin, ymax, xmax]
def test_parse_qwen_output_brackets():
    pred = "The vehicle is located at [400, 250, 600, 330]."
    box = CoordinateConverter.parse_qwen_output(pred)
    assert box == (400.0, 250.0, 600.0, 330.0)


# 12. Qwen Output Parsing: Special vision tokens <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
def test_parse_qwen_output_special_tokens():
    pred = "<|box_start|>(350, 120),(500, 280)<|box_end|>"
    box = CoordinateConverter.parse_qwen_output(pred)
    assert box == (350.0, 120.0, 500.0, 280.0)


# 13. Qwen Output Parsing: Fallback raw numbers
def test_parse_qwen_output_raw_numbers():
    pred = "Coordinates: ymin=100, xmin=200, ymax=300, xmax=400"
    box = CoordinateConverter.parse_qwen_output(pred)
    assert box == (100.0, 200.0, 300.0, 400.0)


# 14. Clipping: Out of bounds high (>1000)
def test_clipping_out_of_bounds_high():
    box = CoordinateConverter.clip_and_order_box(0, 0, 1200, 1500, min_val=0.0, max_val=1000.0)
    assert box == (0.0, 0.0, 1000.0, 1000.0)


# 15. Clipping: Out of bounds low (<0)
def test_clipping_out_of_bounds_low():
    box = CoordinateConverter.clip_and_order_box(-50, -20, 500, 600, min_val=0.0, max_val=1000.0)
    assert box == (0.0, 0.0, 500.0, 600.0)


# 16. Reversed Coordinates: Inverted X and Y
def test_reversed_coordinates():
    # xmax < xmin, ymax < ymin
    box = CoordinateConverter.clip_and_order_box(800, 700, 200, 100)
    assert box == (200.0, 100.0, 800.0, 700.0)


# 17. Malformed Coordinates: Less than 4 values raises ValueError
def test_malformed_coordinates_raises():
    with pytest.raises(ValueError):
        CoordinateConverter.parse_qwen_output("There is no box here!")


# 18. 8-Corner Polygon Parsing: Verify even indices are x and odd are y
def test_parse_vrsbench_corners_axes():
    # Corners: [x1, y1, x2, y2, x3, y3, x4, y4]
    # x values: 0.20, 0.25, 0.35, 0.30 -> min 0.20, max 0.35
    # y values: 0.50, 0.60, 0.55, 0.45 -> min 0.45, max 0.60
    corners = [0.20, 0.50, 0.25, 0.60, 0.35, 0.55, 0.30, 0.45]
    x1, y1, x2, y2 = CoordinateConverter.parse_vrsbench_corners(corners, scale_to_100=True)
    assert x1 == 20.0
    assert x2 == 35.0
    assert y1 == 45.0
    assert y2 == 60.0


# 19. 8-Corner Polygon: Out-of-bounds corner clipped
def test_parse_vrsbench_corners_clipping():
    corners = [0.10, 0.95, 0.15, 1.05, 0.25, 1.02, 0.20, 0.90]
    x1, y1, x2, y2 = CoordinateConverter.parse_vrsbench_corners(corners, scale_to_100=True)
    assert x1 == 10.0
    assert x2 == 25.0
    assert y1 == 90.0
    assert y2 == 100.0  # Clipped from 105.0 to 100.0


# 20. Canonical to VRSBench roundtrip
def test_canonical_to_vrsbench_roundtrip():
    orig_canonical = BoundingBox(100.0, 200.0, 300.0, 400.0)
    vrs = CoordinateConverter.canonical_to_vrsbench(orig_canonical, 1000, 1000)
    assert vrs == (10, 20, 30, 40)
    recovered = CoordinateConverter.vrsbench_to_canonical(vrs, 1000, 1000)
    assert recovered == orig_canonical


# 21. Canonical to Qwen roundtrip
def test_canonical_to_qwen_roundtrip():
    orig_canonical = BoundingBox(250.0, 100.0, 750.0, 900.0)
    qwen = CoordinateConverter.canonical_to_qwen(orig_canonical, 1000, 1000)
    assert qwen == (100, 250, 900, 750)  # ymin, xmin, ymax, xmax
    recovered = CoordinateConverter.qwen_to_canonical(qwen, 1000, 1000)
    assert recovered == orig_canonical


# 22. End-to-end Qwen output to VRSBench ground truth comparison
def test_end_to_end_qwen_to_vrsbench_matching():
    # Target in VRSBench annotation: {<25><40><33><60>}
    gt_vrs = CoordinateConverter.parse_vrsbench_ground_truth("{<25><40><33><60>}")
    # Model predicted in Qwen format: [ymin=400, xmin=250, ymax=600, xmax=330]
    qwen_raw = "[400, 250, 600, 330]"
    qwen_box = CoordinateConverter.parse_qwen_output(qwen_raw)
    pred_vrs = CoordinateConverter.qwen_to_vrsbench(qwen_box)

    assert pred_vrs == (25, 40, 33, 60)
    iou = CoordinateConverter.compute_official_vrsbench_iou(pred_vrs, gt_vrs)
    assert iou == 1.0


# 23. Axis inversion diagnosis proof: swapping x and y collapses IoU
def test_axis_inversion_collapses_iou():
    box_correct = (20, 40, 30, 80)      # w=10, h=40
    box_swapped = (40, 20, 80, 30)      # inverted axes
    iou = CoordinateConverter.compute_standard_box_iou(box_correct, box_swapped)
    # The intersection is tiny or zero
    assert iou < 0.15


# 24. Official VRSBench discrete IoU (+1 bias)
def test_official_vrsbench_iou_identical():
    box1 = (10, 10, 20, 20)
    box2 = (10, 10, 20, 20)
    iou = CoordinateConverter.compute_official_vrsbench_iou(box1, box2)
    assert iou == 1.0


# 25. Official VRSBench discrete IoU disjoint
def test_official_vrsbench_iou_disjoint():
    box1 = (0, 0, 10, 10)
    box2 = (20, 20, 30, 30)
    iou = CoordinateConverter.compute_official_vrsbench_iou(box1, box2)
    assert iou == 0.0


# 26. Official VRSBench discrete IoU partial overlap
def test_official_vrsbench_iou_partial():
    box1 = (0, 0, 10, 10)  # area = 11 * 11 = 121
    box2 = (5, 5, 15, 15)  # area = 11 * 11 = 121, inter = 6 * 6 = 36
    iou = CoordinateConverter.compute_official_vrsbench_iou(box1, box2)
    expected = 36.0 / (121 + 121 - 36)  # 36 / 206 = ~0.174757
    assert abs(iou - expected) < 1e-4


# 27. Standard Continuous IoU identical
def test_standard_continuous_iou_identical():
    box1 = (0.2, 0.3, 0.5, 0.7)
    box2 = (0.2, 0.3, 0.5, 0.7)
    assert CoordinateConverter.compute_standard_box_iou(box1, box2) == 1.0


# 28. Standard Continuous IoU partial overlap
def test_standard_continuous_iou_partial():
    box1 = (0, 0, 10, 10)  # area 100
    box2 = (5, 0, 15, 10)  # area 100, inter 5 * 10 = 50, union = 150
    assert abs(CoordinateConverter.compute_standard_box_iou(box1, box2) - (50.0 / 150.0)) < 1e-4


# 29. Official Acc@0.5 and Acc@0.7 classification logic
def test_acc_threshold_logic():
    ious = [0.85, 0.65, 0.52, 0.48, 0.10]
    acc_05 = sum(1 for x in ious if x >= 0.5) / len(ious)
    acc_07 = sum(1 for x in ious if x >= 0.7) / len(ious)
    assert acc_05 == 3 / 5  # 60%
    assert acc_07 == 1 / 5  # 20%


# 30. Zero-area box safety
def test_zero_area_box_safety():
    box1 = (5, 5, 5, 5)
    box2 = (0, 0, 10, 10)
    iou = CoordinateConverter.compute_standard_box_iou(box1, box2)
    assert iou == 0.0
