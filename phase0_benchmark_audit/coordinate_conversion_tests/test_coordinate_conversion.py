import re
import math
from typing import List, Tuple, Dict, Any

class CoordinateConverter:
    """
    Authoritative Coordinate Conversion Pipeline:
    MODEL BOX (Qwen2.5-VL) -> CANONICAL BOX -> PIXEL BOX -> BENCHMARK BOX (VRSBench)
    
    Model Box (Qwen): [ymin, xmin, ymax, xmax] in [0, 1000] integer scale
    Canonical Box: [xmin, ymin, xmax, ymax] in [0.0, 1.0] normalized float scale
    Pixel Box: [xmin_px, ymin_px, xmax_px, ymax_px] in [0, W] x [0, H] integer pixels
    Benchmark Box (VRSBench):
       - obj_corner: 8-element polygon [x1, y1, x2, y2, x3, y3, x4, y4] in [0.0, 1.0]
       - ground_truth: "{<xmin><ymin><xmax><ymax>}" in [0, 100] integer scale
    """

    @staticmethod
    def parse_qwen_output(text: str) -> List[float]:
        """
        Extract [ymin, xmin, ymax, xmax] in [0, 1000] from Qwen output.
        Handles both <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|> and [ymin, xmin, ymax, xmax].
        """
        # Case 1: Standard Qwen box tokens
        box_pattern = r"<\|box_start\|>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*<\|box_end\|>"
        m = re.search(box_pattern, text)
        if m:
            return [float(m.group(i)) for i in range(1, 5)]

        # Case 2: Bracketed numbers [ymin, xmin, ymax, xmax]
        bracket_pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        m2 = re.search(bracket_pattern, text)
        if m2:
            return [float(m2.group(i)) for i in range(1, 5)]

        # Case 3: Fallback first 4 integers
        nums = re.findall(r"\d+", text)
        if len(nums) >= 4:
            return [float(nums[i]) for i in range(4)]

        raise ValueError(f"Could not parse bounding box from model output: {text}")

    @classmethod
    def model_to_canonical(cls, model_box: List[float]) -> List[float]:
        """
        MODEL BOX [ymin_1000, xmin_1000, ymax_1000, xmax_1000]
        -> CANONICAL BOX [xmin_norm, ymin_norm, xmax_norm, ymax_norm] in [0.0, 1.0]
        """
        ymin_1000, xmin_1000, ymax_1000, xmax_1000 = model_box
        
        # Clamp to [0, 1000]
        ymin = max(0.0, min(1000.0, ymin_1000)) / 1000.0
        xmin = max(0.0, min(1000.0, xmin_1000)) / 1000.0
        ymax = max(0.0, min(1000.0, ymax_1000)) / 1000.0
        xmax = max(0.0, min(1000.0, xmax_1000)) / 1000.0

        # Ensure min < max
        x1, x2 = min(xmin, xmax), max(xmin, xmax)
        y1, y2 = min(ymin, ymax), max(ymin, ymax)

        return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]

    @classmethod
    def canonical_to_pixel(cls, canonical_box: List[float], width: int, height: int) -> List[int]:
        """
        CANONICAL BOX [xmin, ymin, xmax, ymax] in [0.0, 1.0]
        -> PIXEL BOX [xmin_px, ymin_px, xmax_px, ymax_px] in [0, W] x [0, H]
        """
        x1, y1, x2, y2 = canonical_box
        px_x1 = int(round(x1 * width))
        px_y1 = int(round(y1 * height))
        px_x2 = int(round(x2 * width))
        px_y2 = int(round(y2 * height))

        # Clamp to image boundaries
        px_x1 = max(0, min(width, px_x1))
        px_y1 = max(0, min(height, px_y1))
        px_x2 = max(0, min(width, px_x2))
        px_y2 = max(0, min(height, px_y2))

        return [px_x1, px_y1, px_x2, px_y2]

    @classmethod
    def pixel_to_canonical(cls, pixel_box: List[int], width: int, height: int) -> List[float]:
        """
        PIXEL BOX -> CANONICAL BOX [0.0, 1.0]
        """
        x1, y1, x2, y2 = pixel_box
        return [
            round(x1 / max(1, width), 6),
            round(y1 / max(1, height), 6),
            round(x2 / max(1, width), 6),
            round(y2 / max(1, height), 6),
        ]

    @classmethod
    def parse_vrsbench_ground_truth(cls, gt_record: Dict[str, Any]) -> List[float]:
        """
        Parse VRSBench ground truth into CANONICAL BOX [xmin, ymin, xmax, ymax] in [0.0, 1.0].
        Uses obj_corner if available (high precision), else parses ground_truth tag.
        """
        corners = gt_record.get("obj_corner", [])
        if len(corners) >= 8:
            # Even indices: x coordinates
            xs = [corners[i] for i in range(0, 8, 2)]
            # Odd indices: y coordinates
            ys = [corners[i] for i in range(1, 8, 2)]
            return [round(min(xs), 6), round(min(ys), 6), round(max(xs), 6), round(max(ys), 6)]

        # Fallback to ground_truth string: "{<xmin><ymin><xmax><ymax>}" in [0, 100]
        gt_str = gt_record.get("ground_truth", "")
        nums = re.findall(r"\d+", gt_str)
        if len(nums) >= 4:
            x1, y1, x2, y2 = [float(nums[i]) / 100.0 for i in range(4)]
            return [round(min(x1, x2), 6), round(min(y1, y2), 6), round(max(x1, x2), 6), round(max(y1, y2), 6)]

        raise ValueError(f"Cannot parse VRSBench record: {gt_record}")

    @staticmethod
    def compute_canonical_iou(box_a: List[float], box_b: List[float]) -> float:
        """
        Compute IoU between two CANONICAL BOXES [xmin, ymin, xmax, ymax] in [0.0, 1.0].
        """
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return round(inter_area / union_area, 6)


def run_unit_tests():
    print("=" * 70)
    print("RUNNING 20+ DETERMINISTIC COORDINATE CONVERSION TESTS")
    print("=" * 70)

    conv = CoordinateConverter()
    tests_passed = 0
    total_tests = 25

    # Test 1: Standard Qwen Token Parsing
    t1 = "<|box_start|>(400,250),(600,330)<|box_end|>"
    mb1 = conv.parse_qwen_output(t1)
    assert mb1 == [400, 250, 600, 330], f"Test 1 Failed: {mb1}"
    tests_passed += 1

    # Test 2: Bracket Parsing Fallback
    t2 = "The vehicle is located at [400, 250, 600, 330]."
    mb2 = conv.parse_qwen_output(t2)
    assert mb2 == [400, 250, 600, 330], f"Test 2 Failed: {mb2}"
    tests_passed += 1

    # Test 3: Model to Canonical Conversion (Axis Order y,x,y,x -> x,y,x,y)
    cb3 = conv.model_to_canonical(mb1)
    assert cb3 == [0.25, 0.40, 0.33, 0.60], f"Test 3 Failed: {cb3}"
    tests_passed += 1

    # Test 4: Canonical to Pixel Conversion (1024x1024)
    px4 = conv.canonical_to_pixel(cb3, 1024, 1024)
    assert px4 == [256, 410, 338, 614], f"Test 4 Failed: {px4}"
    tests_passed += 1

    # Test 5: Pixel to Canonical Roundtrip (1024x1024)
    cb5 = conv.pixel_to_canonical(px4, 1024, 1024)
    assert abs(cb5[0] - cb3[0]) < 0.001 and abs(cb5[1] - cb3[1]) < 0.001, f"Test 5 Failed: {cb5}"
    tests_passed += 1

    # Test 6: VRSBench obj_corner Parsing
    vrs_rec6 = {
        "obj_corner": [0.333984, 0.589844, 0.287109, 0.603516, 0.255859, 0.416016, 0.289062, 0.400391],
        "ground_truth": "{<25><40><33><60>}"
    }
    cb6 = conv.parse_vrsbench_ground_truth(vrs_rec6)
    assert abs(cb6[0] - 0.255859) < 1e-4 and abs(cb6[1] - 0.400391) < 1e-4, f"Test 6 Failed: {cb6}"
    tests_passed += 1

    # Test 7: Canonical IoU Perfect Match
    iou7 = conv.compute_canonical_iou([0.25, 0.40, 0.33, 0.60], [0.25, 0.40, 0.33, 0.60])
    assert iou7 == 1.0, f"Test 7 Failed: {iou7}"
    tests_passed += 1

    # Test 8: Canonical IoU Zero Overlap Disjoint
    iou8 = conv.compute_canonical_iou([0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6])
    assert iou8 == 0.0, f"Test 8 Failed: {iou8}"
    tests_passed += 1

    # Test 9: Coordinate Inversion Flaw Demonstration (The bug in Colab runner)
    # Correct box: x=[0.25, 0.33], y=[0.40, 0.60]
    # Inverted box: x=[0.40, 0.60], y=[0.25, 0.33]
    iou_bug = conv.compute_canonical_iou([0.25, 0.40, 0.33, 0.60], [0.40, 0.25, 0.60, 0.33])
    assert iou_bug == 0.0, f"Test 9 Failed: {iou_bug}"
    tests_passed += 1

    # Test 10: Boundary Clamping (> 1000)
    mb10 = [1050, -50, 1200, 800]
    cb10 = conv.model_to_canonical(mb10)
    assert cb10[0] == 0.0 and cb10[2] == 0.8 and cb10[3] == 1.0, f"Test 10 Failed: {cb10}"
    tests_passed += 1

    # Test 11: Inverted Min/Max Fix
    mb11 = [600, 400, 200, 100] # ymax first, then ymin; xmax first, then xmin
    cb11 = conv.model_to_canonical(mb11)
    assert cb11 == [0.1, 0.2, 0.4, 0.6], f"Test 11 Failed: {cb11}"
    tests_passed += 1

    # Test 12: 512x512 Pixel Scaling
    cb12 = [0.1, 0.2, 0.5, 0.6]
    px12 = conv.canonical_to_pixel(cb12, 512, 512)
    assert px12 == [51, 102, 256, 307], f"Test 12 Failed: {px12}"
    tests_passed += 1

    # Test 13: 2048x2048 Pixel Scaling
    px13 = conv.canonical_to_pixel(cb12, 2048, 2048)
    assert px13 == [205, 410, 1024, 1229], f"Test 13 Failed: {px13}"
    tests_passed += 1

    # Test 14: Fallback String Ground Truth Parsing
    vrs_rec14 = {"ground_truth": "{<87><29><92><36>}"}
    cb14 = conv.parse_vrsbench_ground_truth(vrs_rec14)
    assert cb14 == [0.87, 0.29, 0.92, 0.36], f"Test 14 Failed: {cb14}"
    tests_passed += 1

    # Test 15: Acc@0.5 Thresholding (Passing)
    box_t15_a = [0.20, 0.20, 0.40, 0.40] # Area = 0.04
    box_t15_b = [0.20, 0.20, 0.35, 0.40] # Area = 0.03, Inter = 0.03, Union = 0.04, IoU = 0.75
    iou15 = conv.compute_canonical_iou(box_t15_a, box_t15_b)
    assert iou15 >= 0.50, f"Test 15 Failed: {iou15}"
    tests_passed += 1

    # Test 16: Acc@0.5 Thresholding (Failing)
    box_t16_a = [0.20, 0.20, 0.40, 0.40] # Area = 0.04
    box_t16_b = [0.35, 0.35, 0.55, 0.55] # Inter = 0.05 * 0.05 = 0.0025, IoU = 0.0025 / 0.0775 ~ 0.032
    iou16 = conv.compute_canonical_iou(box_t16_a, box_t16_b)
    assert iou16 < 0.50, f"Test 16 Failed: {iou16}"
    tests_passed += 1

    # Test 17: Acc@0.7 Thresholding (Strict Passing)
    box_t17_a = [0.10, 0.10, 0.50, 0.50]
    box_t17_b = [0.10, 0.10, 0.48, 0.50]
    iou17 = conv.compute_canonical_iou(box_t17_a, box_t17_b)
    assert iou17 >= 0.70, f"Test 17 Failed: {iou17}"
    tests_passed += 1

    # Test 18: Tiny Target Box Handling (Vehicle Scale: 10x10 px in 1024x1024)
    px18 = [500, 500, 510, 510]
    cb18 = conv.pixel_to_canonical(px18, 1024, 1024)
    assert cb18[2] > cb18[0] and cb18[3] > cb18[1], f"Test 18 Failed: {cb18}"
    tests_passed += 1

    # Test 19: Full Image Bounding Box
    cb19 = [0.0, 0.0, 1.0, 1.0]
    px19 = conv.canonical_to_pixel(cb19, 1000, 1000)
    assert px19 == [0, 0, 1000, 1000], f"Test 19 Failed: {px19}"
    tests_passed += 1

    # Test 20: Degenerate Zero-Area Box
    iou20 = conv.compute_canonical_iou([0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.5, 0.5])
    assert iou20 == 0.0, f"Test 20 Failed: {iou20}"
    tests_passed += 1

    # Test 21: Negative Coordinates in Malformed Output
    mb21 = [-10, -20, 500, 500]
    cb21 = conv.model_to_canonical(mb21)
    assert cb21[0] == 0.0 and cb21[1] == 0.0, f"Test 21 Failed: {cb21}"
    tests_passed += 1

    # Test 22: Enclosure IoU (Box Inside Box)
    box_outer = [0.0, 0.0, 1.0, 1.0]
    box_inner = [0.25, 0.25, 0.75, 0.75] # Area 0.25 vs 1.0 -> IoU = 0.25
    iou22 = conv.compute_canonical_iou(box_outer, box_inner)
    assert abs(iou22 - 0.25) < 1e-4, f"Test 22 Failed: {iou22}"
    tests_passed += 1

    # Test 23: Touching Edges IoU (Zero Intersection)
    box_left = [0.0, 0.0, 0.5, 1.0]
    box_right = [0.5, 0.0, 1.0, 1.0]
    iou23 = conv.compute_canonical_iou(box_left, box_right)
    assert iou23 == 0.0, f"Test 23 Failed: {iou23}"
    tests_passed += 1

    # Test 24: Realistic VRSBench Ground Truth Match Simulation
    # Real item from dataset: P0003_0002.png
    gt_real = {"obj_corner": [0.87109375, 0.30859375, 0.8984375, 0.296875, 0.921875, 0.35546875, 0.888671875, 0.365234375]}
    cb_gt = conv.parse_vrsbench_ground_truth(gt_real)
    # Simulated perfect Qwen prediction in native model tokens: ymin=297, xmin=871, ymax=365, xmax=922
    qwen_pred = "<|box_start|>(297,871),(365,922)<|box_end|>"
    cb_pred = conv.model_to_canonical(conv.parse_qwen_output(qwen_pred))
    iou24 = conv.compute_canonical_iou(cb_pred, cb_gt)
    assert iou24 > 0.80, f"Test 24 Failed: IoU={iou24}"
    tests_passed += 1

    # Test 25: Simulated Flawed Runner Coordinate Inversion on Same Item
    # The flaw passed [xmin, ymin, xmax, ymax] into calculation where [ymin, xmin, ymax, xmax] was expected
    flawed_iou = conv.compute_canonical_iou([cb_pred[1], cb_pred[0], cb_pred[3], cb_pred[2]], cb_gt)
    assert flawed_iou == 0.0, f"Test 25 Failed: Flawed IoU was {flawed_iou}"
    tests_passed += 1

    print(f"ALL {tests_passed}/{total_tests} UNIT TESTS PASSED PERFECTLY!")
    return tests_passed, total_tests

if __name__ == '__main__':
    run_unit_tests()
