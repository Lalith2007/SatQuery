# SATQUERY AI — PHASE 0.5: COORDINATE CONVENTIONS & MATHEMATICAL CONVERSION PROOF

**Document Purpose:** Documents the mathematical root cause of the historical VRSBench Grounding IoU collapse ($0.0484$) and details the verified bidirectional coordinate conversion pipeline.

---

## 1. Forensic Anatomy of the 0.0484 IoU Collapse

In Phase 0, SatQuery AI audited why Qwen2.5-VL achieved $0.6711$ grounding mean IoU on BigEarthNet.txt Stage-1, but collapsed to $0.0484$ on VRSBench.

### The Axis Inversion & Index Swap Bug
In the legacy benchmark runner `colab_qwen_final_benchmark_runner.py` (lines 401-403):
```python
# OBSOLETE BUGGY CODE:
ys = [corners[i] for i in range(0, 8, 2)]  # Even indices taken as Y!
xs = [corners[i] for i in range(1, 8, 2)]  # Odd indices taken as X!
gt_box = [min(ys), min(xs), max(ys), max(xs)]
```
However, in VRSBench's official annotation format:
- `obj_corner` is structured as $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]$:
  - Even indices ($0, 2, 4, 6$) are **X coordinates**!
  - Odd indices ($1, 3, 5, 7$) are **Y coordinates**!
- By taking even indices as $y$ and odd as $x$, the evaluator swapped the horizontal and vertical axes of the ground truth.

### The Scale Mismatch Bug
- In VRSBench Hugging Face JSON files (`VRSBench_EVAL_referring.json`), annotations are normalized to **[0, 100]**:
  $$\text{ground\_truth} = \{<x_1><y_1><x_2><y_2>\}, \quad x, y \in [0, 100]$$
- Qwen2.5-VL outputs coordinates normalized to **[0, 1000]** in $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$ format.
- The legacy evaluator attempted manual division by $1000$ without mapping axes, comparing horizontal spans against vertical spans.
- **Mathematical Result:** Cross-evaluating orthogonal dimensions ($x$ vs $y$) forced the intersection area to zero for nearly all non-diagonal rectangular objects, producing the artificial $0.0484$ mean IoU.

---

## 2. Canonical Coordinate Architecture

Phase 0.5 implements a mathematically rigorous 3-space conversion architecture in [`scripts/vrsbench_coordinate_converter.py`](file:///Users/lalith/Desktop/Projects/SatQuery/scripts/vrsbench_coordinate_converter.py):

```
       +---------------------------------------------+
       |   VRSBench Official Space: [x1, y1, x2, y2] |
       |          Normalized to [0, 100]             |
       +---------------------------------------------+
                             ^
                             | (canonical_to_vrsbench / vrsbench_to_canonical)
                             v
       +---------------------------------------------+
       |     Canonical Pixel Box: BoundingBox        |
       |            [xmin, ymin, xmax, ymax]         |
       |            Absolute Image Pixels            |
       +---------------------------------------------+
                             ^
                             | (canonical_to_qwen / qwen_to_canonical)
                             v
       +---------------------------------------------+
       |     Qwen Model Space: [ymin, xmin, ymax, xmax] |
       |          Normalized to [0, 1000]            |
       +---------------------------------------------+
```

### Direct Mathematical Bridge (Qwen $\leftrightarrow$ VRSBench)
Because both coordinate systems are normalized representations of the underlying image dimensions:
$$x_{1, \text{vrs}} = \text{round}\left( \frac{x_{\min, \text{qwen}}}{1000} \times 100 \right), \quad y_{1, \text{vrs}} = \text{round}\left( \frac{y_{\min, \text{qwen}}}{1000} \times 100 \right)$$
$$x_{2, \text{vrs}} = \text{round}\left( \frac{x_{\max, \text{qwen}}}{1000} \times 100 \right), \quad y_{2, \text{vrs}} = \text{round}\left( \frac{y_{\max, \text{qwen}}}{1000} \times 100 \right)$$
Clamped strictly to $[0, 100]$ integers.

---

## 3. Unit Test Verification Results

The converter is backed by **30 deterministic unit tests** in `tests/test_vrsbench_coordinate_converter.py`:
- 100% of 30 tests pass.
- Proves identity, full-image, corner, wide aspect ratio ($1024 \times 512$), tall aspect ratio ($400 \times 800$), and out-of-bounds coordinate clipping.
- Proves that swapping $x$ and $y$ collapses IoU to $<0.15$, replicating the exact failure observed in the legacy runner.
- Proves that aligned coordinates achieve $\text{IoU} = 1.00$ on identical boxes and $>0.80$ on high-precision detections.
