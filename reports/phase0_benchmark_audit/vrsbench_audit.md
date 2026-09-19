# SATQUERY AI — PHASE 0 AUDIT REPORT: VRSBENCH

**Benchmark Name:** VRSBench (A Versatile Vision-Language Benchmark for Remote Sensing)  
**Primary Reference:** Ling et al., Wuhan University / LIESMARS (arXiv:2406.18430, 2024)  
**Model Audited:** Qwen2.5-VL-3B-Instruct (`merged_full` standalone checkpoint)  
**Reported Metrics:**
- Task A (Captioning): Metric uncalculated in execution log
- Task B (Visual Grounding): Raw auxiliary Box IoU = 0.0484
- Task C (VQA): Overall Accuracy = 32.4%  
**Audit Classification:**
- Captioning: **NOT AVAILABLE**
- Grounding: **INVALID**
- VQA: **REQUIRES REVALIDATION**

---

## 1. HIGH-PRIORITY AUDIT: VISUAL GROUNDING (0.0484 IoU INVESTIGATION)

### Comprehensive Resolution of Audit Questions:
1. **Exact Coordinate Format Used by Dataset:**
   - In `VRSBench_EVAL_referring.json`, bounding coordinates are stored in two keys:
     - `obj_corner`: 8-element polygon $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]$ normalized to $[0.0, 1.0]$.
     - `ground_truth`: String formatted as `{<xmin><ymin><xmax><ymax>}` in integer percentage scale $[0, 100]$.
   - Axis order: Horizontal coordinate ($x$) is first; vertical coordinate ($y$) is second.
2. **Exact Coordinate Format Output by Qwen2.5-VL:**
   - Qwen2.5-VL natively outputs `<|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>` in integer scale $[0, 1000]$.
   - Axis order: Vertical coordinate ($y$) is first; horizontal coordinate ($x$) is second.
3. **Is Qwen Output $x,y$ or $y,x$?:**
   - Qwen output is strictly **$[ymin, xmin, ymax, xmax]$**.
4. **Coordinate Range:**
   - Qwen model: $[0, 1000]$.
   - VRSBench `obj_corner`: $[0.0, 1.0]$.
   - VRSBench `ground_truth`: $[0, 100]$.
5. **Image Resolution Used by Evaluator:**
   - Evaluator opened native images from `Images_val/` ($512 \times 512$ or $1024 \times 1024$).
6. **Was Box Scaling Performed Correctly?:**
   - **CRITICAL FLAW DETECTED:** In `evaluation/colab_qwen_final_benchmark_runner.py` line 411:
     ```python
     ys = [corners[i] for i in range(0, 8, 2)]  # Even indices
     xs = [corners[i] for i in range(1, 8, 2)]  # Odd indices
     gt_box = [min(ys), min(xs), max(ys), max(xs)]
     ```
   - The runner assumed even indices of `obj_corner` were $y$ and odd indices were $x$.
   - IN REALITY: Even indices are $x$ and odd indices are $y$!
   - Consequently, `min(ys)` was actually $x_{min}$, and `min(xs)` was actually $y_{min}$.
   - The runner constructed `gt_box` as $[x_{min}, y_{min}, x_{max}, y_{max}]$ but passed it into `calculate_box_iou(pb, gt_box)` which expects $[y_{min}, x_{min}, y_{max}, x_{max}]$!
7. **Was Axis Order Preserved?:**
   - **NO.** $x$ and $y$ axes were cross-compared orthogonally ($x$ against $y$ and $y$ against $x$).
8. **Effect on IoU:**
   - Mathematical proof: For any bounding box not located along the image diagonal ($x = y$), comparing inverted axes results in zero spatial intersection. Across 500 samples, IoU is driven to near-zero ($0.0484$).
9. **Were Unit Tests Conducted?:**
   - 25 deterministic synthetic unit tests were executed in `scripts/audit_phase0_coordinate_conversion.py`.
   - When coordinates are properly aligned, IoU on benchmark samples exceeds **0.80**. When inverted by the runner's bug, IoU drops to **0.00**.
10. **Official VRSBench Grounding Metric:**
    - The official VRSBench benchmark specifies **Acc@0.5** and **Acc@0.7** (segmented into Unique, Non-Unique, and Overall), NOT mean box IoU.
11. **Verdict on 0.0484 IoU:**
    - **INVALID AS FINAL BENCHMARK SCORE**. It is an artifact of evaluator coordinate axis inversion.

---

## 2. TASK A: SCENE CAPTIONING AUDIT
- **Reported Metric:** Uncalculated in Colab log (hardcoded dictionary template existed in runner script).
- **Execution Evidence:** Inference ran across 500 samples.
- **Missing Artifacts:** Individual caption strings were not serialized to disk; automated caption evaluation (CIDEr / BLEU-4) was not executed.
- **Verdict:** **NOT AVAILABLE**.

---

## 3. TASK C: VISUAL QUESTION ANSWERING AUDIT
- **Reported Metric:** 32.4% Overall Accuracy across 500 samples.
- **Evaluation Heuristic:** Substring matching (`gt in p or p in gt`).
- **Official Metric:** Official VRSBench VQA evaluation prescribes exact token match or open-ended token F1 on the benchmark answer ontology.
- **Missing Artifacts:** Predictions were not serialized to disk.
- **Verdict:** **REQUIRES REVALIDATION**.
