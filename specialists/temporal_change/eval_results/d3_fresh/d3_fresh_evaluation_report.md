# Division 3 TinyCD Fresh Held-Out Evaluation Report

## Checkpoint Identity
- **Path:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`
- **SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`
- **File Size:** `14,326,623` bytes (13.66 MB)
- **Parameters:** `3,565,034` (145 state_dict weight tensors, strict load verified)
- **Architecture:** `TinyCD` (Siamese twin double-conv encoders + 3-stage MAMB space-time cross-attention + U-Net transpose conv decoder + single 1x1 conv classifier)

## Dataset Provenance & Protocol
- **Evaluation Protocol:** Fresh held-out evaluation on the standard LEVIR-CD test split.
- **Dataset:** Official standard LEVIR-CD Held-Out Test Set (`satellite-image-deep-learning/LEVIR-CD`)
- **Archive:** `test.zip` (`496,305,323` bytes / 473.3 MB verified)
- **Scenes Evaluated:** `128` full-scale parent scenes
- **Native Resolution:** `1024x1024` pixels (0.5m/pixel VHR optical Google Earth pairs)
- **Input Resolution:** `256x256` pixels (production pipeline bilinear RGB resampling)
- **Total Evaluated Pixels:** `8,388,608`
- **Ground-Truth Changed Pixels:** `427,414` (`5.0952%`)
- **Ground-Truth Unchanged Pixels:** `7,961,194` (`94.9048%`)
- **Predicted Changed Pixels (Threshold 0.50):** `395,982` (`4.7205%`)

## Split Integrity
- **Train/Val/Test Overlap:** `0` (Zero cross-split parent scene overlap)
- **Parent-Scene Leakage:** `0` (Disjoint scene IDs confirmed)

## Fresh Metrics (Threshold = 0.50)

| Metric | Raw Probability Output | Production Postprocessed (Morphology + CC) |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **`97.99%`** | **`97.99%`** |
| **Precision** | **`82.70%`** | **`83.36%`** |
| **Recall** | **`76.62%`** | **`75.63%`** |
| **F1 Score** | **`79.54%`** | **`79.31%`** |
| **IoU (Jaccard Index)** | **`66.03%`** | **`65.71%`** |
| **Dice Coefficient** | `79.54%` | `79.31%` |
| **Specificity** | `99.14%` | `99.19%` |
| **True Positives (TP)** | `327,477` | `323,254` |
| **False Positives (FP)** | `68,505` | `64,540` |
| **False Negatives (FN)** | `99,937` | `104,160` |
| **True Negatives (TN)** | `7,892,689` | `7,896,654` |

## Threshold Sweep Analysis

| Threshold | Precision | Recall | F1 Score | IoU | Overall Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 74.64% | 84.83% | 79.41% | 65.85% | 97.76% |
| 0.35 | 76.87% | 82.96% | 79.80% | 66.39% | 97.86% |
| 0.40 | 78.93% | 81.00% | 79.95% | 66.60% | 97.93% |
| 0.45 | 80.84% | 78.91% | 79.86% | 66.48% | 97.97% |
| **0.50 (Prod)** | **82.70%** | **76.62%** | **79.54%** | **66.03%** | **97.99%** |
| 0.55 | 84.51% | 74.06% | 78.94% | 65.21% | 97.99% |
| 0.60 | 86.32% | 71.21% | 78.04% | 63.99% | 97.96% |
| 0.65 | 88.15% | 68.02% | 76.79% | 62.32% | 97.90% |
| 0.70 | 89.94% | 64.45% | 75.09% | 60.12% | 97.82% |

*Analysis:* F1 scores remain stable across $[0.35, 0.50]$ ($79.4\% \rightarrow 79.9\%$), indicating robust threshold behavior across varying decision cutoffs. Production threshold remains fixed at 0.50.

## Runtime Benchmark
- **Device:** `MPS` (Apple Silicon Metal)
- **Total Test Set Execution Time:** `10.84 s`
- **Mean Inference Latency:** **`27.32 ms`**
- **Median (P50) Latency:** **`26.98 ms`**
- **P95 Latency:** **`28.23 ms`**
- **Throughput:** **`36.61 FPS`**

## Per-Scene Statistics
- **Total Test Scenes:** `128`
- **Scenes with Change (GT > 0):** `118`
- **Zero-Change Scenes (GT == 0):** `10`
- **F1 Distribution:**
  - Mean: `68.00%`
  - Median: `78.72%`
  - Std: `25.51%`
  - P10: `28.30%`
  - P90: `86.61%`
  - Min: `0.00%`
  - Max: `95.84%`
- **IoU Distribution:**
  - Mean: `56.01%`
  - Median: `64.91%`
  - Std: `23.61%`
  - P10: `16.48%`
  - P90: `76.39%`

## Visual Evidence Artifacts
Representative 6-panel diagnostic comparisons saved under `specialists/temporal_change/eval_results/d3_fresh/`:
1. **Strong Prediction (`test_16`):** Precision `97.34%`, Recall `94.39%`, F1 **`95.84%`**, IoU **`92.01%`** (`qualitative_strong_prediction_test_16.png`).
2. **False Positive Bias (`test_33`):** Precision `57.94%`, Recall `83.18%` (`qualitative_false_positive_bias_test_33.png`).
3. **False Negative Omission (`test_79`):** Precision `77.10%`, Recall `47.16%` (`qualitative_false_negative_omission_test_79.png`).
4. **Difficult Median Case (`test_12`):** F1 `78.72%`, IoU `64.91%` (`qualitative_difficult_median_case_test_12.png`).

## Historical Comparison & Protocol Distinction

| Metric | Historical LEVIR-CD+ (Reported N=348) | Fresh Standard LEVIR-CD (Authoritative N=128) | Fresh Postprocessed | Delta (Raw vs Hist) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 97.30% | **`97.99%`** | **`97.99%`** | `+0.69%` |
| **Precision** | 67.26% | **`82.70%`** | **`83.36%`** | `+15.44%` |
| **Recall** | 65.83% | **`76.62%`** | **`75.63%`** | `+10.79%` |
| **F1 Score** | 66.54% | **`79.54%`** | **`79.31%`** | `+13.00%` |
| **IoU (Jaccard)** | 49.86% | **`66.03%`** | **`65.71%`** | `+16.17%` |
| **Latency** | 15.49 ms (Tesla T4) | **`27.32 ms` (MPS)** | — | +11.83 ms |

*Protocol Distinction:* Historical preliminary reports utilized the extended LEVIR-CD+ benchmark partition ($N=348$ test pairs from a 985-pair dataset). The authoritative fresh evaluation uses the standard official LEVIR-CD held-out benchmark test split ($N=128$ parent scenes, 1024x1024 resolution).

## Error Analysis & Documented Limitations
1. **Thin Boundary Artifacts:** High-contrast rooftop eaves and building perimeter contours occasionally exhibit 1-pixel dilation/erosion offsets between T0 and T1 ground truths.
2. **Illumination and Seasonal Variance:** High solar zenith angle shifts create shadowing adjacent to multi-story buildings that model occasionally flags as false positive changes.
3. **Severe Class Imbalance:** Ground-truth changed pixels comprise `427,414` (`5.0952%`), while natural background terrain comprises `7,961,194` (`94.9048%`).
4. **Small Component Dropout:** Postprocessing with `min_region_area=100` filters out isolated pixel noise but suppresses tiny shed/outbuilding emergence under 100 pixels in area.

## Final Model Status
- **Status Classification:** **`GREEN`** (Validated Production Model)
- **Rationale:** 
  - Exact checkpoint verified (`b9a10093...`).
  - Zero mock fallbacks.
  - Consistent high accuracy (`>97.9%` OA) on real optical satellite imagery.
  - Sub-30ms inference on Apple Silicon MPS.
  - Full API integration confirmed.
