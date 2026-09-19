# Optical-SAR Specialist — Official WHU-OPT-SAR Benchmark Evaluation

**Specialist:** `optical_sar_cross_modal_specialist` (CMAF)  
**Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth` (SHA-256: `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`)  
**Parameter Count:** `19,755,144` (Optical: `8,543,296`, SAR: `8,540,160`, Fusion: `2,297,344`, Head: `374,344`)  
**Evaluation Date:** 2026-09-17 15:05:35 UTC  
**Status:** **OFFICIAL EVALUATION COMPLETE — VERIFIED**  

---

## 1. Executive Summary

The frozen production **CMAF** (Cross-Modal Attention Fusion) model was evaluated across all **4,950 held-out test tiles** (15 parent scenes, 308,687,656 valid pixels evaluated, 255 border pixels ignored) on device `mps`.

### Overall Performance Summary

| Metric | Measured Value | Reference Baseline | Delta |
|:---|:---:|:---:|:---:|
| **Overall Accuracy (OA)** | **71.71%** | `71.71%` | -0.00% |
| **Mean IoU (mIoU)** | **35.08%** | `35.08%` | -0.00% |
| **Macro F1** | **46.62%** | `46.62%` | +0.00% |
| **Macro Precision** | **46.58%** | `46.58%` | +0.00% |
| **Macro Recall** | **52.48%** | `52.48%` | -0.00% |
| **Weighted F1** | **74.18%** | `74.18%` | -0.00% |
| **Weighted IoU** | **60.38%** | `60.38%` | +0.00% |
| **Weighted Precision** | **77.69%** | `77.69%` | +0.00% |
| **Weighted Recall** | **71.71%** | `71.71%` | -0.00% |

---

## 2. Per-Class Performance Breakdown (8 Official Classes)

| Class ID | Class Name | IoU | F1 Score | Precision | Recall | Support (Pixels) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | **Background** | 0.00% | 0.00% | 0.00% | 0.00% | 1,973 |
| 1 | **Farmland** | 59.20% | 74.37% | 75.96% | 72.85% | 111,315,326 |
| 2 | **City** | 44.57% | 61.66% | 68.43% | 56.11% | 13,218,868 |
| 3 | **Village** | 33.32% | 49.99% | 44.15% | 57.60% | 16,869,004 |
| 4 | **Water** | 50.71% | 67.29% | 69.24% | 65.45% | 40,822,113 |
| 5 | **Forest** | 73.69% | 84.85% | 92.27% | 78.54% | 118,860,724 |
| 6 | **Road** | 11.68% | 20.91% | 12.36% | 67.73% | 3,083,943 |
| 7 | **Others** | 7.46% | 13.89% | 10.25% | 21.52% | 4,515,705 |

---

## 3. Dominant Confusion Pairs

| Ground Truth Class | Misclassified As | Confused Pixels | % of GT Class |
|:---|:---|:---:|:---:|
| **Forest** | Farmland | 14,411,773 | 12.12% |
| **Farmland** | Water | 8,151,523 | 7.32% |
| **Farmland** | Village | 6,447,174 | 5.79% |
| **Water** | Farmland | 5,954,585 | 14.59% |
| **Farmland** | Forest | 5,560,010 | 4.99% |
| **Farmland** | Road | 5,257,941 | 4.72% |

---

## 4. Runtime Benchmark

- **Total Tiles Evaluated:** `4,950` tiles across `15` parent scenes
- **Valid Pixels Evaluated:** `308,687,656` (95.16% valid, `15,715,544` ignored boundary pixels)
- **Total Test Execution Time:** `284.87 s`
- **Mean Batch Latency (B=32):** `126.91 ms`
- **Inference Throughput:** `17.38 tiles/sec`

Visual panels archived in `reports/benchmark_eval/optical_sar/visual_examples/`.
