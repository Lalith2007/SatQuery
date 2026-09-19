# Bi-Temporal Change Specialist — Official LEVIR-CD Benchmark Evaluation

**Specialist:** `bitemporal_change_specialist` (TinyCD)  
**Checkpoint:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth` (SHA-256: `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`)  
**Parameter Count:** `3,565,034`  
**Evaluation Date:** 2026-09-17 15:03:04 UTC  
**Status:** **OFFICIAL EVALUATION COMPLETE — VERIFIED**  

---

## 1. Executive Summary

The frozen production **TinyCD** change detection model was evaluated across all **128 held-out parent scenes** of the official **LEVIR-CD** test split (8,388,608 evaluated pixels at 256x256 resolution) on device `mps`.

### Overall Pixel-Pooled Performance (Threshold = 0.5)

| Metric | Raw Probability Output | Production Postprocessed (Morphology + CC) | Reference Authoritative Baseline | Delta (vs Prod) |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy (OA)** | **97.99%** | **97.99%** | `97.99%` | -0.00% |
| **Precision** | **82.70%** | **83.36%** | `83.36%` | -0.00% |
| **Recall** | **76.62%** | **75.63%** | `75.63%` | +0.00% |
| **F1 Score** | **79.54%** | **79.31%** | `79.31%` | -0.00% |
| **IoU (Jaccard Index)** | **66.03%** | **65.71%** | `65.71%` | -0.00% |
| **Specificity** | **99.14%** | **99.19%** | `99.19%` | -0.00% |

---

## 2. Per-Scene Averaged Metrics

To distinguish pixel-pooled totals from per-scene distributions:

- **Mean Scene F1:** `69.27%`
- **Median Scene F1:** `79.22%`
- **Mean Scene IoU:** `58.26%`
- **Median Scene IoU:** `65.59%`

---

## 3. Binary Confusion Matrix (Production Postprocessed)

```
                     Predicted No Change    Predicted Change
GT No Change (TN/FP)        7,896,654              64,540
GT Change    (FN/TP)          104,160             323,254
```

- **True Positives (TP):** `323,254`
- **True Negatives (TN):** `7,896,654`
- **False Positives (FP):** `64,540`
- **False Negatives (FN):** `104,160`

---

## 4. Runtime Benchmark

- **Total Test Set Execution:** `10.14 s`
- **Mean Latency per Scene:** `30.49 ms`
- **Median Latency:** `29.72 ms`
- **Throughput:** `12.62 FPS`

---

## 5. Representative Scene Analysis

- **Best-Scoring Scene:** `test_16` (F1: `95.86%`, IoU: `92.06%`, GT Pixels: `3,723`, Pred Pixels: `3,602`)
- **Median Scene:** `test_11` (F1: `79.97%`, IoU: `66.63%`, GT Pixels: `704`, Pred Pixels: `694`)
- **Worst-Case Scene:** `test_82` (F1: `27.46%`, IoU: `15.92%`, GT Pixels: `3,344`, Pred Pixels: `829`)
- **Area Discrepancy Outlier:** `test_21` (GT: `9,849` vs Pred: `6,830`)

Visual panels are archived in `reports/benchmark_eval/bitemporal/visual_examples/`.
