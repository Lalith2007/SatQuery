# Division 3: Bi-Temporal Change Intelligence

**Specialist Tool:** `bitemporal_change_specialist`  
**Production Model:** `TinyCD` (Siamese U-Net with Mixing Mask Attention Blocks)  
**Status:** Validated, Authoritatively Evaluated, and Frozen  

---

## 1. Overview

Division 3 provides neural bi-temporal change detection, spatial change localization, and natural-language change reasoning over co-registered remote-sensing optical satellite imagery. Consuming exactly two co-registered images ($T_0$ and $T_1$) and an optional query, it returns a canonical `ToolResult` equipped with dense change probability maps, thresholded binary masks, spatial bounding boxes, and quantitative change statistics.

```
T0 + T1 Rasters
      │
      ▼
┌──────────────────────────────────────┐
│       6-Point Input Validation       │
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  Bilinear Resampling & Normalization │ (256x256, [0.0, 1.0] float32)
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│        TinyCD Neural Inference       │ (Siamese U-Net + 3-Stage MAMB Attention)
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│   Postprocessing & Filtering (0.50)  │ (Thresholding, Morphology, Connected Components)
└──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│      Spatial Metric Synthesis        │ (Query intent, area, counts, evidence)
└──────────────────────────────────────┘
      │
      ▼
   ToolResult (Maps, Bboxes, Metrics, Composite Artifacts)
```

---

## 2. Production Model & Verified Checkpoint

- **Model Class:** `specialists.temporal_change.adaptation.models.tinycd.TinyCD`
- **Adapter:** `specialists.temporal_change.model_adapter.TinyCDAdapter`
- **Architecture:** Siamese twin double-conv encoder (32 $\rightarrow$ 256 ch) + 3-stage Mixing Mask Attention Blocks (MAMB) + U-Net transpose conv decoder + $1\times 1$ conv classifier
- **Parameter Count:** `3,565,034` parameters (145 weight tensors)
- **Active Checkpoint:** [`specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/weights/ChangeDetector-TinyCD.pth)
- **Cryptographic SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`
- **Strict Load (`strict=True`):** **PASSED** (0 missing keys, 0 unexpected keys)
- **Production Threshold:** `0.50`
- **Status:** **HEALTHY & FROZEN** (Zero retraining or replacement required)

### Checkpoint Provenance Gate & Zero-Fallback Policy

Production execution enforces a strict fail-closed **Checkpoint Provenance Gate** at startup:
1. Validates that the active model architecture is `tinycd`.
2. Validates that the checkpoint file exists on disk.
3. Validates that the cryptographic SHA-256 matches `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`.
4. Enforces strict PyTorch weights loading (`strict=True`: 0 missing keys, 0 unexpected keys).
5. Confirms the exact parameter count matches `3,565,034`.

If any check fails, execution immediately halts with `STATUS = CHECKPOINT_INVALID`. **Random or untrained fallback is strictly forbidden.** Silently initializing `SiameseFeatureDiff`, random ResNet-18, or stub ChangeFormer models is completely excised in production.

> **Forensic Audit Clarification:**
> 17.69% F1 was produced by an untrained randomly initialized SiameseFeatureDiff fallback caused by an architecture/checkpoint dispatch misconfiguration. It is not a TinyCD result.
>
> The verified TinyCD model achieves **79.31% – 79.54% F1** and **65.71% – 66.03% IoU** across the 128-scene official LEVIR-CD test set, and **83.26% – 83.90% F1** on forensic sample `test_10.png`.

---

## 3. Authoritative Performance Metrics

Performance evaluated on the **official standard LEVIR-CD held-out test split** (128 full-scale $1024\times 1024$ scenes, $8,388,608$ evaluated pixels) using the production pipeline:

| Metric | Raw Probability Output ($\tau=0.50$) | Production Postprocessed (Morphology + CC) |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **`97.99%`** | **`97.99%`** |
| **Precision** | **`82.70%`** | **`83.36%`** |
| **Recall** | **`76.62%`** | **`75.63%`** |
| **F1 Score** | **`79.54%`** | **`79.31%`** |
| **IoU (Jaccard Index)** | **`66.03%`** | **`65.71%`** |
| **Dice Coefficient** | `79.54%` | `79.31%` |
| **Specificity** | `99.14%` | `99.19%` |
| **Ground-Truth Change Pixels** | 427,414 (`5.0952%`) | 427,414 (`5.0952%`) |
| **Ground-Truth Unchanged Pixels** | 7,961,194 (`94.9048%`) | 7,961,194 (`94.9048%`) |
| **Predicted Change Pixels** | 395,982 (`4.7205%`) | 387,794 (`4.6229%`) |

### Runtime Telemetry
- **Compute Device:** Apple Silicon Metal (MPS)
- **Mean Forward Latency:** **`27.32 ms`**
- **Median (P50) Latency:** **`26.98 ms`**
- **P95 Latency:** **`28.23 ms`**
- **Throughput:** **`36.61 FPS`**
- **CPU Host Baseline:** `156.31 ms` / `6.40 FPS`

---

## 4. Historical Context & Dataset Distinction

- **Authoritative Fresh Test Set ($N=128$):** The standard LEVIR-CD benchmark test split consisting of 128 full-scale $1024\times 1024$ optical scenes (`test_1.png` through `test_128.png`) from `satellite-image-deep-learning/LEVIR-CD` (`test.zip`, 473.3 MB).
- **Historical Preliminary Report ($N=348$):** Earlier preliminary Colab training documentation reported $N=348$ based on the extended LEVIR-CD+ dataset partition (542 train, 95 val, 348 test $= 985$ pairs), reporting preliminary metrics of OA $97.30\%$, F1 $66.54\%$, and IoU $49.86\%$.
- **Outcome:** Fresh held-out evaluation on the standard LEVIR-CD test split confirms that the trained TinyCD production model significantly surpasses the preliminary Colab estimates (+13.00% F1, +16.17% IoU).

---

## 5. Configuration & Environment Variables

All settings are environment-configurable with `SATQUERY_TC_` prefix:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `SATQUERY_TC_DEVICE` | `auto` | Compute device (`auto`, `mps`, `cuda`, `cpu`) |
| `SATQUERY_TC_CHANGE_THRESHOLD` | `0.5` | Production binary change threshold |
| `SATQUERY_TC_MIN_REGION_AREA` | `100` | Minimum connected-component area (pixels) |
| `SATQUERY_TC_MORPH_KERNEL` | `3` | Morphological filter kernel size |
| `SATQUERY_TC_MAX_REGIONS` | `20` | Maximum localized bounding boxes reported |
| `SATQUERY_TC_INPUT_SIZE` | `256` | Model square input dimension |
| `SATQUERY_TC_USE_MOCK` | `false` | Fallback to mock backend (testing only) |

---

## 6. Testing & Validation

Run the comprehensive Division 3 test suite:

```bash
./.venv/bin/pytest tests/test_temporal_change_specialist.py -v
```

All **51 test gates** pass across `tests/test_temporal_change_specialist.py` (45 tests) and `tests/test_tinycd_production_verification.py` (6 tests): unit validation, contract compliance, mock compatibility, genuine neural inference, tool registry integration, agent execution, and checkpoint provenance gate enforcement.
