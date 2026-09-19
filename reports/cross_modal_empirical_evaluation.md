# Cross-Modal Optical-SAR — Empirical QA Audit Report

**Auditor:** Senior Remote-Sensing QA Engineer & Production ML Auditor  
**Specialist Tool:** `optical_sar_cross_modal_specialist` (`OpticalSARCrossModalSpecialistTool`)  
**Production Model Architecture:** `CMAF` (Cross-Modal Attention Fusion with ResNet-18 Optical & SAR Encoders)  
**Verified Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
**Cryptographic Hash (SHA-256):** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`  
**Total Parameter Count:** `19,755,144` (Strict state_dict match, 0 missing, 0 unexpected)  
**Evaluation Dataset:** Official WHU-OPT-SAR held-out test split (15 full scenes / 4,950 tiles of $256\times 256$)  
**Total Evaluated Pixels:** `308,687,656` valid labeled pixels (`15,715,544` ignored/void pixels)  
**Evaluation Timestamp:** 2026-09-10  
**Final Component Status:** **PASS — EMPIRICAL MODEL VALIDATION**  

---

## 1. Executive Summary

The Cross-Modal Optical-SAR specialist has undergone full empirical validation using real paired remote sensing observations from Sentinel-1 (C-band SAR) and Sentinel-2 / GF-2 (multispectral optical) platforms. The production architecture is Cross-Modal Attention Fusion (CMAF), combining dual-stream ResNet-18 encoders, bidirectional cross-attention modules, and an intent-conditioned land-cover classification head.

Empirical evaluation on the full held-out WHU-OPT-SAR test set (4,950 tiles, 308.7M valid pixels) confirms **Overall Accuracy of 71.71%**, **mean IoU of 35.08%**, **Weighted F1 of 74.18%**, and **Weighted IoU of 60.38%**. Live inference on real held-out tiles achieved an average processing rate of **20.1 FPS** (2.49s for 50 tiles) with 100% deterministic output generation across identical test runs.

---

## 2. Natural-Language Query Routing (CM-01 to CM-06)

All 6 representative multimodal queries were evaluated against the agent's query resolution layer:

| Query ID | Natural Language Query | Resolved Task | Confidence | Selected Tool | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **CM-01** | "Use the optical and SAR images together to identify built-up and water-covered regions." | `optical_sar_analysis` | 0.96 | `optical_sar_cross_modal_specialist` | **PASS** |
| **CM-02** | "Analyze these optical and SAR images together and classify the land-cover types." | `optical_sar_analysis` | 0.96 | `optical_sar_cross_modal_specialist` | **PASS** |
| **CM-03** | "Which regions appear to be urban, agricultural, forested, water, or roads based on both modalities?" | `optical_sar_analysis` | 0.88 | `optical_sar_cross_modal_specialist` | **PASS** |
| **CM-04** | "Use the optical image and SAR image jointly to identify built-up areas." | `optical_sar_analysis` | 0.88 | `optical_sar_cross_modal_specialist` | **PASS** |
| **CM-05** | "Compare the information provided by the optical and SAR observations and identify the dominant land-cover classes." | `optical_sar_analysis` | 0.96 | `optical_sar_cross_modal_specialist` | **PASS** |
| **CM-06** | "Perform multimodal land-cover analysis using both optical and SAR." | `optical_sar_analysis` | 0.96 | `optical_sar_cross_modal_specialist` | **PASS** |

**Routing Summary:** 6/6 queries (100%) successfully routed to the `optical_sar_cross_modal_specialist` with high confidence ($\ge 0.88$).

---

## 3. Real Empirical Model Evaluation

### 3.1 Benchmark Evaluation on Official Held-Out Test Set (4,950 Tiles)

The model was evaluated against ground-truth raster masks across all 15 test scenes (4,950 tiles of $256\times 256$ resolution, $308,687,656$ valid pixels):

| Metric | Reference Production Result | Newly Measured Live Run | Verification Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | `71.71%` | **`71.71%`** (0.717099) | `0.00%` | **PASS** |
| **Mean IoU (mIoU)** | `35.08%` | **`35.08%`** (0.350793) | `0.00%` | **PASS** |
| **Macro F1 Score** | `46.62%` | **`46.62%`** (0.466209) | `0.00%` | **PASS** |
| **Macro Precision** | `46.58%` | **`46.58%`** (0.465837) | `0.00%` | **PASS** |
| **Macro Recall** | `52.48%` | **`52.48%`** (0.524761) | `0.00%` | **PASS** |
| **Weighted F1 Score** | `74.18%` | **`74.18%`** (0.741756) | `0.00%` | **PASS** |
| **Weighted IoU** | `60.38%` | **`60.38%`** (0.603848) | `0.00%` | **PASS** |
| **Weighted Precision** | `77.69%` | **`77.69%`** (0.776939) | `0.00%` | **PASS** |
| **Weighted Recall** | `71.71%` | **`71.71%`** (0.717099) | `0.00%` | **PASS** |

### 3.2 Per-Class Empirical Breakdown

The evaluation incorporates all 8 standard WHU-OPT-SAR semantic land-cover categories:

| Class Index | Class Name | GT Pixel Share | IoU | F1 / Dice | Precision | Recall | TP Pixels | FP Pixels | FN Pixels |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Background / Void | 0.001% | 0.00% | 0.00% | 0.00% | 0.00% | 0 | 2,946,950 | 1,973 |
| **1** | Farmland | 36.06% | 59.20% | 74.37% | 75.96% | 72.85% | 81,090,930 | 25,663,088 | 30,224,396 |
| **2** | City / Built-up | 4.28% | 44.57% | 61.66% | 68.43% | 56.11% | 7,417,501 | 3,421,946 | 5,801,367 |
| **3** | Village | 5.47% | 33.32% | 49.99% | 44.15% | 57.60% | 9,717,374 | 12,291,918 | 7,151,630 |
| **4** | Water Bodies | 13.22% | 50.71% | 67.29% | 69.24% | 65.45% | 26,716,511 | 11,866,998 | 14,105,602 |
| **5** | Forest / Woodland | 38.51% | 73.69% | 84.85% | 92.27% | 78.54% | 93,356,674 | 7,822,373 | 25,504,050 |
| **6** | Road Network | 1.00% | 11.68% | 20.91% | 12.36% | 67.73% | 2,088,756 | 14,806,310 | 995,187 |
| **7** | Others / Bare Land | 1.46% | 7.46% | 13.89% | 10.25% | 21.52% | 971,956 | 8,508,371 | 3,543,749 |

**Observations:**
- Dominant classes (Forest, Farmland, Water, City) accounting for >92% of ground truth show robust F1 scores between **61.66% and 84.85%**.
- Highly linear features (Roads) and minority classes (Others) experience confusion due to resolution limits ($256\times 256$ spatial patches).
- Background class represents rare edge artifacts in WHU tiles.

---

## 4. Cross-Modal Output Validation

Validation was performed on full end-to-end inference pipelines:
- **Output Artifacts:** Generated 5 distinct artifacts per analysis session (semantic segmentation raster, class probability heatmaps, overlay imagery, statistical land-cover summary, and metadata JSON).
- **Evidence Count:** 17 structured evidence nodes generated per run.
- **Determinism:** Bit-exact identical output masks verified across 3 successive inference executions of the same optical/SAR scene pairs.
- **Multi-device Alignment:** Fixed multi-device tensor assignment in `specialists/optical_sar/service.py` to ensure input batches dynamically align to encoder device (`CPU`/`MPS`/`CUDA`).

---

## 5. Edge-Case Matrix (CM-EDGE-01 to CM-EDGE-12)

| Test ID | Condition Tested | System Behavior | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **CM-EDGE-01** | Missing SAR image | Validator halted with `MissingModalityError` | **PASS** | Strictly enforced |
| **CM-EDGE-02** | Missing Optical image | Validator halted with `MissingModalityError` | **PASS** | Strictly enforced |
| **CM-EDGE-03** | Reversed Modality ($SAR \leftrightarrow OPT$) | Pipeline handled inputs deterministically | **PASS** | Mapped by modality tag |
| **CM-EDGE-04** | Mismatched dimensions ($512\times 512$ vs $256\times 256$) | Preprocessor resizes to standard $256\times 256$ | **PASS** | Bilinear interpolation |
| **CM-EDGE-05** | CRS Mismatch | Pipeline completed without unhandled crash | **PASS** | Spatial alignment applied |
| **CM-EDGE-06** | Spatial Extent Mismatch | Pipeline resolved bounding extent | **PASS** | Graceful handling |
| **CM-EDGE-07** | Small Spatial Shift (10-pixel offset) | Pipeline remained numerically stable | **PASS** | Cross-attention robust |
| **CM-EDGE-08** | Cloudy Optical + Valid SAR | Inference completed successfully | **PASS** | SAR penetration utilized |
| **CM-EDGE-09** | Noisy / Degraded SAR (Speckle) | Inference completed stably without NaNs | **PASS** | Feature filtering active |
| **CM-EDGE-10** | Wrong Modality passed as SAR | Specialist executed without raster rejection | **FAIL / ANOMALY** | Documented in `qa_failure_log.md` |
| **CM-EDGE-11** | Corrupted raster file bytes | PIL survived in-memory tensor instantiation | **FAIL / ANOMALY** | Documented in `qa_failure_log.md` |
| **CM-EDGE-12** | Constant / Blank Imagery | Produced valid predictions without overflow | **PASS** | Numerically bounded |

---

## 6. Checkpoint Integrity & Security

- **Path:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`
- **Integrity:** SHA-256 hash verified as `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`.
- **Parameter Count:** Exactly 19,755,144 parameters.
- **Fail-Closed Gate:** Zero random fallback permitted; if checkpoint is absent or tampered, specialist immediately raises a fatal execution error.
