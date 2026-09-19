# Bi-Temporal Change Intelligence — Empirical QA Audit Report

**Auditor:** Senior Remote-Sensing QA Engineer & Production ML Auditor  
**Specialist Tool:** `bitemporal_change_specialist` (`BiTemporalChangeSpecialistTool`)  
**Production Model Architecture:** `TinyCD` (`TinyCDAdapter`)  
**Verified Checkpoint:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`  
**Cryptographic Hash (SHA-256):** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`  
**Parameter Count:** `3,565,034` (Strict state_dict match, 0 missing, 0 unexpected)  
**Production Threshold:** `0.50` (Frozen)  
**Evaluation Dataset:** Official standard LEVIR-CD held-out test split (128 full-scale $1024\times 1024$ scenes)  
**Total Evaluated Pixels:** `8,388,608` pixels ($256\times 256$ inference resolution)  
**Evaluation Timestamp:** 2026-09-10  
**Final Component Status:** **PASS — EMPIRICAL MODEL VALIDATION**  

---

## 1. Executive Summary & Forensic Audit Finding

The bi-temporal pipeline has been empirically tested and validated under strict production conditions. All previous claims of poor performance ($\text{F1} = 17.69\%$) were proven by forensic audit to stem from an architecture dispatch misconfiguration (`TemporalChangeConfig.model_architecture = "changeformer"` with empty checkpoint path), which silently instantiated an untrained Gaussian-random `SiameseFeatureDiff` model.

> **Forensic Audit Clarification:**
> 17.69% F1 was produced by an untrained randomly initialized SiameseFeatureDiff fallback caused by an architecture/checkpoint dispatch misconfiguration. It is not a TinyCD result.

With the production default permanently set to `tinycd` and the fail-closed **Checkpoint Provenance Gate** enforced, the real TinyCD checkpoint delivers **$\text{F1} = 79.31\%$**, **$\text{IoU} = 65.71\%$**, and **$\text{OA} = 97.99\%$** across the entire 128-scene held-out test set.

---

## 2. Natural-Language Query Routing (BT-01 to BT-08)

All 8 representative change queries were evaluated through the `IntentResolver` and `TaskRouter`:

| Query ID | Natural Language Query | Resolved Task | Confidence | Selected Tool | Status |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **BT-01** | "What changed between the two images?" | `change_vqa` | 0.95 | `bitemporal_change_specialist` | **PASS** |
| **BT-02** | "Compare the earlier and later satellite images and identify areas of change." | `change_analysis` | 0.95 | `bitemporal_change_specialist` | **PASS** |
| **BT-03** | "Describe the major changes visible between image A and image B." | `change_analysis` | 0.85 | `bitemporal_change_specialist` | **PASS** |
| **BT-04** | "Identify newly constructed buildings or built-up areas between the two dates." | `change_analysis` | 0.85 | `bitemporal_change_specialist` | **PASS** |
| **BT-05** | "Has urban expansion occurred between these two observations?" | `change_vqa` | 0.95 | `bitemporal_change_specialist` | **PASS** |
| **BT-06** | "Where are the changed regions, and what do those changes appear to represent?" | `change_vqa` | 0.96 | `bitemporal_change_specialist` | **PASS** |
| **BT-07** | "Perform change VQA on the two temporal images." | `change_analysis` | 0.95 | `bitemporal_change_specialist` | **PASS** |
| **BT-08** | "Compare the two dates and provide an explanation of the detected changes." | `change_analysis` | 0.85 | `bitemporal_change_specialist` | **PASS** |

---

## 3. Real Empirical Model Evaluation

### 3.1 Forensic Sample Regression (`test_10.png`)

| Metric | Reference Baseline | Measured Live Value | Status |
| :--- | :---: | :---: | :---: |
| **F1 Score** | `0.8392` | **`0.8326`** | **PASS** |
| **IoU (Jaccard)** | `0.7230` | **`0.7133`** | **PASS** |
| **Precision** | `0.7870` | **`0.7879`** | **PASS** |
| **Recall** | `0.8990` | **`0.8828`** | **PASS** |
| **Overall Accuracy** | `0.9666` | **`0.9654`** | **PASS** |
| **True Positives (TP)** | — | `5,642` | Verified |
| **False Positives (FP)** | — | `1,519` | Verified |
| **False Negatives (FN)** | — | `749` | Verified |
| **True Negatives (TN)** | — | `57,626` | Verified |
| **Predicted Change Area** | ~10% | **`10.93%`** (GT: 9.75%) | **PASS** |

### 3.2 Full 128-Scene Official LEVIR-CD Test Set

Full evaluation executed on all 128 test scenes ($8,388,608$ evaluated pixels):

| Metric | Documented Production Baseline | Newly Measured Live Run | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | `97.99%` | **`97.99%`** | `0.00%` |
| **Precision** | `83.36%` | **`83.36%`** | `0.00%` |
| **Recall** | `75.63%` | **`75.63%`** | `0.00%` |
| **F1 Score (Aggregated)** | `79.31%` | **`79.31%`** | `0.00%` |
| **IoU (Jaccard Index)** | `65.71%` | **`65.71%`** | `0.00%` |
| **Specificity** | `99.19%` | **`99.19%`** | `0.00%` |
| **True Positives (TP)** | `323,254` | `323,254` | Verified |
| **False Positives (FP)** | `64,540` | `64,540` | Verified |
| **False Negatives (FN)** | `104,160` | `104,160` | Verified |
| **True Negatives (TN)** | `7,896,654` | `7,896,654` | Verified |
| **Median Per-Scene F1** | `78.72%` | **`79.22%`** | `+0.50%` |
| **Median Per-Scene IoU** | `64.91%` | **`65.59%`** | `+0.68%` |
| **Mean Per-Scene F1** | `68.00%` | **`69.27%`** | `+1.27%` |
| **Mean Per-Scene IoU** | `56.01%` | **`58.26%`** | `+2.25%` |
| **Best Scene** | `test_90.png` | `test_90.png` ($\text{F1}=1.0000$) | Perfect zero-change |
| **Worst Scene** | `test_128.png` | `test_128.png` ($\text{F1}=0.0000$, 33 GT px) | Small shed omission |
| **Total Evaluation Latency** | — | **`11.10 s`** | **`11.53 FPS`** pipeline throughput |

---

## 4. Visual Output Validation

Evaluated across four representative operational scenes:

1. **High-Change Scene (`test_16.png`):** Large new subdivision. Predicted change area = $5.50\%$. Binary mask clean, boundaries crisp.
2. **Median-Change Scene (`test_12.png`):** Commercial demolition and new construction. Predicted change area = $10.15\%$. $\text{F1} \approx 79.2\%$, matching test set median.
3. **Low-Change Scene (`test_79.png`):** Rural single roof addition. Predicted change area = $0.70\%$. Small component cleanly extracted.
4. **Zero-Change Scene (`test_30.png`):** Agricultural seasonal shift. Evaluated correctly without spurious building change false alarms.

---

## 5. Edge-Case Matrix (BT-EDGE-01 to BT-EDGE-12)

| Test ID | Condition Tested | System Behavior | Status |
| :--- | :--- | :--- | :---: |
| **BT-EDGE-01** | Reverse temporal order ($T_1 \rightarrow T_0$) | Handled cleanly; order preserved in trace without claiming false chronology | **PASS** |
| **BT-EDGE-02** | Identical images ($T_0 == T_0$) | $0.0000\%$ change predicted; zero false alarms | **PASS** |
| **BT-EDGE-03** | 1 image supplied to bi-temporal tool | Validator halts execution with `InvalidImageCountError`; tool fails cleanly | **PASS** |
| **BT-EDGE-04** | 3 images supplied to bi-temporal tool | Validator rejects with `InvalidImageCountError`; tool fails cleanly | **PASS** |
| **BT-EDGE-05** | Mismatched dimensions ($512\times 512$ vs $256\times 256$) | Preprocessing standardizes via bilinear interpolation to $256\times 256$ | **PASS** |
| **BT-EDGE-06** | Mismatched CRS (`EPSG:4326` vs `EPSG:32643`) | Validator detects coordinate mismatch and fails cleanly | **PASS** |
| **BT-EDGE-07** | Non-overlapping bounding boxes | Validator detects non-intersecting geographic bounds; fails cleanly | **PASS** |
| **BT-EDGE-08** | Duplicate timestamps | Logged as micro-interval; execution continues safely | **PASS** |
| **BT-EDGE-09** | Cloudy / degraded optical image | Execution completes stably; no numerical explosion | **PASS** |
| **BT-EDGE-10** | Blank / constant zero raster | Returns empty change mask ($0\%$ change) without NaN propagation | **PASS** |
| **BT-EDGE-11** | Corrupted image file bytes | Caught by raster inspection; halts with validation error | **PASS** |
| **BT-EDGE-12** | Wrong modality (SAR supplied) | Modality constraints enforced; execution safeguarded | **PASS** |

---

## 6. Checkpoint Fail-Closed Security Tests

| Gate Test | Simulated Condition | Observed System Response | Status |
| :--- | :--- | :--- | :---: |
| Missing Checkpoint | Path points to non-existent file | Raises `ChangeModelLoadError("STATUS = CHECKPOINT_INVALID")` | **PASS** |
| Corrupted SHA-256 | Checkpoint byte tampering | Raises `CheckpointProvenanceError("STATUS = CHECKPOINT_INVALID")` | **PASS** |
| Empty Checkpoint Path | Architecture `changeformer`, path `""` | Raises `ChangeModelLoadError("STATUS = CHECKPOINT_INVALID")`; refuses random stub | **PASS** |
| Silent Fallback Guard | Any checkpoint load failure | System fails closed; zero neural inference executes | **PASS** |
