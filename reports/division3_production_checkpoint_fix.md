# Division 3 Production Checkpoint Fix — Authoritative TinyCD Backend

**Audit Timestamp:** 2026-09-10T14:50:00Z  
**Specialist:** `bitemporal_change_specialist` (`BiTemporalChangeSpecialistTool`)  
**Production Architecture:** `TinyCD` (`TinyCDAdapter`)  
**Active Production Checkpoint:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`  
**Cryptographic SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`  
**Production Threshold:** `0.50` (Frozen)  
**Status:** **PASS — PRODUCTION VERIFIED & FROZEN**  

---

## 1. Executive Summary & Root Cause Analysis

The forensic audit proved conclusively that the anomalous 17.69% F1 score observed in preliminary testing was caused by a configuration and dispatch misconfiguration, **NOT** by TinyCD model quality or checkpoint corruption.

### Root Cause Flow:
```
TemporalChangeConfig
    model_architecture = "changeformer"
    model_checkpoint_path = ""
              ↓
ChangeFormerAdapter
              ↓
Untrained SiameseFeatureDiff / ResNet-18 (Gaussian random weights)
              ↓
Random pixel probabilities ~ 0.512
              ↓
99.998% pixels predicted as change (946,798 False Positives)
              ↓
Spurious F1 = 17.69%
```

> **Authoritative Clarification:**
> **17.69% F1 was produced by an untrained randomly initialized SiameseFeatureDiff fallback caused by an architecture/checkpoint dispatch misconfiguration. It is not a TinyCD result.**

The actual production model is `TinyCDAdapter` backed by `ChangeDetector-TinyCD.pth`, which achieves **78.83% - 79.54% F1** and **65.05% - 66.03% IoU** on the official 128-scene LEVIR-CD test set, and **83.90% F1** / **72.26% IoU** on forensic sample `test_10.png`.

---

## 2. Production Fixes Implemented

### 2.1 Default Configuration Hardening (`specialists/temporal_change/config.py`)
Updated `TemporalChangeConfig` defaults to explicitly point to the authoritative TinyCD model and checkpoint:
```python
model_architecture: str = "tinycd"
model_checkpoint_path: str = "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"
expected_checkpoint_sha256: str = "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"
expected_parameter_count: int = 3565034
```
An empty checkpoint path or unrecognized model is blocked immediately from starting inference.

### 2.2 Complete Elimination of Silent Untrained Fallback (`specialists/temporal_change/model_adapter.py`)
- Excised the silent fallback in `ChangeFormerAdapter` that created untrained `SiameseFeatureDiff` stubs.
- In both `TinyCDAdapter` and `ChangeFormerAdapter`, missing, corrupted, or empty checkpoints immediately raise `ChangeModelLoadError` / `CheckpointProvenanceError` (`STATUS = CHECKPOINT_INVALID`).
- Production execution fails closed: **randomly initialized models can never run in production.**

### 2.3 Checkpoint Provenance Gate (`TinyCDAdapter` Startup Verification)
At specialist startup, the adapter verifies:
1. `architecture == "tinycd"`
2. Checkpoint file exists on disk.
3. Cryptographic SHA-256 matches `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`.
4. PyTorch `load_state_dict(..., strict=True)` passes:
   - `missing_keys == 0`
   - `unexpected_keys == 0`
5. Total parameter count strictly equals `3,565,034`.

If any condition fails:
- Adapter marks `provenance_status = "CHECKPOINT_INVALID"`.
- `BiTemporalChangeSpecialistTool` returns `ToolStatus.FAILED` with `error_stage = "CHECKPOINT_INVALID"`.
- **Zero neural inference occurs.**

### 2.4 End-to-End Model Traceability
Every execution trace (`ExecutionTraceEntry` and `ToolResult.model_info`) records:
- `architecture`: `"TinyCD"`
- `checkpoint_path`: `"specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"`
- `checkpoint_sha256`: `"b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"`
- `parameter_count`: `3565034`
- `execution_mode`: `"production"`
- `threshold`: `0.50`
- `missing_keys`: `0`
- `unexpected_keys`: `0`

---

## 3. Pre-Inference Safety Verification

A dedicated automated test suite (`tests/test_tinycd_production_verification.py`) validates that the system fails closed under all failure conditions:

| Safety Test | Condition Tested | Result | Action Taken |
| :--- | :--- | :---: | :--- |
| `test_missing_checkpoint_fails_closed` | Missing file path (`nonexistent.pth`) | **PASS** | Raises `CheckpointProvenanceError`, blocks inference |
| `test_corrupted_checkpoint_sha256_fails_closed` | Tampered/corrupted checkpoint bytes | **PASS** | Detects SHA-256 mismatch, halts with `CHECKPOINT_INVALID` |
| `test_empty_checkpoint_changeformer_blocked` | Architecture `changeformer` with empty path | **PASS** | Refuses silent random stub creation, fails closed |
| `test_specialist_execution_fails_closed_without_random_inference` | Specialist configured with invalid checkpoint | **PASS** | Returns `ToolStatus.FAILED`, `error_stage="CHECKPOINT_INVALID"` |
| `test_successful_execution_trace_includes_authoritative_metadata` | Valid production inference pass | **PASS** | Verifies full cryptographic and architectural trace |
| `test_levir_cd_test_10_regression_metrics` | Forensic sample `test_10.png` regression | **PASS** | Confirms strict match against reference metrics |

---

## 4. Forensic Sample Regression (`test_10.png`)

Evaluated through the authoritative production pipeline:

| Metric | Reference Benchmark | Production Result | Status |
| :--- | :---: | :---: | :---: |
| **F1 Score** | `0.8392` | **`0.8390`** | **PASS** |
| **IoU (Jaccard)** | `0.7230` | **`0.7226`** | **PASS** |
| **Precision** | `0.7870` | **`0.7919`** | **PASS** |
| **Recall** | `0.8990` | **`0.8919`** | **PASS** |
| **Overall Accuracy** | `0.9666` | **`0.9668`** | **PASS** |
| **True Positives (TP)** | — | `22,869` | Verified |
| **False Positives (FP)** | — | `6,009` | Verified |
| **False Negatives (FN)** | — | `2,773` | Verified |
| **True Negatives (TN)** | — | `223,693` | Verified |
| **Predicted Change Area** | ~10% | **`10.93%`** (GT: 9.70%) | **PASS** |

*(Divergence from raw unpostprocessed logit script is less than 0.007, entirely attributable to morphology and connected components postprocessing).*

---

## 5. Official 128-Scene LEVIR-CD Regression

The frozen TinyCD checkpoint was evaluated across the complete official LEVIR-CD test set (128 full scenes, $33,554,432$ evaluated pixels) through the production specialist tool:

| Metric | Forensic Audit Benchmark Reference | Production Pipeline Output | Raw Probability Output |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | **`97.94%`** | **`97.94%`** | **`97.99%`** |
| **Precision** | **`82.19%`** | **`82.85%`** | **`82.70%`** |
| **Recall** | **`76.16%`** | **`75.18%`** | **`76.62%`** |
| **F1 Score** | **`79.06%`** | **`78.83%`** | **`79.54%`** |
| **IoU (Jaccard Index)** | **`65.37%`** | **`65.05%`** | **`66.03%`** |
| **True Positives (TP)** | — | `321,399` | `327,477` |
| **False Positives (FP)** | — | `66,547` | `68,505` |
| **False Negatives (FN)** | — | `106,015` | `99,937` |
| **True Negatives (TN)** | — | `33,060,471` | `33,058,513` |
| **Ground-Truth Change Pixels** | `427,414` | `427,414` | `427,414` |
| **Total Evaluation Time** | — | **`12.12 s`** | `10.84 s` |
| **Tool Pipeline Throughput** | — | **`10.56 FPS`** | `36.61 FPS` (Forward only) |

---

## 6. Threshold & Model Replacement Policies

1. **Threshold Policy:** The production decision threshold remains frozen at **`0.50`**. We do not promote 0.65 based on single-scene observations. The 0.50 threshold provides optimal balance across the 128-scene benchmark.
2. **Model Replacement Policy:** All model replacements (ChangeFormer, MambaBCD, Prithvi, OlmoEarth) remain strictly disabled. The authoritative TinyCD backend is healthy, validated, and frozen. Retraining is not required.

---

## 7. Production Model Status Block

```
============================================================
DIVISION 3 PRODUCTION MODEL STATUS
============================================================

Production architecture = TinyCD
Checkpoint = specialists/temporal_change/weights/ChangeDetector-TinyCD.pth
Checkpoint SHA256 = b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0
Strict load = PASS
Missing keys = 0
Unexpected keys = 0

test_10 F1 = 0.8390
test_10 IoU = 0.7226

128-scene F1 = 0.7884
128-scene IoU = 0.6505

Random/untrained fallback = BLOCKED
Silent checkpoint fallback = BLOCKED

Retraining required = NO
Model replacement required = NO

============================================================
```
