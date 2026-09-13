# SatQuery AI — QA Failure, Anomaly & Edge-Case Investigation Log

**Date:** 2026-09-10 (Updated 2026-09-11)
**Role:** Senior Remote-Sensing QA Engineer, AI Systems Validator, Production ML Auditor
**Scope:** Bi-Temporal Pipeline, Optical-SAR Pipeline, Agentic Controller & Input Validation

---

## 1. Historical Anomaly: Bi-Temporal 17.69% F1 Investigation

### Finding
During initial pipeline validation, LEVIR-CD `test_10.png` produced an anomalous score:
F1 = 0.1769, IoU = 0.0970, with 946,798 false-positive pixels (99.998% of pixels flagged as changed).

### Forensic Root Cause
`TemporalChangeConfig` defaulted to:
```python
model_architecture = "changeformer"
model_checkpoint_path = ""
```
Because the checkpoint path was empty, `ChangeFormerAdapter` silently invoked
`_create_feature_diff_model()`, instantiating an untrained Gaussian-random `SiameseFeatureDiff`
stub (ResNet-18 diff neck). Model output was pure noise near 0.512, tripping the 0.50 threshold everywhere.

### Corrective Action & Verification
1. Production default updated to `model_architecture = "tinycd"` pointing to
   `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`.
2. Excised all silent fallback paths to random weights.
3. Implemented a startup **Checkpoint Provenance Gate** validating SHA-256 (`b9a1009355865c02...`),
   exact parameter count (3,565,034), and strict state dict loading (0 missing, 0 unexpected keys).
4. Automated tests (`tests/test_tinycd_production_verification.py`) verify that missing or corrupted
   checkpoints fail closed with `STATUS = CHECKPOINT_INVALID`.
5. **Empirical Regression:** TinyCD achieves F1 = 0.8326–0.8390 on `test_10.png` and
   F1 = 79.31%, IoU = 65.71%, OA = 97.99% across the complete 128-scene LEVIR-CD test set.

> **Clarification Note:**
> 17.69% F1 was produced by an untrained randomly initialized SiameseFeatureDiff fallback caused by
> an architecture/checkpoint dispatch misconfiguration. It is NOT a TinyCD result.

---

## 2. Multi-Device Execution in Cross-Modal Specialist

### Anomaly Identified
When running `OpticalSarSpecialist.execute()` on Apple Silicon MPS or CUDA devices:
`RuntimeError: slow_conv2d_forward_mps: input(device='cpu') and weight(device='mps:0') must be on the same device`.

### Root Cause
`preprocessor.align_spatial_dimensions()` creates tensors on CPU. When model encoder weights were
migrated to MPS or CUDA, `opt_batch` and `sar_batch` were not transferred to the encoder's device
before the forward pass.

### Resolution
Updated `specialists/optical_sar/service.py` to dynamically query the active device:
```python
dev = next(self.optical_encoder.parameters()).device
opt_batch = opt_aligned.unsqueeze(0).to(dev)
sar_batch = sar_aligned.unsqueeze(0).to(dev)
intent_vec = self.query_interpreter.get_intent_vector(request.query).unsqueeze(0).to(dev)
```
Outputs are transferred back to CPU for spatial evidence rendering.

---

## 3. CM-EDGE-10: Modality Heuristic Fallback Bypass (CONFIRMED & FIXED)

### Anomaly Identified
`CM-EDGE-10` ("Wrong modality passed as SAR") was recorded as PASS in the original QA suite, but
the verdict was **incorrect**. When two images both declaring `ImageModality.OPTICAL` were submitted
to `OpticalSarSpecialist`, `extract_and_validate_optical_sar_inputs` returned `is_valid=True`,
silently assigning the second optical image into the SAR slot and allowing fabricated cross-modal
inference to complete without error.

### Root Cause
The fallback heuristic in `specialists/optical_sar/schemas.py` contained an unconditional positional
clause that fired even when images had **explicit** declared modalities:
```python
# BUG: fires regardless of declared modality
for idx, img in enumerate(request.images):
    if optical_img is None and (... or idx == 0):
        optical_img = img
    elif sar_img is None and (... or idx == 1):  # idx==1 overrides OPTICAL declaration
        sar_img = img
```
The `idx == 1` guard promoted any second image to the SAR slot irrespective of its
`ImageModality.OPTICAL` declaration, bypassing the multi-modal contract.

### Corrective Action
Refactored `extract_and_validate_optical_sar_inputs` in `specialists/optical_sar/schemas.py`:

1. **First pass** — match images strictly by declared modality (OPTICAL/MULTISPECTRAL → optical
   slot, SAR → SAR slot).
2. **Second pass (filename heuristic)** — applied **only** to images with `ImageModality.UNKNOWN`;
   never overrides an explicit declaration.
3. **Third pass (positional fallback)** — applied **only** when **every** image in the request
   carries `UNKNOWN` modality.

### Verification
- CM-EDGE-10 now returns `is_valid=False` with "No Synthetic Aperture Radar (SAR) image input
  identified in request." ✅
- `test_extract_and_validate_optical_sar_inputs_missing_sar` asserts `is_valid is False` ✅
- CM-EDGE-03 (SAR at index 0, Optical at index 1 with correct metadata) still routes correctly ✅
- Full test suite: **248/248 PASS**

---

## 4. CM-EDGE-11: Corrupted Image Silent Synthetic Fallback (CONFIRMED & FIXED)

### Anomaly Identified
When a corrupted / non-image byte sequence was supplied as the Optical input,
`OpticalSarSpecialist` returned `ToolStatus.SUCCESS` with a plausible-looking land-cover prediction
derived entirely from a randomly-generated synthetic tensor. No error or warning appeared in the API
response.

### Root Cause
`OpticalSarPreprocessor.load_raw_raster()` in `specialists/optical_sar/preprocessing.py` had a bare
`except Exception` clause that substituted any PIL/rasterio decode failure with a (3, 512, 512)
random float32 array:
```python
# BUG: fabricates data on corrupt input
except Exception as exc:
    logger.error(f"... Generating synthetic fallback tensor.")
    arr = np.random.rand(3, 512, 512).astype(np.float32) * 255.0
    return arr, meta  # Inference proceeds on random noise
```
The resulting "analysis" was indistinguishable from a genuine prediction.

### Corrective Action
1. Added `allow_synthetic_fallback: bool = False` to `PreprocessingConfig` in
   `specialists/optical_sar/config.py`. Defaults to `False` (production-safe).
2. Updated `load_raw_raster` in `specialists/optical_sar/preprocessing.py` to raise `IOError`
   on failure unless `allow_synthetic_fallback=True`.
3. Wrapped preprocessing in `specialists/optical_sar/service.py` `execute()` with `try/except`
   returning `ToolStatus.FAILED` with an explicit error message on any `IOError`.

### Verification
- CM-EDGE-11 returns `ToolStatus.FAILED` with:
  `"Preprocessing failed: Failed to load raster image from '...': corrupted or unreadable format (...)"`  ✅
- No fabricated land-cover predictions emitted for corrupt inputs ✅
- `test_preprocessing_file_loading_fallback` passes (uses `allow_synthetic_fallback=True` in test config) ✅
- Full test suite: **248/248 PASS**

---

## 5. Agentic Trace Provenance Gap (CONFIRMED & FIXED)

### Anomaly Identified
The agentic controller QA suite's `has_model_provenance` audit asserted the inference trace entry
must contain `"architecture": "TinyCD"`. Two separate gaps existed:

1. `TinyCDAdapter.__init__` was missing `self.parameter_count = 0` and `self.architecture = "TinyCD"`.
   These attributes were only written after `initialize()` / checkpoint load, creating a window where
   pre-initialization trace reads returned empty/missing values.
2. The Optical-SAR `cmaf_landcover_head` trace entry lacked equivalent provenance fields
   (`model`, `architecture`, `checkpoint`, `sha256`, `parameter_count`), creating an asymmetric
   audit trail between Division 3 and Division 4 specialists.

### Root Cause
Incomplete attribute initialization at construction time rather than checkpoint-load time.

### Corrective Action
1. Added `self.parameter_count = 0` and `self.architecture = "TinyCD"` to `TinyCDAdapter.__init__`
   in `specialists/temporal_change/model_adapter.py`.
2. Enriched the `cmaf_landcover_head` `ExecutionTraceEntry` in `specialists/optical_sar/service.py`
   with full provenance fields: `model`, `architecture`, `checkpoint`, `sha256`, `parameter_count`.
3. Added `self.loaded_checkpoint_path` population in `OpticalSarSpecialist._load_trained_checkpoint()`.

### Verification
- Full 15-entry trace lifecycle verified in correct stage order ✅
- Trace [8] contains `architecture: "TinyCD (Siamese U-Net + MAMB)"`, `sha256`, `parameter_count: 3565034` ✅
- `has_model_provenance = True` asserted by QA suite ✅
- Full test suite: **248/248 PASS**

---

## 6. Schema & Input Edge-Case Log

| Test Case | Scenario | Observed System Behavior | Verdict | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **BT-EDGE-01** | Reverse temporal order (T1 as A, T0 as B) | Inference preserves supplied ordering; no chronological hallucination. | **PASS** | Symmetric spatial diff. |
| **BT-EDGE-02** | Identical images (T0 == T1) | 0.0000% change pixels; no crash or division-by-zero. | **PASS** | Background stability. |
| **BT-EDGE-03** | Single image to bi-temporal tool | `InvalidImageCountError`; `ToolStatus.FAILED`. | **PASS** | Fails closed pre-inference. |
| **BT-EDGE-04** | Three images to bi-temporal tool | `InvalidImageCountError`; `ToolStatus.FAILED`. | **PASS** | Rejects unsupported cardinality. |
| **BT-EDGE-05** | Mismatched dims (512x512 vs 256x256) | Bilinear resampling to 256x256; execution succeeds. | **PASS** | Deterministic resize. |
| **BT-EDGE-06** | Mismatched CRS (EPSG:4326 vs EPSG:32643) | `ToolStatus.FAILED` with explicit CRS mismatch detail. | **PASS** | Prevents erroneous spatial comparisons. |
| **BT-EDGE-07** | Non-overlapping bounding boxes | `IncompatiblePairError`; `ToolStatus.FAILED`. | **PASS** | Zero false-overlap inference. |
| **BT-EDGE-08** | Duplicate timestamps | Logged as micro-interval; execution continues safely. | **PASS** | Supported edge case. |
| **BT-EDGE-09** | Cloudy / degraded optical imagery | Change mask localises contrast discrepancies; no numerical explosion. | **PASS** | Stable under extreme pixel values. |
| **BT-EDGE-10** | Blank / constant-zero raster | Empty change mask (0% change); no NaN or div-by-zero. | **PASS** | Robust numerical bounds. |
| **BT-EDGE-11** | Corrupted bytes (Bi-Temporal) | Header inspection catches corruption; controlled validation error. | **PASS** | Prevents unhandled PIL crash. |
| **BT-EDGE-12** | Wrong modality (SAR to Bi-Temporal) | Rejected at `validate_task_compatibility`. | **PASS** | Modal boundaries respected. |
| **CM-EDGE-01** | Missing SAR image | `is_valid=False` from `extract_and_validate_optical_sar_inputs`. | **PASS** | Multi-modal contract enforced. |
| **CM-EDGE-02** | Missing Optical image | `is_valid=False`. | **PASS** | Multi-modal contract enforced. |
| **CM-EDGE-03** | Reversed modality order (SAR idx 0, Optical idx 1) | Correct modality-aware dispatch to respective encoders. | **PASS** | Metadata-driven routing. |
| **CM-EDGE-04** | Optical-SAR dimension mismatch | SAR grid resampled to Optical grid via bilinear interpolation. | **PASS** | Resampled cleanly. |
| **CM-EDGE-08** | Cloudy Optical + valid SAR | CMAF attends to SAR backscatter; cloud-penetrating prediction succeeds. | **PASS** | CMAF architecture verified. |
| **CM-EDGE-09** | Noisy / degraded SAR | Quantile normalisation clips speckle; forward pass stable. | **PASS** | Preprocessing effective. |
| **CM-EDGE-10** | Two OPTICAL images as Optical-SAR pair | ~~Was: `is_valid=True` (positional heuristic bypass).~~ **Fixed:** `is_valid=False` — "No SAR image identified." | **PASS** *(post-fix)* | Heuristic now scoped to UNKNOWN modality only. |
| **CM-EDGE-11** | Corrupted bytes as Optical input | ~~Was: `ToolStatus.SUCCESS` via silent synthetic fallback.~~ **Fixed:** `ToolStatus.FAILED` with exact corrupt-file error. | **PASS** *(post-fix)* | Fail-closed; no fabricated predictions. |
| **MIS-02** | Zero images in request | Schema rejects: "List should have at least 1 item after validation, not 0." | **PASS** | Schema-level fail-closed. |

---

## 7. Current Certified Component Status

1. **Bi-Temporal Intelligence (Division 3)**
   - **Status: PASS — EMPIRICAL MODEL VALIDATION**
   - Checkpoint frozen, strictly loaded, verified on 128 held-out LEVIR-CD scenes.

2. **Optical-SAR Cross-Modal Intelligence (Division 4)**
   - **Status: PASS — EMPIRICAL MODEL VALIDATION**
   - Checkpoint frozen, strictly loaded, verified on 4,950 held-out WHU-OPT-SAR test tiles.
   - Post-session fixes: CM-EDGE-10 modality heuristic bypass (§3), CM-EDGE-11 silent synthetic
     fallback (§4), provenance trace enrichment (§5).

3. **Agentic Controller (Division 1 Core)**
   - **Status: PASS — FUNCTIONAL VALIDATION & ROUTING ONLY**
   - Deterministic routing, intent resolution, prompt-injection defence, and 15-stage execution
     trace auditing are 100% operational.
   - Post-session fix: Model provenance attributes initialised at adapter construction; CMAF trace
     enriched with full provenance metadata (§5).

4. **Single-Image Vision-Language (Division 2)**
   - **Status: PASS — ROUTING ONLY (TRAINING PENDING)**
   - Single-image Qwen accuracy is NOT evaluated or claimed per QA audit scope.
     Training remains in progress in Colab.

---

*Full regression suite: **248/248 tests passing** — 2026-09-11.*
