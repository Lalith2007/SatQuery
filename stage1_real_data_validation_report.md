# Stage 1 Real Data Materialization & Sensor Conversion Validation Report

**Status**: **PASS (100% Verified)**  
**Generated Panels Directory**: `data/curated_mixture/diagnostics/stage1_sensor_validation`  
**Report Artifact**: `data/curated_mixture/stage1_real_data_validation_report.json`  

---

## 1. Summary of Verification

| Metric | Target / Requirement | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Materialized S1/S2 Pairs** | $\ge 25$ pairs | **48 verified pairs** | **PASS** |
| **Operational Granules Tested** | Multiple granules | **4 unique granules (T34TCR, T34TCS, T34TFN, T34TEQ)** | **PASS** |
| **Geographic Validation Note** | Distinct operational label | **Conversion Sanity Sample (1 country: Serbia)** | **PASS** |
| **SAR Channel 3 Formula** | $\mathrm{VV}_{\text{dB}} - \mathrm{VH}_{\text{dB}}$ | **`vv - vh` ($10 \log_{10}(\mathrm{VV}_{\text{lin}} / \mathrm{VH}_{\text{lin}})$)** | **PASS** |
| **S1 GeoTIFF Validation** | Valid `(2, 120, 120)` float32 | **48 / 48 valid (0 NaN, 0 Inf)** | **PASS** |
| **S2 GeoTIFF Validation** | Valid `(10, 120, 120)` uint16 | **48 / 48 valid (0 NaN, 0 Inf)** | **PASS** |
| **S1 Sensor Conversion** | R=VV, G=VH, B=ratio | **48 passed (0 failures)** | **PASS** |
| **S2 Sensor Conversion** | R=B04, G=B03, B=B02 | **48 passed (0 failures)** | **PASS** |
| **Diagnostic Panels** | $\ge 5$ panels generated | **8 high-res panels saved (across 4 granules)** | **PASS** |
| **Grounding Precision** | Roundtrip error $\le 0.002$ | **0.000000 max error** | **PASS** |
| **Qwen Multimodal Processor** | Correct token & label masking | **PASS (Supervised: 443, Masked: 1237)** | **PASS** |

> [!NOTE]
> **Geographic Scope Distinction**: This 48-pair materialization validation is an operational multi-sensor conversion and format sanity check across 4 parent granules in Serbia, and is **NOT** described as broad geographic validation. Broad geographic diversity is preserved in the full Stage-1 manifest consisting of **8,000 pairs across 115 parent granules and 8 European countries**.

---

## 2. Critical SAR Representation Audit

* **Source Units**: `dB` (calibrated radar backscatter $\sigma^0$)
* **Actual Channel 3 Formula**: `ratio_db = vv - vh` (i.e. $\mathrm{VV}_{\text{dB}} - \mathrm{VH}_{\text{dB}}$)
* **Documented Channel 3 Formula**: $\mathrm{VV}_{\text{dB}} - \mathrm{VH}_{\text{dB}} = 10 \log_{10}\left(\frac{\mathrm{VV}_{\text{linear}}}{\mathrm{VH}_{\text{linear}}}\right)$
* **Correct**: **`True`**
* **Verification Detail**:
  * Input arrays from GeoTIFF are verified to be in decibels (mean backscatter $-25$ to $-1$ dB).
  * The converter executes `ratio_db = vv - vh`, which correctly implements the logarithmic ratio of linear intensities without taking an erroneous logarithm of decibel values.
  * In `tests/test_sensor_converters.py`, `test_sentinel1_ratio_mathematical_identity` proves that conversion from linear intensities and conversion from dB values yield identical representations within machine precision.

---

## 3. Sensor Conversion Integrity

### A. Sentinel-1 SAR (Dual-Pol Ratio)
* **Channel 1 (R)**: VV backscatter normalized from $[-25, 0]$ dB.
* **Channel 2 (G)**: VH backscatter normalized from $[-32, -5]$ dB.
* **Channel 3 (B)**: $\mathrm{VV}_{\text{dB}} - \mathrm{VH}_{\text{dB}}$ cross-ratio normalized from $[-5, 20]$ dB.
* **Integrity**: Zero NaNs, zero Infs, deterministic uint8 RGB conversion.

### B. Sentinel-2 MSI (True Color Composite)
* **Channel 1 (R)**: Band 04 (Red, 665nm).
* **Channel 2 (G)**: Band 03 (Green, 560nm).
* **Channel 3 (B)**: Band 02 (Blue, 490nm).
* **Integrity**: Correct band indices (2, 1, 0), zero NaNs, scaled to $[0, 255]$.

---

## 4. Grounding Encode / Decode Roundtrip Audit
Representative normalized coordinates $[y_1, x_1, y_2, x_2]$ encoded to `<|box_start|>(y1,x1),(y2,x2)<|box_end|>` in $[0, 1000)$ integer space and decoded back:
* **Max Measured Error**: **0.000000**
* **Quantization Tolerance**: $\le 0.002$
* **Result**: **PASS (Well within integer binning resolution)**

---

## 5. Qwen Multimodal Processor & Label Masking Audit
* **Input IDs Shape**: `[6, 280]`
* **Labels Shape**: `[6, 280]`
* **Pixel Values Shape**: `[384, 1176]`
* **Supervised Assistant Tokens**: **443**
* **Masked User/Prompt/Image Tokens (`-100`)**: **1237**
* **Accidental Image/Prompt Supervision**: **0 (Strictly zero)**

---

## 6. Final Confirmation
```
SAR REPRESENTATION: PASS
STAGE 1 MATERIALIZATION: PASS
GROUNDING: PASS
QWEN PROCESSOR: PASS
TRAINING: NOT STARTED
```
