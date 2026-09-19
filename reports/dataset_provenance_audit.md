# SatQuery AI — Dataset Provenance & Empirical Evaluation Audit

**Auditor Role:** Senior ML Evaluation Auditor & Remote-Sensing AI Systems Validator  
**Problem Statement:** Smart India Hackathon — ISRO/SAC Problem Statement  
**Audit Scope:** Forensic Dataset Provenance Verification for **RSVQA**, **CDVQA**, and **ISRO/SAC Target Data**, cross-referenced against Production Baseline Benchmarks (**LEVIR-CD** and **WHU-OPT-SAR**).  
**Audit Date:** 2026-09-11  
**Audit Status:** FINAL — CERTIFIED FORENSIC RECORD  

---

## 1. Executive Summary

This forensic audit evaluates the empirical evaluation status of all datasets referenced within the SatQuery AI system. Per strict non-fabrication mandates and the **Final Evaluation Rule**, an evaluation is certified as **EMPIRICALLY EVALUATED** if and only if:
1. Real samples from an established or held-out split were ingested.
2. Real production model weights executed forward inference on those samples.
3. Model predictions were objectively scored against held-out ground truth.

Any dataset falling short of all three criteria—including harness unit tests, synthetic verification strings, unexecuted manifests, or architectural stubs—is strictly categorized as **HARNESS ONLY / NOT EVALUATED**.

### Summary Provenance Classification

| Dataset | Modality | Component / Evaluator | Evaluation Status | Forensic Evidence Summary |
|:---|:---|:---|:---:|:---|
| **LEVIR-CD** | Bi-Temporal Optical (0.5m) | Division 3 (`TinyCD`) | **EMPIRICALLY EVALUATED** | 128 held-out test scenes (8,388,608 px), live forward pass, 79.31% F1, 65.71% IoU. Production checkpoint frozen. |
| **WHU-OPT-SAR** | Optical + SAR (CSK/GF-3) | Division 4 (`CMAF`) | **EMPIRICALLY EVALUATED** | Held-out test split (15 scenes, 4,950 tiles), dual-stream forward pass, 71.71% OA, 35.08% mIoU, 46.62% Macro-F1, 74.18% Weighted-F1. Checkpoint frozen. |
| **RSVQA** | Sentinel-2 Optical (10m) | `RSVQAEvaluator` | **HARNESS ONLY / NOT EVALUATED** | Parquet file exists on disk; `RSVQAEvaluator` logic & arithmetic verified via 3 mock records in unit tests. **Zero VLM model inference executed**. |
| **CDVQA** | Bi-Temporal Optical | Composed Pipeline (`TinyCD` + `Qwen`) | **COMPOSED WORKFLOW FUNCTIONAL** | Visual crop handoff functional (TinyCD detection + ROI crop handoff); VLM returns honest MODEL_NOT_READY pending training. |
| **ISRO/SAC Target** | Cartosat-2S & RISAT SAR | `ISROSACGenericEvaluator` | **NOT EVALUATED / STUB** | Zero bytes of real spaceborne Cartosat or RISAT data exist in environment (classified/restricted). Generic evaluator is an **unexecuted interface stub**. |

> [!IMPORTANT]
> **False-Positive Warning Regarding System Reports:**  
> In internal test suites (`tests/test_evaluation_benchmarks.py`) and summary validation reports (`reports/SYSTEM_VALIDATION_REPORT.md`), entries marked as `PASS` for RSVQA, CDVQA, and ISRO/SAC represent **evaluation harness plumbing and metric arithmetic checks** (e.g., verifying that string cleaning works or BLEU formulas compute correctly on dummy dictionaries). They do **NOT** represent empirical model evaluation or model accuracy on real remote-sensing benchmarks.

---

## 2. Audit Methodology & The Final Rule

### 2.1 The "Final Rule" for Empirical Evaluation
To prevent conflation between software testing and machine learning validation, this audit enforces the following criteria:

$$\text{Empirically Evaluated} \iff (\text{Real Data Ingested}) \land (\text{Real Model Forward Pass}) \land (\text{Held-Out GT Comparison})$$

A benchmark or dataset is certified as **EMPIRICALLY EVALUATED** if and only if:
- **Criterion A (Real Data):** Actual remote-sensing imagery and corresponding annotations from an authoritative benchmark or held-out test split are loaded.
- **Criterion B (Production Inference):** Production model weights (e.g., trained checkpoint tensors) are instantiated and execute real forward inference over the dataset without mocked outputs, rule-based bypasses, or cached predictions.
- **Criterion C (Held-Out Ground Truth):** Model output predictions are evaluated against verified ground-truth labels using standardized evaluation metrics.

### 2.2 Forensic Disqualification Rules (What Does NOT Count)
Per auditor instructions, none of the following constitute empirical evaluation:
- Presence of raw dataset files, parquet archives, or manifests on disk without executed model inference.
- Unit tests or integration tests passing on synthetic, mock, or hardcoded sample dictionaries.
- Evaluator metric arithmetic verification (e.g., verifying that an evaluator outputs `1.0` when given identical input strings).
- Agentic controller natural-language routing tests (e.g., routing `"What changed?"` to a specialist).
- Historical cached logs, documentation claims, or unverified script outputs.

---

## 3. Deep-Dive Forensic Findings: Target Datasets

### 3.1 RSVQA (Remote Sensing Visual Question Answering)
*Reference: Lobry et al., IEEE TGRS 2020 (RSVQA Low-Resolution / High-Resolution)*

#### Codebase Artifacts Inspected:
- **Evaluator Implementation:** [`evaluation/benchmarks/rsvqa.py`](file:///Users/lalith/Desktop/SatQuery/evaluation/benchmarks/rsvqa.py)
  - Defines `RSVQAEvaluator(BaseBenchmarkEvaluator)`.
  - Implements static text normalization (`clean_text`), numerical word parsing (`extract_number`), and metric computation for presence, comparison, and count queries (MAE / RMSE).
- **Data on Disk:**
  - Directory: [`data/benchmark_samples/rsvqa/`](file:///Users/lalith/Desktop/SatQuery/data/benchmark_samples/rsvqa/)
  - File: `rsvqa_lr_val.parquet` (174,052,999 bytes / ~166 MB).
  - File: `sample_records.json` (2,267 bytes).
  - Subdirectory: `images/`.
- **Test Invocations:**
  - Test file: [`tests/test_evaluation_benchmarks.py`](file:///Users/lalith/Desktop/SatQuery/tests/test_evaluation_benchmarks.py#L76-L98) (`test_rsvqa_evaluation`).
  - Input: 3 synthetic mock dictionaries:
    ```python
    predictions = [{"answer": "yes"}, {"answer": "4"}, {"answer": "no"}]
    ground_truths = [{"answer": "yes", "type": "presence"}, {"answer": "4", "type": "count"}, {"answer": "no", "type": "comparison"}]
    ```
  - Test outcome: Verifies evaluator arithmetic produces `overall_accuracy: 1.0` and `count_rmse: 0.0`.
- **System Validation Documentation:**
  - [`reports/SYSTEM_VALIDATION_REPORT.md`](file:///Users/lalith/Desktop/SatQuery/reports/SYSTEM_VALIDATION_REPORT.md#L233) explicitly records:
    > `| RSVQA-LR | RSVQAEvaluator | Codec & Harness Arithmetic | PASS — HARNESS ONLY (Arithmetic & text normalization logic verified; not Qwen model accuracy). |`

#### Forensic Analysis:
While `rsvqa_lr_val.parquet` is physically present on disk, **no end-to-end model inference has been executed on this dataset**. The Vision-Language component (Division 2, targeting Qwen2.5-VL) remains in a routing-only state pending external fine-tuning. The production specialists (Division 3 TinyCD and Division 4 CMAF) do not process single-image VQA queries.

#### Audit Verdict:
**HARNESS ONLY / NOT EVALUATED**.  
The evaluator arithmetic and parquet codec are fully functional, but no empirical model evaluation has taken place.

---

### 3.2 CDVQA (Change Detection Visual Question Answering)
*Reference: Remote Sensing Change Detection VQA Benchmark Suites*

#### Codebase Artifacts Inspected:
- **Evaluator Implementation:** [`evaluation/benchmarks/cdvqa.py`](file:///Users/lalith/Desktop/SatQuery/evaluation/benchmarks/cdvqa.py)
  - Defines `CDVQAEvaluator(BaseBenchmarkEvaluator)`.
  - Implements tokenization, modified n-gram BLEU (`compute_bleu_n`), Longest Common Subsequence ROUGE-L (`compute_rouge_l`), and binary change question classification.
- **Data on Disk:**
  - No dedicated CDVQA dataset directory exists under `data/benchmark_samples/`.
  - The harness leverages image pairs from [`data/official_levir_cd/`](file:///Users/lalith/Desktop/SatQuery/data/official_levir_cd/) to test the data-loading pipeline.
- **Test Invocations:**
  - Test file: [`tests/test_evaluation_benchmarks.py`](file:///Users/lalith/Desktop/SatQuery/tests/test_evaluation_benchmarks.py#L103-L122) (`test_cdvqa_evaluation`).
  - Input: 2 hardcoded synthetic text strings:
    ```python
    predictions = [
        {"answer": "Yes, new commercial buildings and paved roads appeared."},
        {"answer": "no"},
    ]
    ground_truths = [
        {"answer": "Yes, new commercial buildings and asphalt roads were constructed."},
        {"answer": "no"},
    ]
    ```
  - Test outcome: Verifies that BLEU-1 exceeds 0.5 and binary change accuracy equals 1.0.
- **System Validation Documentation:**
  - [`reports/SYSTEM_VALIDATION_REPORT.md`](file:///Users/lalith/Desktop/SatQuery/reports/SYSTEM_VALIDATION_REPORT.md#L236) explicitly records:
    > `| CDVQA | CDVQAEvaluator | Textual Metric Formulation | PASS — HARNESS ONLY (BLEU & ROUGE string matching algorithms verified). |`

#### Forensic Analysis:
A critical distinction must be maintained between **Change Detection (CD)** and **Change Detection Visual Question Answering (CDVQA)**:
- LEVIR-CD is a pixel-level change detection benchmark evaluated using TinyCD (F1/IoU on binary masks). This is fully and empirically evaluated under Division 3.
- CDVQA requires natural language questions and answers regarding bi-temporal changes (e.g., describing what type of structures were built).
The SatQuery system contains an evaluator for CDVQA text metrics, but has never executed an empirical evaluation of a VQA model on a real CDVQA benchmark dataset.

#### Audit Verdict:
**HARNESS ONLY / NOT EVALUATED**.  
The metric computation harness is verified, but no empirical CDVQA model evaluation has occurred.

---

### 3.3 ISRO/SAC Target Data (Cartosat-2S & RISAT SAR)
*Reference: Indian Space Research Organisation / Space Applications Centre Mission Datasets*

#### Codebase Artifacts Inspected:
- **Evaluator Implementation:** [`evaluation/benchmarks/isro_sac.py`](file:///Users/lalith/Desktop/SatQuery/evaluation/benchmarks/isro_sac.py)
  - Defines `ISROSACGenericEvaluator(BaseBenchmarkEvaluator)`.
  - Implements generic evaluation logic accepting arbitrary prediction and ground-truth dictionaries without pre-programmed answers.
  - Computes exact match VQA accuracy, token-level F1, spatial bounding box IoU (mIoU and P@50), and per-sensor breakdowns (`cartosat_2s`, `risat_sar`, `joint_fusion`).
- **Data on Disk:**
  - **Zero bytes** of real ISRO spaceborne mission data exist in the workspace.
  - Raw Cartosat-2S (0.65m optical) and RISAT-1/1A (C-band SAR) imagery are classified/proprietary mission archives requiring institutional security clearance.
- **Test Invocations:**
  - Test file: [`tests/test_evaluation_benchmarks.py`](file:///Users/lalith/Desktop/SatQuery/tests/test_evaluation_benchmarks.py#L127-L148) (`test_isro_sac_generic_evaluation_without_hardcoded_answers`).
  - Input: 2 mock synthetic records simulating sensor metadata:
    ```python
    predictions = [
        {"answer": "Metallic port cranes visible through cloud cover.", "bbox": [0.2, 0.2, 0.4, 0.4], "sensor": "risat_sar"},
        {"answer": "High-resolution agricultural crop plots.", "bbox": [0.5, 0.5, 0.9, 0.9], "sensor": "cartosat_2s"},
    ]
    ```
- **System Validation Documentation:**
  - [`reports/SYSTEM_VALIDATION_REPORT.md`](file:///Users/lalith/Desktop/SatQuery/reports/SYSTEM_VALIDATION_REPORT.md#L30-L31) states:
    > `- ACTUAL ISRO/SAC DATA EVALUATION = NOT EXECUTED (Classified spaceborne archives unavailable in development environment)`  
    > `- ISRO/SAC EVALUATOR COMPATIBILITY = PASS — HARNESS ONLY (Verified with simulated metadata)`
  - Line 62: `| ISRO Cartosat-2S & RISAT | 0 scenes | NOT AVAILABLE (Classified mission data) |`
  - Line 276: `| ISRO/SAC ACTUAL DATA EVALUATION | NOT EXECUTED |`

#### Forensic Analysis:
The evaluator code was architected as a plug-and-play evaluation harness to allow the ISRO/SAC technical jury to run evaluation on private spaceborne test splits during competition judging. Within this workspace, no real data exists and no empirical evaluation has been performed.

#### Audit Verdict:
**NOT EVALUATED / STUB**.  
The evaluation harness schema is implemented and tested with mock dictionaries, but actual data evaluation is strictly **NOT EXECUTED**.

---

## 4. Comprehensive Dataset Provenance Matrix

The following table provides the definitive, certified provenance record across all public and target benchmarks evaluated against the SatQuery repository:

| Dataset | Modality / Resolution | Role in System | Split Used | Samples on Disk | Model Inference Executed? | Provenance Status | Certified Score / Metrics |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **LEVIR-CD** | Bi-temporal Optical (0.5m) | Division 3 Specialist (`TinyCD`) | Official Held-out Test | 128 scenes (1024x1024) | **YES** (Production Checkpoint) | **OFFICIAL FULL EVALUATION COMPLETE** | **F1: 79.31%**, **IoU: 65.71%**, OA: 97.99% (8,388,608 px) |
| **WHU-OPT-SAR** | Optical (0.45m) + SAR (5m) | Division 4 Specialist (`CMAF`) | Official Held-out Test | 15 scenes (4,950 tiles) | **YES** (Production Checkpoint) | **OFFICIAL FULL EVALUATION COMPLETE** | **OA: 71.71%**, **mIoU: 35.08%**, Macro-F1: 46.62%, Weighted-F1: 74.18% |
| **RSVQA-LR** | Sentinel-2 Optical (10m) | Benchmark Harness | Val split (parquet) | 1 Parquet (~166 MB) + 2k records | **NO** (No model inference executed) | **DATA READY / MODEL NOT READY** | None certified. (Harness arithmetic verified on synthetic items) |
| **VRSBench Grounding** | High-Res Aerial Optical | Single-Image Specialist | Sample records | Referring JSON + imagery | **NO** (Qwen fine-tuning pending) | **DATA READY / MODEL NOT READY** | None certified. (IoU/mIoU harness verified) |
| **VRSBench VQA** | High-Res Aerial Optical | Single-Image Specialist | Sample records | VQA JSON + imagery | **NO** (Qwen fine-tuning pending) | **DATA READY / MODEL NOT READY** | None certified. (Exact match & F1 harness verified) |
| **VRSBench Captioning** | High-Res Aerial Optical | Single-Image Specialist | Sample records | Image rasters on disk | **NO** (Qwen fine-tuning pending) | **DATA READY / MODEL NOT READY** | None certified. (BLEU/ROUGE harness verified) |
| **CDVQA** | Bi-temporal Optical | Composed Pipeline (TinyCD + VLM) | LEVIR-CD pairs on disk | 128 test scenes | **COMPOSED WORKFLOW FUNCTIONAL** | **END-TO-END FUNCTIONAL — VLM CHECKPOINT PENDING** | Composed workflow functional (TinyCD detection + ROI crop handoff); VLM returns honest MODEL_NOT_READY. |
| **BigEarthNet.txt** | Sentinel-1 SAR + Sentinel-2 MSI | Single-Image Specialist | Test split (held-out) | 8,000 Stage 1 pairs (QUARANTINED) | **NO** (Qwen fine-tuning pending) | **DATA READY / MODEL NOT READY** | None certified. (Stage 1 subset strictly training only) |
| **ISRO/SAC Target** | Cartosat-2S + RISAT SAR | Jury Generic Evaluation Harness | None (Classified data) | 0 bytes | **NO** (Data not available) | **NOT AVAILABLE / AWAITING OFFICIAL DATA** | None certified. (Generic harness interface verified) |

---

## 5. Non-Interference & Integrity Certification

1. **Division 3 & Division 4 Freeze**:
   - The production checkpoint `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth` (SHA-256: `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`) was not modified, retrained, or re-calibrated during this audit.
   - The production checkpoint `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth` (SHA-256: `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`) remains intact and unmodified.
2. **BigEarthNet Stage 1 Quarantine Verification**:
   - The 8,000 materialized Sentinel-1/Sentinel-2 image pairs referenced in `data/curated_mixture/bigearthnet_stage1_manifest.jsonl` are strictly isolated to training only. No benchmark numbers are derived from them.
3. **Zero Fabrication Mandate**:
   - No benchmark scores have been fabricated, estimated, or extrapolated.
   - All statuses reflect factual ground conditions in the repository.

---

## 6. Cross-Referenced Audit Deliverables

For deep-dive architectural and operational evidence, see:
- [`reports/change_vqa_pipeline_audit.md`](file:///Users/lalith/Desktop/SatQuery/reports/change_vqa_pipeline_audit.md) & [`reports/change_vqa_pipeline_audit.json`](file:///Users/lalith/Desktop/SatQuery/reports/change_vqa_pipeline_audit.json): Full code-level call graph, runtime tests on `test_10.png`, visual evidence gap analysis, and edge-case matrix for Change-VQA.
- [`reports/official_public_benchmark_audit.md`](file:///Users/lalith/Desktop/SatQuery/reports/official_public_benchmark_audit.md) & [`reports/official_public_benchmark_audit.json`](file:///Users/lalith/Desktop/SatQuery/reports/official_public_benchmark_audit.json): Comprehensive official public benchmark evaluation scoreboard covering all 8 benchmark categories plus ISRO/SAC.

