# SatQuery AI — Official Public Benchmark Evaluation Audit

**Auditor Role:** Senior Remote-Sensing AI Integration Engineer, Benchmark Engineer, and Production QA Validator  
**Problem Statement:** Smart India Hackathon — ISRO/SAC Problem Statement  
**Audit Scope:** Official Public Benchmark Coverage across **BigEarthNet.txt**, **VRSBench (Captioning, Grounding, VQA)**, **RSVQA**, **CDVQA**, **LEVIR-CD**, **WHU-OPT-SAR**, and **ISRO/SAC Target Data**.  
**Audit Date:** 2026-09-11  
**Audit Status:** FINAL — COMPREHENSIVE BENCHMARK SCOREBOARD  

---

## 1. Executive Summary & Audit Mandate

This audit provides the authoritative, forensic evaluation scoreboard for all public and target benchmarks evaluated against SatQuery AI. In strict compliance with the **Non-Fabrication Policy** and **Dataset Acquisition Policy**, benchmark results are certified as **EMPIRICALLY COMPLETE** only when all six mandatory empirical conditions are satisfied:
1. Real benchmark images ingested from official sources.
2. Real benchmark annotations/questions loaded from held-out splits.
3. Real production or fine-tuned model checkpoint instantiated.
4. Real forward model inference executed across the evaluation split.
5. Real model predictions generated and recorded.
6. Objective ground-truth comparison computed using benchmark-standard metrics.

Where model fine-tuning is pending (such as Division 2 Vision-Language fine-tuning on BigEarthNet in Google Colab), the benchmark is formally classified as **DATA READY / MODEL NOT READY**. No benchmark accuracy is fabricated, estimated, or inferred from evaluation harnesses.

---

## 2. Master Public Benchmark Status Scoreboard

| Benchmark Suite | Specific Task | Modality / Sensor | Evaluator / Specialist | Split / Evaluation Volume | Model Weights Status | Official Benchmark Status | Certified Metric Results |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **LEVIR-CD** | Bi-Temporal Change Detection | Optical ($0.5$m GSD) | `bitemporal_change_specialist`<br>(TinyCD) | Official Held-Out Test<br>($128$ scenes, $8,388,608$ px) | Production Frozen<br>(SHA-256: `b9a10093...`) | **OFFICIAL FULL EVALUATION COMPLETE** | **F1: 79.31%**<br>**IoU: 65.71%**<br>OA: 97.99%<br>Precision: 83.36%<br>Recall: 75.63% |
| **WHU-OPT-SAR** | Cross-Modal Land-Cover Classification | Optical ($0.45$m) + SAR (CSK/GF-3 $5$m) | `optical_sar_cross_modal_specialist`<br>(CMAF) | Official Held-Out Test<br>($15$ scenes, $4,950$ tiles) | Production Frozen<br>(SHA-256: `26288ce0...`) | **OFFICIAL FULL EVALUATION COMPLETE** | **OA: 71.71%**<br>**mIoU: 35.08%**<br>Macro-F1: 46.62%<br>Weighted F1: 74.18% |
| **RSVQA-LR** | Remote Sensing VQA (Presence, Count, Comp.) | Sentinel-2 Optical ($10$m GSD) | `RSVQAEvaluator`<br>(`evaluation/benchmarks/rsvqa.py`) | Val split on disk<br>($174$ MB Parquet, $2,000$ records) | Pending Colab Training<br>(Qwen2.5-VL-3B) | **DATA READY / MODEL NOT READY** | None certified.<br>Harness arithmetic verified.<br>Model forward pass pending. |
| **VRSBench Grounding** | Text-Guided Bounding Box Localization | High-Resolution Optical ($0.5-2$m GSD) | `VRSBenchEvaluator`<br>(`evaluation/benchmarks/vrsbench.py`) | Sample records on disk<br>(Referring expressions) | Pending Colab Training<br>(Qwen2.5-VL-3B) | **DATA READY / MODEL NOT READY** | None certified.<br>IoU/mIoU/P@50 harness verified.<br>Model forward pass pending. |
| **VRSBench VQA** | Remote Sensing VQA (Presence, Count, Landcover) | High-Resolution Optical ($0.5-2$m GSD) | `VRSBenchEvaluator`<br>(`evaluation/benchmarks/vrsbench.py`) | Sample records on disk<br>(VQA question pairs) | Pending Colab Training<br>(Qwen2.5-VL-3B) | **DATA READY / MODEL NOT READY** | None certified.<br>Exact match/Token F1 verified.<br>Model forward pass pending. |
| **VRSBench Captioning** | Detailed Remote-Sensing Scene Captioning | High-Resolution Optical ($0.5-2$m GSD) | `VRSBenchEvaluator`<br>(`evaluation/benchmarks/vrsbench.py`) | Image records on disk | Pending Colab Training<br>(Qwen2.5-VL-3B) | **DATA READY / MODEL NOT READY** | None certified.<br>BLEU/ROUGE-L formulas verified.<br>Model forward pass pending. |
| **CDVQA** | Change Detection Visual Question Answering | Bi-Temporal High-Resolution Optical | Composed Pipeline<br>(TinyCD + Qwen2.5-VL) | LEVIR-CD pairs on disk<br>(128 test scenes) | Composed Pipeline Verified<br>(VLM Weights Pending) | **END-TO-END FUNCTIONAL — VLM CHECKPOINT PENDING** | Pipeline operational.<br>TinyCD detects change & extracts crops.<br>VLM returns MODEL_NOT_READY (honest). |
| **BigEarthNet.txt** | Multi-Modal Zero-Shot / Classification | Sentinel-1 SAR + Sentinel-2 MSI | `SingleImageRSSpecialistTool` | 8,000 pairs in Stage 1<br>(Strictly Quarantined for Training) | Pending Colab Training<br>(Qwen2.5-VL-3B) | **DATA READY / MODEL NOT READY** | None certified.<br>Training subset strictly isolated.<br>Test split awaits checkpoint. |
| **ISRO/SAC Target** | Multi-Sensor Optical-SAR Joint Intelligence | Cartosat-2S ($0.65$m) + RISAT SAR ($1-3$m) | `ISROSACGenericEvaluator`<br>(`evaluation/benchmarks/isro_sac.py`) | $0$ bytes in environment<br>(Classified mission rasters) | Unexecuted Interface Stub | **NOT AVAILABLE / AWAITING OFFICIAL DATA** | None certified.<br>Private jury interface ready.<br>Awaiting official dataset release. |

---

## 3. Dataset Acquisition Policy & Quarantining Rules

### 3.1 BigEarthNet Stage 1 Training Isolation
- **Quarantine Principle**: The 8,000 materialized Sentinel-1/Sentinel-2 image pairs referenced in `data/curated_mixture/bigearthnet_stage1_manifest.jsonl` are **STRICTLY TRAINING ONLY**.
- **Zero Evaluation Contamination**: No benchmark evaluation numbers may be derived from the 8,000 training pairs. Official evaluation must execute exclusively on held-out test splits (e.g., `BigEarthNet.txt` test split or official validation partition).

### 3.2 Dataset Sampling & Provenance Requirements
For large benchmark suites evaluated in constrained development environments:
1. Subsets must be selected deterministically using fixed seeds.
2. Every sampled subset must record a `subset_manifest.jsonl` (with dataset name, split, sample ID, question ID, seed) and a `subset_summary.json`.
3. Terminology must strictly state: *"Official [dataset] test split — N-sample evaluation subset"*. It must never be misrepresented as full-benchmark performance.

---

## 4. Deep-Dive Benchmark Status Analyses

### 4.1 LEVIR-CD (Division 3 — Bi-Temporal Change Intelligence)
- **Model Architecture**: TinyCD (Siamese U-Net + MAMB Space-Time Attention).
- **Checkpoint**: `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`.
- **SHA-256**: `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`.
- **Parameter Count**: $3,565,034$ (Strict state-dict load: 0 missing, 0 unexpected keys).
- **Evaluation Split**: Official held-out test split ($128$ scenes, $1024 \times 1024$ downsampled to $256 \times 256$, total $8,388,608$ evaluated pixels).
- **Certified Empirical Performance**:
  - **F1 Score**: **79.31%**
  - **Intersection over Union (IoU)**: **65.71%**
  - **Overall Accuracy (OA)**: **97.99%**
  - **Precision**: **83.36%**
  - **Recall**: **75.63%**
  - **Specificity**: **99.19%**
  - **True Positives**: $323,254$ pixels | **False Positives**: $64,540$ pixels | **False Negatives**: $104,160$ pixels
- **Status**: **OFFICIAL FULL EVALUATION COMPLETE** (FROZEN PRODUCTION).

---

### 4.2 WHU-OPT-SAR (Division 4 — Cross-Modal Optical-SAR Analysis)
- **Model Architecture**: CMAF (Cross-Modal Attention Fusion — Dual ResNet-18 Encoders + 8-Class Segmentation Head).
- **Checkpoint**: `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`.
- **SHA-256**: `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`.
- **Parameter Count**: $19,755,144$ (Strict load verified).
- **Evaluation Split**: Official held-out test split ($15$ full scenes, $4,950$ non-overlapping tiles of $256 \times 256$, $308,687,656$ valid pixels).
- **Certified Empirical Performance**:
  - **Overall Accuracy (OA)**: **71.71%** (0.717099)
  - **mean IoU (mIoU)**: **35.08%** (0.350812)
  - **Macro F1 Score**: **46.62%** (0.466187)
  - **Weighted F1 Score**: **74.18%** (0.741753)
  - **Weighted IoU**: **60.38%**
  - **Per-Class IoU**:
    - Farmland: 48.06%
    - City: 58.74%
    - Village: 10.98%
    - Water: 75.31%
    - Forest: 58.07%
    - Road: 0.17%
    - Others: 2.22%
  - **Reconciliation Note**: An earlier unverified draft entry noted 84.44% OA / 0.8122 Kappa. Forensic audit proved that 84.44% was an ungrounded draft entry lacking any test log, confusion matrix, or evaluation script in git history. The certified frozen checkpoint evaluation across all 15 held-out scenes (4,950 tiles, 308,687,656 labeled pixels) is OA: 71.71%, mIoU: 35.08%, Macro-F1: 46.62%, Weighted-F1: 74.18% (documented in `specialists/optical_sar/eval_results/v3_fresh/v3_fresh_evaluation_report.md` and `reports/cross_modal_empirical_evaluation.md`). The frozen production result has been fully restored and certified.
- **Status**: **OFFICIAL FULL EVALUATION COMPLETE** (FROZEN PRODUCTION).

---

### 4.3 RSVQA (Remote Sensing Visual Question Answering)
- **Evaluator**: `RSVQAEvaluator` in `evaluation/benchmarks/rsvqa.py`.
- **Data on Disk**: `data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet` ($174,052,999$ bytes / ~166 MB), containing $2,000$ validation records and Sentinel-2 low-resolution optical imagery.
- **Harness Status**: Metric calculation (Overall Accuracy, Presence Accuracy, Comparison Accuracy, Count MAE, Count RMSE) is verified via unit tests (`tests/test_evaluation_benchmarks.py`).
- **Model Execution Status**: Qwen2.5-VL remote-sensing fine-tuning is currently pending in Colab. No forward model inference has been executed against `rsvqa_lr_val.parquet`.
- **Status**: **DATA READY / MODEL NOT READY**.

---

### 4.4 VRSBench (Visual Question Answering, Grounding, and Captioning)
- **Evaluator**: `VRSBenchEvaluator` in `evaluation/benchmarks/vrsbench.py`.
- **Data on Disk**: `data/benchmark_samples/vrsbench/` contains `vrsbench_sample_vqa.json`, `vrsbench_sample_referring.json`, and corresponding high-resolution aerial imagery.
- **Harness Status**: Dual evaluation pipeline computing VQA Exact Match, Token F1, spatial bounding box IoU, mIoU, and P@50 ($IoU \ge 0.5$) is verified in `tests/test_evaluation_benchmarks.py`.
- **Model Execution Status**: Single-image Vision-Language model (Qwen2.5-VL) is awaiting checkpoint fine-tuning.
- **Status**: **DATA READY / MODEL NOT READY**.

---

### 4.5 CDVQA (Change Detection Visual Question Answering)
- **Evaluator**: `CDVQAEvaluator` in `evaluation/benchmarks/cdvqa.py`.
- **Architectural Requirement**: Change-VQA is a composed workflow requiring TinyCD (for spatial change detection) + VLM (for textual change description).
- **Audit Findings**:
  - TinyCD executes genuinely on LEVIR-CD pairs.
  - The visual handoff gap has been resolved: `extract_changed_region_evidence` extracts deterministic change regions, applies $15\%$ spatial padding, and `ExecutionEngine` pipes the cropped $T_1$ visual patch directly into the VLM.
  - When the fine-tuned Qwen2.5-VL checkpoint is pending, the VLM stage returns honest structured `CHANGE_VQA_MODEL_NOT_READY` with zero hallucination, no PaliGemma, and no random weights.
  - Zero ground-truth leakage verified through regression tests.
- **Status**: **END-TO-END FUNCTIONAL — VLM CHECKPOINT PENDING**.

---

### 4.6 BigEarthNet.txt (Benchmark Split)
- **Scope**: Multi-modal land-cover classification and zero-shot VQA on Sentinel-1 SAR and Sentinel-2 MSI rasters.
- **Quarantine Policy**: The 8,000-pair Stage 1 materialization is reserved exclusively for model fine-tuning.
- **Benchmark Split**: Evaluation requires downloading the official held-out `BigEarthNet.txt` test split annotations once the fine-tuned Qwen checkpoint is exported.
- **Status**: **DATA READY / MODEL NOT READY**.

---

### 4.7 ISRO/SAC Target Data (Cartosat-2S & RISAT SAR)
- **Scope**: Multi-sensor spaceborne intelligence benchmark for Cartosat-2S high-resolution optical ($0.65$m GSD) and RISAT-1/1A C-band SAR ($1-3$m GSD) imagery.
- **Data Status**: Zero spaceborne mission imagery exists in the development environment. Mission data is proprietary and classified by ISRO.
- **Evaluator Status**: `ISROSACGenericEvaluator` is fully implemented as a generic, plug-and-play evaluation harness without hardcoded answers, designed for air-gapped evaluation by the competition technical jury.
- **Status**: **NOT AVAILABLE / AWAITING OFFICIAL DATA**.

---

## 5. Non-Interference & Integrity Certification

1. **Frozen Production Baseline Integrity**:
   - `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth` was neither retrained nor modified.
   - `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth` was neither retrained nor modified.
2. **Strict Non-Fabrication Guarantee**:
   - No benchmark scores were estimated or fabricated.
   - All statuses reflect the verified empirical state of the SatQuery codebase.
