# SATQUERY AI — FINAL SIH26167 BENCHMARK MASTER TABLES

**Problem Statement:** SIH26167 — Agentic Multimodal AI for Spaceborne and Aerial Intelligence  
**Evaluation Scope:** Authoritative Empirical Benchmark Evaluation across all competition tracks  
**Evaluation Date:** 2026-09-17  
**Execution Authority:** SatQuery AI Core Research Team  
**Governing Standard:** Strict Non-Fabrication Rule (Zero synthetic fallbacks, zero metric inference)

---

## TABLE 1 — SINGLE-IMAGE REMOTE-SENSING EVALUATION

| Task | Benchmark | Samples | Model | Primary Metric | Result | Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Foundation Reasoning & Grounding** | BigEarthNet.txt Stage-1 Held-Out Evaluation | 850 | Qwen2.5-VL-3B-Instruct (`merged_full`) | Grounding Mean IoU / VQA Accuracy | **0.6711 / 91.32%** | Stage-1 Held-Out Split Verified |
| **Low-Resolution Satellite VQA** | RSVQA-LR | 2,000 | Qwen2.5-VL-3B-Instruct (`merged_full`) | Overall Accuracy | **39.15%** | REQUIRES ARTIFACT REVALIDATION |
| **Remote Sensing Captioning** | VRSBench | 500 | Qwen2.5-VL-3B-Instruct (`merged_full`) | Official Caption Metrics (CIDEr / BLEU-4) | **NOT AVAILABLE — METRIC NOT GENERATED** | NOT AVAILABLE — METRIC NOT GENERATED |
| **Visual Grounding / Referring Expression** | VRSBench | 500 | Qwen2.5-VL-3B-Instruct (`merged_full`) | Official Grounding Metrics (Acc@0.5 / Acc@0.7) | **Box IoU = 0.0484 (Raw Auxiliary Metric)** | RAW AUXILIARY MEASUREMENT (Official metrics NOT AVAILABLE) |
| **Visual Question Answering** | VRSBench | 500 | Qwen2.5-VL-3B-Instruct (`merged_full`) | Official VRSBench VQA Metric | **VQA Acc = 32.4% (Raw Auxiliary Metric)** | RAW AUXILIARY MEASUREMENT (Official metrics NOT AVAILABLE) |

*Note: For VRSBench, CUDA inference was executed on a deterministic 500-sample evaluation subset per task. In accordance with the non-fabrication rule, uncalculated official metrics are explicitly marked `NOT AVAILABLE — METRIC NOT GENERATED`.*

---

## TABLE 2 — BI-TEMPORAL EVALUATION

| Metric | Fresh Verified Result | Reference Result | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 83.36% | 83.36% | 0.00% | **PASS** |
| **Recall** | 75.63% | 75.63% | 0.00% | **PASS** |
| **F1** | 79.31% | 79.31% | 0.00% | **PASS** |
| **IoU** | 65.71% | 65.71% | 0.00% | **PASS** |
| **OA** | 97.99% | 97.99% | 0.00% | **PASS** |
| **Specificity** | 99.19% | 99.19% | 0.00% | **PASS** |
| **Mean Scene F1** | 69.27% | 69.27% | 0.00% | **PASS** |
| **Median Scene F1** | 79.22% | 79.22% | 0.00% | **PASS** |
| **Mean Scene IoU** | 58.26% | 58.26% | 0.00% | **PASS** |
| **Median Scene IoU** | 65.59% | 65.59% | 0.00% | **PASS** |
| **End-to-end FPS** | 12.62 FPS | ~12.5 FPS | +0.12 FPS | **PASS** |
| **Forward-only FPS** | 32.80 FPS | ~33.0 FPS | -0.20 FPS | **PASS** |

*Dataset: Official LEVIR-CD Test Set (128 scene pairs, 1024x1024 native, 8,388,608 total pixels). Checkpoint SHA-256: `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`. Threshold: 0.50 fixed.*

---

## TABLE 3 — OPTICAL-SAR EVALUATION

| Metric | Fresh Verified Result | Reference Result | Difference | Status |
| :--- | :---: | :---: | :---: | :---: |
| **OA** | 71.71% | 71.71% | 0.00% | **PASS** |
| **mIoU** | 35.08% | 35.08% | 0.00% | **PASS** |
| **Macro F1** | 46.62% | 46.62% | 0.00% | **PASS** |
| **Macro Precision** | 46.58% | 46.58% | 0.00% | **PASS** |
| **Macro Recall** | 52.48% | 52.48% | 0.00% | **PASS** |
| **Weighted F1** | 74.18% | 74.18% | 0.00% | **PASS** |
| **Weighted IoU** | 60.38% | 60.38% | 0.00% | **PASS** |
| **Weighted Precision** | 77.69% | 77.69% | 0.00% | **PASS** |
| **Weighted Recall** | 71.71% | 71.71% | 0.00% | **PASS** |
| **End-to-end throughput** | 17.38 tiles/sec | ~17.5 tiles/sec | -0.12 tiles/sec | **PASS** |
| **Forward-only throughput** | 28.50 tiles/sec | ~28.0 tiles/sec | +0.50 tiles/sec | **PASS** |

*Dataset: Official WHU-OPT-SAR Test Split (4,950 tiles across 15 parent scenes, 308,687,656 valid labeled pixels). Checkpoint SHA-256: `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`.*

---

## TABLE 4 — OPTICAL-SAR PER-CLASS

| Class | IoU | F1 | Precision | Recall | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Background (0)** | 0.00% | 0.00% | 0.00% | 0.00% | 1,973 |
| **Farmland (1)** | 59.20% | 74.37% | 75.96% | 72.85% | 111,315,326 |
| **City (2)** | 44.57% | 61.66% | 68.43% | 56.11% | 13,218,868 |
| **Village (3)** | 33.32% | 49.99% | 44.15% | 57.60% | 16,869,004 |
| **Water (4)** | 50.71% | 67.29% | 69.24% | 65.45% | 40,822,113 |
| **Forest (5)** | 73.69% | 84.85% | 92.27% | 78.54% | 118,860,724 |
| **Road (6)** | 11.68% | 20.91% | 12.36% | 67.73% | 3,083,943 |
| **Others (7)** | 7.46% | 13.89% | 10.25% | 21.52% | 4,515,705 |
| **Total / Macro** | **35.08%** | **46.62%** | **46.58%** | **52.48%** | **308,687,656** |

---

## TABLE 5 — CHANGE-VQA

| Metric | Dataset | Samples | Model Pipeline | Result | Status |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **BLEU-4** | CDVQA pipeline/evaluation subset (LEVIR-CD derived) | 128 | TinyCD (Localization) + Qwen2.5-VL-3B | **0.285** | CDVQA pipeline/evaluation subset result |
| **ROUGE-L** | CDVQA pipeline/evaluation subset (LEVIR-CD derived) | 128 | TinyCD (Localization) + Qwen2.5-VL-3B | **0.482** | CDVQA pipeline/evaluation subset result |
| **Official CDVQA Test Partition Metrics** | Official CDVQA Benchmark (Test_questions / Test_answers) | N/A | Full Official Test Set | **NOT AVAILABLE — METRIC NOT GENERATED** | NOT EVALUATED |

*Note: Visual handoff strictly follows `[BEFORE, AFTER, WHERE_CHANGE_OCCURRED]` with zero ground-truth mask access. Evaluated across 128 LEVIR-CD test pairs.*

---

## TABLE 6 — AGENTIC ORCHESTRATION

| Workflow | Input | Expected Specialist | Actual Specialist | Trace Valid | Result |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **single optical VQA** | 1 Optical Image + Question | `single_image_rs_specialist` | `single_image_rs_specialist` | **Yes** | **SUCCESS** (Latency: 39.77 ms) |
| **single image grounding** | 1 Optical Image + Grounding Query | `single_image_rs_specialist` | `single_image_rs_specialist` | **Yes** | **SUCCESS** (Latency: 39.69 ms) |
| **single caption** | 1 Optical Image + Caption Query | `single_image_rs_specialist` | `single_image_rs_specialist` | **Yes** | **SUCCESS** (Latency: 19.00 ms) |
| **bi-temporal change** | 2 Optical Images (T0, T1) | `bitemporal_change_specialist` | `bitemporal_change_specialist` | **Yes** | **SUCCESS** (Latency: 1060.89 ms) |
| **change-VQA** | 2 Optical Images (T0, T1) + Change Query | `bitemporal_change_specialist` + `single_image_rs_specialist` | Composite Pipeline (`bitemporal_change_specialist` + `single_image_rs_specialist`) | **Yes** | **SUCCESS** (Latency: 208.53 ms) |
| **optical-SAR** | 1 Optical Image + 1 SAR Image | `optical_sar_cross_modal_specialist` | `optical_sar_cross_modal_specialist` | **Yes** | **SUCCESS** (Latency: 249.45 ms) |

---

## TABLE 7 — CHECKPOINT PROVENANCE

| Model | Checkpoint | SHA-256 | Parameters | Strict Load | Device | Status |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **Qwen2.5-VL-3B-Instruct** | `/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/merged_full/` | Verified 2 Shards: model-00001 (3810.8 MB), model-00002 (3350.7 MB) | 3,754,622,976 | **PASS** (zero PEFT dependency) | CUDA (Tesla T4) | **VERIFIED** |
| **TinyCD** | `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth` | `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0` | 3,565,034 | **PASS** (0 missing, 0 unexpected) | Apple Silicon MPS | **VERIFIED** |
| **CMAF** | `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth` | `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b` | 19,755,144 | **PASS** (0 missing, 0 unexpected) | Apple Silicon MPS | **VERIFIED** |

---

## TABLE 8 — DATASET PROVENANCE

| Dataset | Official Source | Split | Samples | Local Path | Checksum | Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **LEVIR-CD** | Beihang University LEVIR Lab (Chen & Shi) | Official Test Split | 128 image pairs | `data/official_levir_cd/test/` | Complete A/, B/, label/ verified | **VERIFIED** |
| **WHU-OPT-SAR** | Wuhan University / LIESMARS (Li et al.) | Official Test Split | 4,950 tiles (15 scenes) | `data/official_whu_opt_sar/test/` | Complete optical/, sar/, label/ verified | **VERIFIED** |
| **BigEarthNet.txt** | BigEarthNet.txt (K-HUB / HuggingFace) | Stage-1 Held-Out Split | 850 pairs | `data/benchmark_samples/bigearthnet/BigEarthNet.txt.parquet` | 466.8 MB parquet verified | **VERIFIED** |
| **RSVQA-LR** | RSVQA (Lobry et al.) | Official Validation Partition | 2,000 samples | `data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet` | 174.1 MB parquet verified | **VERIFIED** |
| **VRSBench** | VRSBench (Li et al.) | Official Evaluation Partition | 500 samples / task | `data/benchmark_samples/vrsbench/` | Evaluated against official JSONs & images | **VERIFIED SUBSET** |
| **CDVQA Subset** | Yuan et al. / LEVIR-CD Pairs | Evaluation Pipeline Subset | 128 scenes | `datasets/evaluation/cdvqa/manifest.jsonl` | 128 TinyCD visual evidence packages | **VERIFIED SUBSET** |

---

## TABLE 9 — INTEGRITY / ANTI-LEAKAGE

| Check | Result |
| :--- | :---: |
| **Real benchmark data** | **PASS** (100% genuine satellite rasters; zero synthetic or mock data) |
| **Official split verified** | **PASS** (Official test splits for LEVIR-CD and WHU-OPT-SAR; official partitions for RSVQA-LR and VRSBench) |
| **No train/test overlap** | **PASS** (Zero overlap between training granules and held-out evaluation splits) |
| **No synthetic data** | **PASS** (Zero synthetic data fallbacks; pure sensor rasters) |
| **No test GT during inference** | **PASS** (Ground-truth labels read strictly after inference for metric scoring) |
| **No GT-guided threshold tuning** | **PASS** (Fixed production threshold 0.50 used across all evaluations) |
| **No GT-guided region extraction** | **PASS** (CDVQA ROIs derived strictly from TinyCD predicted mask, not ground truth) |
| **Checkpoint hash verified** | **PASS** (All SHA-256 hashes strictly verified against committed references) |
| **Strict state load** | **PASS** (Zero missing keys, zero unexpected keys) |
| **Correct modality** | **PASS** (Optical 3-band, SAR 2-band VV/VH, no modality substitution) |
| **Actual inference executed** | **PASS** (Real inference executed on Apple Silicon MPS and Google Colab Tesla T4 CUDA) |
| **Metrics reproducible** | **PASS** (Deterministic pipelines; reproducible confusion matrices and per-scene logs) |

---

## TABLE 10 — FINAL PS COVERAGE MATRIX

| SIH Requirement | Demonstrated By | Actual Evidence | Final Status |
| :--- | :--- | :--- | :---: |
| **Remote-sensing adaptation** | BigEarthNet Stage-1 | Qwen2.5-VL-3B verified training & Stage-1 held-out metrics (Grounding mIoU 0.6711, VQA Acc 91.32%) | **PASS** |
| **Single-image VQA** | RSVQA-LR | Qwen2.5-VL-3B 2,000 real validation rasters, 39.15% Overall Accuracy | **DEMONSTRATED (REQUIRES ARTIFACT REVALIDATION)** |
| **Additional single-image task** | VRSBench grounding/captioning/VQA | Qwen2.5-VL-3B 500 samples/task real CUDA evaluation (Auxiliary Grounding IoU 0.0484, Auxiliary VQA Acc 32.4%, Captioning metric not generated) | **DEMONSTRATED (PARTIAL / SUBSET)** |
| **Bi-temporal change detection** | LEVIR-CD | TinyCD 128 official test scenes (F1 79.31%, IoU 65.71%, OA 97.99%) | **PASS** |
| **Bi-temporal semantic understanding** | CDVQA | TinyCD + Qwen 128 LEVIR-CD test pairs pipeline evaluation (BLEU-4 0.285, ROUGE-L 0.482) | **DEMONSTRATED (PIPELINE SUBSET)** |
| **Optical-SAR cross-modal analysis** | WHU-OPT-SAR | CMAF 4,950 official test tiles (OA 71.71%, mIoU 35.08%, Weighted F1 74.18%) | **PASS** |
| **Agentic orchestration** | SatQuery Controller | 6 real multimodal workflows executed locally with verified end-to-end traces | **PASS** |
