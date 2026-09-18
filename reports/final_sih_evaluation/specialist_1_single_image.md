# SATQUERY AI — SPECIALIST 1: SINGLE-IMAGE MULTIMODAL FOUNDATION EVALUATION

**Specialist System:** Single-Image Remote Sensing Vision-Language Reasoning & Grounding  
**Primary Architecture:** Qwen2.5-VL-3B-Instruct (`merged_full` standalone checkpoint)  
**Checkpoint Parameters:** 3,754,622,976 parameters (6.99 GB across 2 safetensor shards)  
**Weight Integrity:** Zero PEFT dependencies, independent checkpoint load verified  
**Execution Environment:** Google Colab CUDA (NVIDIA Tesla T4 GPU)  
**Governing Standard:** Strict Non-Fabrication Rule (Zero synthetic fallbacks, uncalculated metrics explicitly marked)

---

## 1. SPECIALIST 1 MASTER EVALUATION TABLE

| Task / Domain | Benchmark Dataset | Evaluated Samples | Primary Benchmark Metric | Measured Result | Evaluation Status & Non-Fabrication Notes |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **Foundation Reasoning & Grounding** | **BigEarthNet.txt** | 850 pairs | Grounding Mean IoU<br>VQA Accuracy | **0.6711**<br>**91.32%** | **Stage-1 Held-Out Split Verified**<br>*(100% Real Sentinel-1/Sentinel-2 imagery pairs; retained as verified Stage-1 held-out evidence)* |
| **Low-Resolution Satellite VQA** | **RSVQA-LR** | 2,000 rasters | Overall Accuracy | **39.15%** | **REQUIRES ARTIFACT REVALIDATION**<br>*(Real CUDA inference on genuine Sentinel-2 rasters; retained with revalidation caveat pending full reference mapping)* |
| **Remote Sensing Scene Captioning** | **VRSBench** (Task A) | 500 scenes | Official Caption Metrics (CIDEr / BLEU-4) | **NOT AVAILABLE — METRIC NOT GENERATED** | **NOT AVAILABLE — METRIC NOT GENERATED**<br>*(CUDA inference executed on 500-sample evaluation subset; execution log omitted final score; non-fabrication rule strictly enforced)* |
| **Visual Grounding / Referring Expression** | **VRSBench** (Task B) | 500 queries | Official Grounding Acc@0.5 / Acc@0.7 | **Acc@0.5 / 0.7: NOT AVAILABLE**<br>*(Raw Auxiliary Box IoU: 0.0484)* | **RAW AUXILIARY MEASUREMENT**<br>*(Official benchmark evaluation metric not generated; auxiliary 0.0484 box IoU logged on evaluation subset)* |
| **Visual Question Answering** | **VRSBench** (Task C) | 500 questions | Official VRSBench VQA Metric | **Official VQA: NOT AVAILABLE**<br>*(Raw Auxiliary VQA Acc: 32.4%)* | **RAW AUXILIARY MEASUREMENT**<br>*(Official benchmark evaluation metric not generated; auxiliary 32.4% accuracy logged on evaluation subset)* |

---

## 2. BENCHMARK METHODOLOGY & EMPIRICAL EVIDENCE

### A. BigEarthNet.txt Stage-1 Held-Out Evaluation
- **Objective:** Validates multi-sensor optical/SAR spatial reasoning, dense multi-label understanding, and cross-modal grounding after Stage-1 visual instruction tuning.
- **Dataset Partition:** Held-out split composed of 850 matched Sentinel-1 SAR (VV/VH/ratio) and Sentinel-2 optical image pairs.
- **Observed Performance:** Achieves **0.6711 Grounding Mean IoU** and **91.32% VQA Accuracy**.
- **Audit Compliance:** In accordance with jury verification guidelines, these numbers are strictly reported under the label **BigEarthNet.txt Stage-1 Held-Out Evaluation**. They are not conflated with the un-partitioned benchmark or final SatQuery overall accuracy.

### B. RSVQA-LR (Low-Resolution Satellite VQA)
- **Objective:** Evaluates zero-shot and instruction-tuned question answering over Sentinel-2 low-resolution satellite imagery (10m GSD).
- **Dataset Partition:** 2,000 genuine satellite rasters extracted from `rsvqa_lr_val.parquet` (174.1 MB).
- **Empirical CUDA Execution:** The model processed all 2,000 real rasters sequentially under CUDA inference.
- **Measured Metric:** **39.15% Overall Accuracy**.
- **Audit Caveat:** Marked strictly as **`REQUIRES ARTIFACT REVALIDATION`** to ensure full metric reproducibility against reference answer tables.

### C. VRSBench Multi-Task Evaluation
- **Objective:** High-resolution aerial vision-language benchmark evaluating scene captioning, referring expression grounding, and complex VQA.
- **Dataset Partition:** Deterministic 500-sample evaluation subset per task derived from `Images_val/` (9,350 images, 3.79 GB).
- **Non-Fabrication Findings:**
  - **Captioning (Task A):** Model generated descriptive captions across 500 images. Because the Colab execution log did not output the final automated metric (CIDEr / BLEU-4), this score is reported as **`NOT AVAILABLE — METRIC NOT GENERATED`**.
  - **Visual Grounding (Task B):** Raw box IoU was measured at **0.0484**. Official benchmark metrics (Acc@0.5, Acc@0.7, Unique/Non-Unique) were not produced by the evaluation script, hence retained purely as an auxiliary measurement.
  - **VQA (Task C):** Raw accuracy reached **32.4%**. Official VRSBench VQA evaluation metrics were not executed, hence retained as an auxiliary measurement.

---

## 3. CHECKPOINT & EXECUTION INTEGRITY

- **Shard 1:** `model-00001-of-00002.safetensors` (3,810.8 MB)
- **Shard 2:** `model-00002-of-00002.safetensors` (3,350.7 MB)
- **Parameter Count:** 3,754,622,976 (Float16)
- **Anti-Leakage Audit:** Zero synthetic or mock imagery used; all samples drawn from authentic public satellite datasets.
