# VRSBench Public Benchmark Readiness & Provenance Report

**Benchmark:** VRSBench Multi-Task Suite (High-Resolution Optical)  
**Evaluator:** `VRSBenchEvaluator` (`evaluation/benchmarks/vrsbench.py`)  
**Data Provenance:** LIESMARS, Wuhan University (Ling et al., 2024)  
**Dataset Split:** Representative Test Split (`18` records in `datasets/evaluation/vrsbench/manifest.jsonl`)  
**Evaluation Status:** **EVALUATOR HARNESS VERIFIED — MODEL PENDING**  

---

## 1. Non-Fabrication Policy Compliance

- **Model Target:** `Qwen2.5-VL-3B-Instruct (Vision-Language Adapter)`
- **Model Status:** `NOT_READY (PENDING QWEN TRAINING MERGE)`
- **Benchmark VQA Score:** **`NOT AVAILABLE`**
- **Benchmark Grounding mIoU Score:** **`NOT AVAILABLE`**
- **Benchmark Captioning Score:** **`NOT AVAILABLE`**
- **Score Fabrication Policy:** Zero synthetic or proxy scores permitted prior to authentic model forward pass.

---

## 2. Dataset Provenance & Sample Breakdown

- **DATASET:** VRSBench (Visual Question Answering & Visual Grounding for Remote Sensing)
- **SOURCE:** Wuhan University / LIESMARS (Ling et al., 2024)
- **SPLIT:** `representative_test_partition` (`datasets/evaluation/vrsbench/manifest.jsonl`)
- **TOTAL ACQUIRED:** 18 records (13 visual grounding with bounding boxes + 5 VQA question-answer pairs)
- **IMAGE COUNT:** 1 high-resolution optical scene (`images/vrsbench_fig_example.png`, 0.5m-2.0m GSD)
- **HARNESS SMOKE SUBSET:** 1 record tested for bounding box IoU computation
- **EVALUATION SUBSET:** 18 materialized records
- **REASON FOR 18 vs 150:** Earlier draft documents noted 150 as a planned target sample quota. Exactly 18 records (13 referring grounding + 5 VQA) were materialized in `data/benchmark_samples/vrsbench/` and partitioned into `datasets/evaluation/vrsbench/manifest.jsonl`. No VRSBench score is claimed or computed before genuine Qwen inference.
- **ROLE:** Benchmark evaluation partition (quarantined test partition; harness verified)

---

## 3. Evaluation Harness Verification

- **Bounding Box IoU:** Verified mathematically (identical box IoU = 1.0)
- **Token F1 & Exact Match:** Verified
- **Multi-Turn VQA Accumulation:** Operational
