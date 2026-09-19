# SATQUERY AI — PHASE 0 AUDIT REPORT: RSVQA-LR

**Benchmark Name:** RSVQA Low-Resolution (RSVQA-LR)  
**Primary Reference:** Sylvain Lobry, Diego Marcos, Devis Tuia (Zenodo DOI: 10.5281/zenodo.6344334)  
**Model Audited:** Qwen2.5-VL-3B-Instruct (`merged_full` standalone checkpoint)  
**Reported Metric:** Overall Accuracy = 39.15% across 2,000 samples  
**Audit Classification:** **REQUIRES REVALIDATION**

---

## 1. DATASET PROVENANCE & ON-DISK INTEGRITY
- **Source File:** `data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet` (174.1 MB).
- **Split Identified:** Official Validation partition.
- **Record Count:** Exactly 2,000 records, each containing embedded Sentinel-2 image bytes, a question string, and a reference answer.
- **Answer Distribution:**
  - Binary Answers: `no` (690, 34.5%), `yes` (678, 33.9%) $\rightarrow$ Combined: 68.4% of dataset.
  - Counting Answers: `0` (140), `1` (36), `2` (24), `4` (18), `6` (18), `3` (18), etc.
  - Categorical Answers: `rural` (15), `urban` (14).
- **Contamination Check:** PASS. The validation partition contains zero overlap with BigEarthNet Stage-1 training tiles.

---

## 2. INFERENCE & EVALUATION PROTOCOL FORENSICS
- **Inference Verification:** Confirmed genuine CUDA inference. The Colab log indicates all 2,000 samples were processed at ~1.4 samples/sec over 23.5 minutes.
- **Model Prompting Protocol:**
  - In `evaluation/colab_qwen_final_benchmark_runner.py` line 262:
    `pred = generate_qwen_response(model, processor, [img], r["question"], max_new_tokens=32)`
  - **Deficiency:** The prompt passed ONLY the raw question (e.g. `"What is the number of commercial buildings?"`) without any instruction constraining output format (e.g. `"Answer in one word or number"`).
  - Consequently, Qwen generated open-ended natural language sentences (e.g., `"There are no visible commercial buildings in this image."`).
- **Answer Matching & Normalization Heuristic:**
  - Line 268:
    `if gt in p or p in gt or gt == p: correct_count += 1`
  - **Critical Flaw:** Substring containment was substituted for official RSVQA exact-match vocabulary evaluation.
  - If ground truth is `"no"` and model says `"There are no buildings"`, it counts as correct.
  - If ground truth is `"0"` and model says `"zero"` or `"none"`, it counts as incorrect because `"0"` is not a substring of `"zero"`.
  - If ground truth is `"between 10 and 20"` and model says `"15"`, it fails.

---

## 3. PREDICTION ARTIFACT AUDIT
- **Prediction File On Disk:** **DOES NOT EXIST**.
- In `colab_qwen_final_benchmark_runner.py`, the individual 2,000 predictions were accumulated in RAM and used to compute summary statistics, but were never serialized to `predictions.json`.
- **Reproducibility Verdict:** Because individual predictions were not saved, independent recomputation directly from disk artifacts cannot be performed without rerunning inference.

---

## 4. AUDIT FINDINGS & RECOMMENDATIONS
1. **Is 39.15% Reproducible?** The summary metric is logged in `evaluation_report.json`, but raw prediction strings are missing from disk.
2. **Is Answer Normalization Correct?** NO. Substring matching is non-standard and distorts numerical count scoring.
3. **Pre-Stage-2 Requirement:**
   - Implement strict logit-biasing or prompt-constrained generation (`"Answer with a single word or number"`).
   - Serialize all 2,000 predictions to `predictions.json`.
   - Implement the official RSVQA 744-class vocabulary evaluation.
