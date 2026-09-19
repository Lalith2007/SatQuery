# SATQUERY AI — PHASE 0 AUDIT REPORT: CDVQA

**Benchmark Name:** Change Detection Visual Question Answering (CDVQA)  
**Primary Reference:** Yuan et al., Wuhan University (IEEE TGRS, DOI: 10.1109/TGRS.2022.3168127)  
**Model Audited:** TinyCD (Localization) + Qwen2.5-VL-3B Pipeline  
**Reported Metrics:** BLEU-4 = 0.285, ROUGE-L = 0.482 across 128 samples  
**Audit Classification:** **INVALID AS OFFICIAL CDVQA / REQUIRES REVALIDATION AS PIPELINE SUBSET**

---

## 1. DATASET PARTITION AUDIT: OFFICIAL VS. SUBSET
- **Dataset Evaluated:** 128 scene pairs derived from `data/official_levir_cd/test/`.
- **Official CDVQA Benchmark Alignment:**
  - The official CDVQA dataset (Yuan et al.) contains human-annotated multi-turn QA pairs across change existence, count, and descriptive semantics (`Test_questions.json`, `Test_answers.json`).
  - The 128 samples evaluated in the current run were **NOT** the official CDVQA test questions/answers.
  - Instead, they were the 128 scenes of LEVIR-CD paired with an ad-hoc query string:
    `"Describe what structural and land-cover changes occurred in the highlighted region between the two dates."`
- **Verdict:** This evaluation represents an ad-hoc pipeline demonstration subset, NOT the official CDVQA benchmark.

---

## 2. MODEL INPUT & ANTI-LEAKAGE AUDIT
- **Visual Evidence Contract:**
  - Image 1: `BEFORE` (T0 native optical crop)
  - Image 2: `AFTER` (T1 native optical crop)
  - Image 3: `WHERE_CHANGE_OCCURRED` (TinyCD predicted change overlay)
- **Ground-Truth Leakage Check:** **PASS**.
  - Bounding boxes and overlay masks were derived strictly from TinyCD's predicted probability map.
  - Zero ground-truth change masks were read during image handoff assembly.

---

## 3. CRITICAL FLAW: SYNTHETIC GROUND-TRUTH REFERENCE TEXT
- **Forensic Code Inspection:**
  - In `evaluation/colab_qwen_final_benchmark_runner.py` line 512:
    ```python
    gt_desc = (
        "New residential buildings and infrastructure constructed in the cleared agricultural area."
        if rec.get("has_change")
        else "No significant change detected."
    )
    ground_truths.append({"answer": gt_desc, "description": gt_desc})
    ```
  - **Critical Flaw:** The ground truth used to compute BLEU-4 (0.285) and ROUGE-L (0.482) was **NOT** human-authored benchmark reference annotations!
  - It was a **hardcoded synthetic template string**!
  - Every predicted description was compared against that single synthetic sentence!
- **Metric Implementation:**
  - Evaluator used custom token matching functions in `evaluation/benchmarks/cdvqa.py`, not the official benchmark evaluation script.
- **Verdict:** The reported scores (BLEU-4 0.285, ROUGE-L 0.482) are **METHODOLOGICALLY INVALID** as official benchmark metrics because they measure lexical similarity against a hardcoded template.
