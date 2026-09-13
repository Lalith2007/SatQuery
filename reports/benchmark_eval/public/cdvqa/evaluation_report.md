# CDVQA Public Benchmark Readiness & Composed Workflow Report

**Benchmark:** CDVQA (Change Detection Visual Question Answering)  
**Evaluator:** `CDVQAEvaluator` (`evaluation/benchmarks/cdvqa.py`)  
**Data Provenance:** Beihang University LEVIR Lab / Yuan et al.  
**Dataset Split:** Official LEVIR-CD Held-Out Test Set (`128` scenes in `datasets/evaluation/cdvqa/manifest.jsonl`)  
**Evaluation Status:** **CDVQA: PIPELINE VERIFIED — QWEN SEMANTIC EVALUATION PENDING**  

---

## 1. Non-Fabrication Policy Compliance

- **Composed Workflow Status:** `PIPELINE VERIFIED — QWEN SEMANTIC EVALUATION PENDING`
- **TinyCD Spatial Change Detector:** `VERIFIED & FROZEN (79.31% F1, 65.71% IoU, 97.99% OA)`
- **Qwen2.5-VL-3B Semantic Change Interpreter:** `NOT_READY (PENDING QWEN TRAINING MERGE)`
- **CDVQA Benchmark Score:** **`NOT AVAILABLE`**
- **Proxy/Random Substitution:** **`ZERO`** (No synthetic or simulated responses allowed)

---

## 2. Production Composed Pipeline Verification

The CDVQA workflow strictly adheres to the authentic 3-image handoff:
1. **Stage 1 (Change Detection):** Frozen TinyCD processes T0 (Before) and T1 (After) to generate predicted binary change mask.
2. **Region Extraction:** Deterministic connected-component extraction identifies primary change ROI bounding box. Zero ground-truth mask access.
3. **Visual Package Preparation:** Crops T0 and T1 to change bounding box, generates full-scene change overlay, and packages 3 PIL images:
   - `[Image 1: BEFORE / T0_crop]`
   - `[Image 2: AFTER / T1_crop]`
   - `[Image 3: WHERE_CHANGE_OCCURRED / change_overlay]`
4. **Stage 2 (VLM Interpretation):** Dispatches 3 images and structured prompt to `run_change_vqa(...)`. Returns honest `CHANGE_VQA_MODEL_NOT_READY` pending trained weights.
