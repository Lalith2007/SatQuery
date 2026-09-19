# RSVQA-LR Public Benchmark Readiness & Provenance Report

**Benchmark:** RSVQA Low Resolution (Sentinel-2 10m GSD)  
**Evaluator:** `RSVQAEvaluator` (`evaluation/benchmarks/rsvqa.py`)  
**Data Provenance:** Sylvain Lobry, Diego Marcos, Devis Tuia / Zenodo (`10.5281/zenodo.6344334`)  
**Dataset Split:** Held-Out Validation Split (`2,000` records in `datasets/evaluation/rsvqa/manifest.jsonl`)  
**Evaluation Status:** **EVALUATOR HARNESS VERIFIED — MODEL PENDING**  

---

## 1. Non-Fabrication Policy Compliance

SatQuery AI strictly enforces zero score fabrication. Because fine-tuning of Qwen2.5-VL-3B-Instruct on the curated mixture is actively executing on Google Colab, official model inference on RSVQA-LR has not yet taken place.

- **Model Target:** `Qwen2.5-VL-3B-Instruct (Vision-Language Adapter)`
- **Model Status:** `NOT_READY (PENDING QWEN TRAINING MERGE)`
- **Benchmark Accuracy Score:** **`NOT AVAILABLE`**
- **Proxy/Random Substitution:** **`ZERO`** (No simulated or synthetic scores permitted)

---

## 2. Dataset Provenance Profile

- **DATASET:** RSVQA-LR (Remote Sensing Visual Question Answering - Low Resolution)
- **SOURCE:** Sylvain Lobry, Diego Marcos, Devis Tuia (Zenodo Archive, DOI: `10.5281/zenodo.6344334`)
- **SPLIT:** `validation` (`rsvqa_lr_val.parquet`, 174,052,999 bytes)
- **RECORD COUNT:** 2,000 question-answer pairs
- **IMAGE COUNT:** 100 unique Sentinel-2 optical tiles ($256 \times 256$ pixels, 10m GSD)
- **QUESTION COUNT:** 869 unique questions (Presence: 1,002, Comparison: 998)
- **ROLE:** Benchmark evaluation partition (held-out from training; validation split per official source naming)
- **Quarantine Guarantee:** Zero overlap with model training splits.

---

## 3. Evaluation Harness Verification

The metric arithmetic pipeline was smoke tested on real RSVQA records:
- **Clean Text Normalization:** Operational
- **Number Parsing & Numerical Match:** Operational
- **Multi-Category Accuracy Accumulation:** Operational
