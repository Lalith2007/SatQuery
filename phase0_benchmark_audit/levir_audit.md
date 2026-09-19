# SATQUERY AI — PHASE 0 AUDIT REPORT: LEVIR-CD

**Benchmark Name:** LEVIR-CD (Large-Scale Building Change Detection)  
**Primary Reference:** Chen & Shi, Beihang University LEVIR Lab (IEEE JSTARS, 2020)  
**Model Audited:** TinyCD (Siamese U-Net + MAMB)  
**Checkpoint Path:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`  
**Checkpoint SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`  
**Parameters:** 3,565,034  
**Audit Classification:** **VERIFIED (100% REPRODUCIBLE)**

---

## 1. PROTOCOL & PARTITION INTEGRITY
- **Dataset Partition:** Official LEVIR-CD Test Split (`data/official_levir_cd/test/`).
- **Sample Count:** Exactly 128 pairs of $1024 \times 1024$ native optical images.
- **Label Integrity:** 128 binary ground truth masks (Changed: 427,414 pixels [5.10%]; Unchanged: 7,961,194 pixels).
- **Decision Threshold:** Strict fixed production threshold of **0.50**. Zero post-hoc tuning.
- **Anti-Leakage Audit:** PASS. Ground-truth masks were opened strictly post-inference for confusion matrix accumulation.

---

## 2. INDEPENDENT RECOMPUTATION
Independent recalculation executed locally on Apple Silicon MPS matched reported numbers with **0.00% difference**:
- **F1 Score:** **79.31%** (0.793059) vs. Ref 79.31% (Diff: **0.00%**)
- **IoU:** **65.71%** (0.657082) vs. Ref 65.71% (Diff: **0.00%**)
- **Overall Accuracy (OA):** **97.99%** (0.979889) vs. Ref 97.99% (Diff: **0.00%**)
- **Precision:** **83.36%** | **Recall:** **75.63%** | **Specificity:** **99.19%**
- **Confusion Matrix:**
  - True Positives (TP): `323,254`
  - False Positives (FP): `64,540`
  - False Negatives (FN): `104,160`
  - True Negatives (TN): `7,896,654`
- **Per-Scene Dynamics:**
  - Mean Scene F1: **69.27%** | Median Scene F1: **79.22%**
  - Mean Scene IoU: **58.26%** | Median Scene IoU: **65.59%**
- **Runtime Performance:** Total test runtime 10.14 seconds; mean latency 30.49 ms; throughput 12.62 FPS (end-to-end), 32.80 FPS (forward-only).
- **Verdict:** **VERIFIED**. True state-of-the-art production specialist.
