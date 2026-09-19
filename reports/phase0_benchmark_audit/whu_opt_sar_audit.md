# SATQUERY AI — PHASE 0 AUDIT REPORT: WHU-OPT-SAR

**Benchmark Name:** WHU-OPT-SAR (Cross-Modal Optical-SAR Land-Cover Classification)  
**Primary Reference:** Li et al., Wuhan University / LIESMARS (ISPRS Journal, 2022)  
**Model Audited:** CMAF (Dual ResNet Encoders + Cross-Modal Attention Fusion)  
**Checkpoint Path:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
**Checkpoint SHA-256:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`  
**Parameters:** 19,755,144  
**Audit Classification:** **VERIFIED (100% REPRODUCIBLE)**

---

## 1. PROTOCOL & SENSOR INTEGRITY
- **Dataset Partition:** Official WHU-OPT-SAR Test Split (`data/official_whu_opt_sar/test/`).
- **Sample Quota:** 4,950 tiles of $256 \times 256$ pixels across 15 parent geographic scenes.
- **Labeled Pixels:** Exactly 308,687,656 valid pixels (15,715,544 border/background pixels labeled 255 masked out).
- **Modality Contract:** Verified genuine multi-sensor ingestion:
  - Optical Encoder receives 3-channel optical MSI (RGB/B8).
  - SAR Encoder receives 2-channel Sentinel-1 SAR (VV/VH backscatter).
  - Zero modality substitution or synthetic channel synthesis.
- **Anti-Leakage Audit:** PASS. Ground-truth segmentation masks read strictly post-forward-pass for multi-class confusion accumulation.

---

## 2. INDEPENDENT RECOMPUTATION
Independent recalculation executed locally on Apple Silicon MPS matched reported numbers with **0.00% difference**:
- **Overall Accuracy (OA):** **71.71%** (0.717099) vs. Ref 71.71% (Diff: **0.00%**)
- **mean IoU (mIoU):** **35.08%** (0.350793) vs. Ref 35.08% (Diff: **0.00%**)
- **Weighted F1:** **74.18%** (0.741756) vs. Ref 74.18% (Diff: **0.00%**)
- **Macro F1:** **46.62%** (0.466209) vs. Ref 46.62% (Diff: **0.00%**)
- **Weighted IoU:** **60.38%** | **Weighted Precision:** **77.69%**
- **Macro Precision:** **46.58%** | **Macro Recall:** **52.48%**
- **Per-Class Breakdown:**
  - Class 0 (Background): 0.00% IoU (1,973 support)
  - Class 1 (Farmland): 59.20% IoU, 74.37% F1 (111,315,326 support)
  - Class 2 (City): 44.57% IoU, 61.66% F1 (13,218,868 support)
  - Class 3 (Village): 33.32% IoU, 49.99% F1 (16,869,004 support)
  - Class 4 (Water): 50.71% IoU, 67.29% F1 (40,822,113 support)
  - Class 5 (Forest): 73.69% IoU, 84.85% F1 (118,860,724 support)
  - Class 6 (Road): 11.68% IoU, 20.91% F1 (3,083,943 support)
  - Class 7 (Others): 7.46% IoU, 13.89% F1 (4,515,705 support)
- **Runtime Performance:** Total runtime 284.87 seconds; throughput 17.38 tiles/second (end-to-end), 28.50 tiles/second (forward-only).
- **Verdict:** **VERIFIED**. Robust cross-modal segmentation specialist.
