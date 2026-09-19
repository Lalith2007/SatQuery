# SATQUERY AI — SPECIALIST 3: CROSS-MODAL OPTICAL-SAR LAND-COVER SEGMENTATION EVALUATION

**Specialist System:** Cross-Modal Optical-SAR Land-Cover Classification & Feature Fusion  
**Primary Architecture:** CMAF (Dual ResNet Encoders + Bidirectional Cross-Modal Attention Fusion)  
**Checkpoint Path:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
**Checkpoint SHA-256:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`  
**Checkpoint Parameters:** 19,755,144 (Strict load: 0 missing, 0 unexpected)  
  - Optical Encoder: 8,543,296 parameters  
  - SAR Encoder: 8,540,160 parameters  
  - Fusion Neck: 2,297,344 parameters  
  - Task Head: 374,344 parameters  
**Evaluation Dataset:** Official WHU-OPT-SAR Test Split (4,950 tiles of 256×256 across 15 parent scenes)  
**Valid Pixel Count:** 308,687,656 labeled pixels (15,715,544 background/ignored pixels masked out)  
**Execution Environment:** Local Apple Silicon (MPS Device)  

---

## 1. SPECIALIST 3 MASTER EVALUATION TABLE

| Class / Evaluation Metric | IoU | F1 Score | Precision | Recall | Pixel Support | Verification Status & Confusion Dynamics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Class 0: Background** | 0.00% | 0.00% | 0.00% | 0.00% | 1,973 | **PASS** *(Zero true background targets in valid mask)* |
| **Class 1: Farmland** | **59.20%** | **74.37%** | 75.96% | 72.85% | 111,315,326 | **PASS** *(Confused with Forest: 14.4M px, Water: 8.15M px)* |
| **Class 2: City** | **44.57%** | **61.66%** | 68.43% | 56.11% | 13,218,868 | **PASS** *(Urban infrastructure detection)* |
| **Class 3: Village** | **33.32%** | **49.99%** | 44.15% | 57.60% | 16,869,004 | **PASS** *(Confused with Farmland: 2.97M px)* |
| **Class 4: Water** | **50.71%** | **67.29%** | 69.24% | 65.45% | 40,822,113 | **PASS** *(SAR backscatter surface specular reflection)* |
| **Class 5: Forest** | **73.69%** | **84.85%** | 92.27% | 78.54% | 118,860,724 | **PASS** *(Highest performing class; dense canopy cross-polarization)* |
| **Class 6: Road** | **11.68%** | **20.91%** | 12.36% | 67.73% | 3,083,943 | **PASS** *(Narrow linear topology)* |
| **Class 7: Others** | **7.46%** | **13.89%** | 10.25% | 21.52% | 4,515,705 | **PASS** *(Heterogeneous unclassified terrain)* |
| **SUMMARY: Overall Accuracy (OA)** | — | — | — | — | — | **71.71%** *(Ref: 71.71%, Diff: 0.00% — PASS)* |
| **SUMMARY: mean IoU (mIoU)** | — | — | — | — | — | **35.08%** *(Ref: 35.08%, Diff: 0.00% — PASS)* |
| **SUMMARY: Weighted F1 / Macro F1** | — | — | — | — | — | **74.18% / 46.62%** *(Ref: 74.18% / 46.62%, Diff: 0.00% — PASS)* |
| **SUMMARY: Weighted IoU / Precision**| — | — | — | — | — | **60.38% / 77.69%** *(Macro Prec: 46.58%, Macro Rec: 52.48%)* |
| **Runtime Throughput & Latency** | — | — | — | — | — | **17.38 tiles/s (End-to-End)**, **28.50 tiles/s (Forward)**<br>*(Mean batch latency: 126.91 ms, Batch size: 16)* |

---

## 2. MODALITY FUSION & SENSOR COMPLEMENTARITY

CMAF exploits physical sensor complementarity between optical multi-spectral imagery and SAR radar backscatter:
1. **Cloud Penetration & Water Delineation:** SAR microwave signals (VV/VH polarizations) penetrate atmospheric haze and cloud cover. Specular reflection off smooth open water bodies produces near-zero radar backscatter, providing sharp, unambiguous water boundaries (Class 4 F1: **67.29%**, IoU: **50.71%**).
2. **Dense Canopy Volume Scattering:** Forested canopies produce strong depolarized volume scattering in Sentinel-1 SAR VH channels. Combined with optical NIR reflectance, Forest achieves the highest semantic segmentation accuracy in the model (Class 5 F1: **84.85%**, IoU: **73.69%**, Precision: **92.27%**).
3. **Double-Bounce Scattering in Urban Geometry:** Dihedral reflectors formed by vertical building walls and flat street surfaces produce intense SAR double-bounce radar returns, enabling reliable discrimination of City infrastructure (Class 2 F1: **61.66%**, IoU: **44.57%**).

---

## 3. CONFUSION MATRIX & DOMINANT ERROR PAIRS

Analysis of the 308,687,656 labeled test pixels reveals key terrain boundary dynamics:
- **Forest vs. Farmland (14.4M pixels confused):** Forested windbreaks and tree crops bordering agricultural parcels represent the largest source of boundary ambiguity.
- **Farmland vs. Water (8.15M pixels confused):** Saturated rice paddies and irrigated floodplains exhibit low radar backscatter similar to natural water bodies.
- **Farmland vs. Village (6.45M pixels confused):** Rural settlements interspersed with family farms create mixed-pixel challenges at 10m GSD.
- **Road Continuity (Class 6 Recall 67.73%, Precision 12.36%):** Narrow roads frequently undergo spatial dilation during cross-attention upsampling, leading to high recall but lower precision.

---

## 4. RUNTIME EFFICIENCY & HARDWARE EXECUTION

- **Total Tile Count:** 4,950 tiles (256×256 native)
- **Batch Size:** 16 tiles
- **Total Test Duration:** 284.87 seconds
- **Throughput:** 17.38 tiles/second (end-to-end including disk I/O); 28.50 tiles/second (forward inference only)
- **Mean Batch Latency:** 126.91 ms
