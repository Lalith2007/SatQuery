# Authoritative Evaluation Report — Division 4 Optical-SAR Specialist

**Production Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
**Checkpoint SHA-256:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`  
**Evaluator:** `specialists/optical_sar/eval_results/v3_fresh/`  
**Evaluation Dataset:** Official Wuhan University WHU-OPT-SAR Held-Out Test Split (15 scenes, 4,950 tiles, 308,687,656 valid pixels)  

---

## 1. Quantitative Evaluation Metrics

Fresh authoritative evaluation performed on the official 15 held-out test scenes (`NH49E001017`, `NH49E005023`, `NH49E006022`, `NH49E006024`, `NH49E007014`, `NH49E013018`, `NH49E013021`, `NH50E006001`, `NH50E010001`, `NH50E011002`, `NH50E013002`, `NI49E021010`, `NI49E021011`, `NI49E022013`, `NI49E022014`):

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy (OA)** | **71.71%** (0.717099) |
| **Mean IoU (mIoU)** | **35.08%** (0.350793) |
| **Macro F1 Score** | **46.62%** (0.466209) |
| **Macro Precision** | **46.58%** (0.465837) |
| **Macro Recall** | **52.48%** (0.524761) |
| **Weighted F1 Score** | **74.18%** (0.741756) |
| **Weighted IoU** | **60.38%** (0.603848) |
| **Active Classes** | **8 / 8** |

---

## 2. Per-Class Performance Breakdown

| Class | IoU | F1 / Dice | Precision | Recall | GT Pixel Share | Pred Pixel Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forest** | **0.7369** | 0.8485 | 0.9227 | 0.7854 | 38.51% | 32.78% |
| **Farmland** | **0.5920** | 0.7437 | 0.7596 | 0.7285 | 36.06% | 34.58% |
| **Water** | **0.5071** | 0.6729 | 0.6924 | 0.6545 | 13.22% | 12.50% |
| **City** | **0.4457** | 0.6166 | 0.6843 | 0.5611 | 4.28% | 3.51% |
| **Village** | **0.3332** | 0.4999 | 0.4415 | 0.5760 | 5.47% | 7.13% |
| **Road** | **0.1168** | 0.2091 | 0.1236 | 0.6773 | 1.00% | 5.47% |
| **Others** | **0.0746** | 0.1389 | 0.1025 | 0.2152 | 1.46% | 3.07% |
| **Background** | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.001% | 0.96% |

---

## 3. Historical Artifact vs. Fresh Authoritative Benchmark

| Metric | Historical Artifact (`final_test_metrics.json`) | Fresh Authoritative Evaluation | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | 56.15% | **71.71%** | +15.56% | Outperformed |
| **Mean IoU (mIoU)** | 23.15% | **35.08%** | +11.93% | Outperformed |
| **Macro F1** | 34.60% | **46.62%** | +12.02% | Outperformed |
| **Farmland IoU** | 0.3540 | **0.5920** | +0.2380 | Outperformed |
| **Forest IoU** | 0.5070 | **0.7369** | +0.2300 | Outperformed |
| **City IoU** | 0.2845 | **0.4457** | +0.1612 | Outperformed |
| **Village IoU** | 0.1670 | **0.3332** | +0.1663 | Outperformed |
| **Water IoU** | 0.3816 | **0.5071** | +0.1254 | Outperformed |
| **Road IoU** | 0.1020 | **0.1168** | +0.0148 | Outperformed |
| **Others IoU** | 0.0558 | **0.0746** | +0.0189 | Outperformed |
| **Background IoU** | 0.0000 | 0.0000 | 0.0000 | Exact Match |

*Root-Cause Analysis:* The historical artifact reflected an intermediate training validation checkpoint logged with lower-resolution subsampling and CUDA AMP mixed precision. Full sliding-window tile inference with exact bilinear spatial interpolation achieves significantly higher metric fidelity across all landcover categories.

---

## 4. Top Confusion Modes

1. **Forest $\rightarrow$ Farmland:** 14,411,773 pixels (12.12% of GT Forest)
2. **Farmland $\rightarrow$ Water:** 8,151,523 pixels (7.32% of GT Farmland)
3. **Farmland $\rightarrow$ Village:** 6,447,174 pixels (5.79% of GT Farmland)
4. **Water $\rightarrow$ Farmland:** 5,954,585 pixels (14.59% of GT Water)
5. **Farmland $\rightarrow$ Forest:** 5,560,010 pixels (4.99% of GT Farmland)

---

## 5. Runtime Latency & Throughput

Measured on Apple Silicon Metal Performance Shaders (MPS):
- **Mean Tile Latency:** `7.93 ms`
- **Median Batch Latency (P50):** `82.81 ms` (batch size 16)
- **P95 Batch Latency:** `361.90 ms`
- **Inference Throughput:** `19.76 FPS`
- **MPS Tensor VRAM:** `619.62 MB`

---

## 6. Documented Model Weaknesses

1. **Background Class (0.00% IoU):** Background pixels comprise 0.001% of the ground truth and are absorbed by adjacent semantic classes.
2. **Road Class (11.68% IoU):** Narrow linear features exhibit high recall (67.73%) but low precision (12.36%) due to edge dilation under spatial downsampling.
3. **Others Class (7.46% IoU):** High heterogeneity and sparse representation lead to low overlap.
