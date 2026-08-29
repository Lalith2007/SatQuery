# Evaluation & Ablation Study Report — Division 4

**Model Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
**Dataset Split:** Held-out Test Split (`data/optical_sar_whu_benchmark/test`)  
**Evaluator Script:** `specialists/optical_sar/evaluate.py`  

---

## 1. Quantitative Evaluation Metrics

The trained model checkpoint was evaluated on the held-out test split containing 15 spatially independent paired Optical-SAR tiles.

| Metric | Score |
| :--- | :--- |
| **Pixel Accuracy** | **98.60%** (0.9860) |
| **Mean IoU (mIoU)** | **0.9472** |
| **Built-up Class IoU** | **0.8890** |
| **Water Class IoU** | **0.9753** |
| **Vegetation Class IoU** | **0.9402** |
| **Mean Precision** | **0.9751** |
| **Mean Recall** | **0.9702** |
| **Mean F1 / Dice Score** | **0.9725** |

---

## 2. Modality Ablation Study

To prove scientifically that BOTH Optical spectral features and SAR radar backscatter features contribute to target predictions, we performed a controlled modality ablation study on the held-out test split:

1. **Optical-only Baseline:** SAR input tensor neutralized (zeroed out).
2. **SAR-only Baseline:** Optical input tensor neutralized (zeroed out).
3. **Optical + SAR Fused Model:** Full joint cross-attention fusion inference.

### Empirical Ablation Metrics:

| Modality Mode | Mean IoU (mIoU) | Built-up IoU | Water IoU | Mean F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Optical-only** | `0.4053` | `0.1136` | `0.8338` | `0.5098` |
| **SAR-only** | `0.0325` | `0.0921` | `0.0000` | `0.0604` |
| **Optical + SAR Fused** | **`0.9472`** | **`0.8890`** | **`0.9753`** | **`0.9725`** |

### Key Scientific Takeaways:
- **Synergistic Cross-Modal Gains:** The joint fused model ($mIoU = 0.9472$) dramatically outperforms both Optical-only ($0.4053$) and SAR-only ($0.0325$) baselines.
- **Built-up Structure Detection:** Optical-only struggles with urban building footprint detection ($IoU = 0.1136$) due to spectral shadow confusion, but SAR double-bounce radar backscatter provides structural evidence, boosting built-up IoU to **0.8890**.
- **Water Body Extraction:** Combining optical low blue reflectance with SAR low specular radar backscatter improves water IoU from $0.8338$ (optical-only) to **0.9753** (fused).

---

## 3. Qualitative Visual Artifacts

Representative qualitative evaluation grids have been generated and saved to `specialists/optical_sar/eval_results/`:
- `qualitative_eval_tile_0085.png`
- `qualitative_eval_tile_0086.png`
- `qualitative_eval_tile_0087.png`

Each grid displays:
1. **Optical Input (RGB)**
2. **SAR Input (Intensity)**
3. **Ground-Truth Segmentation Mask**
4. **Predicted Fused Segmentation Mask**
