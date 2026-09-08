# Model Card — Division 3 Bi-Temporal Change Intelligence

**Model Name:** `ChangeDetector-TinyCD`  
**Tool Identifier:** `bitemporal_change_specialist`  
**Version:** `1.0.0`  
**Supported Tasks:** `TaskType.CHANGE_ANALYSIS`, `TaskType.CHANGE_VQA`  

---

## 1. Model Summary & Intended Use

The `BiTemporalChangeSpecialist` is a specialized remote-sensing change detection model designed to perform space-time comparative analysis between co-registered bi-temporal optical acquisitions ($T_0$ and $T_1$). It detects structural surface changes—predominantly building emergence, construction, and demolition—while maintaining robustness against seasonal, phenological, and atmospheric illumination shifts.

### Primary Capabilities:
- **Bi-Temporal Difference Modeling:** Ingests paired optical rasters ($T_0$ and $T_1$), computes cross-attention spatial-channel differences, and predicts dense per-pixel change probabilities in $[0.0, 1.0]$.
- **Spatial Localization & Bounding Boxes:** Extracts discrete connected components, calculates bounding boxes (`[ymin, xmin, ymax, xmax]`), and computes total changed area metrics.
- **Change VQA & Reasoning:** Supports natural-language queries regarding surface change counts, presence, and spatial characteristics.

---

## 2. Architecture & Components

The model implements the `TinyCD` Siamese U-Net architecture enhanced with Mixing Mask Attention Blocks (MAMB) (Andrea Codegoni et al.):

```
T0 [1, 3, 256, 256] ─┐
                     ├─► Shared Siamese Stem [ConvBlock: 3 -> 32]
T1 [1, 3, 256, 256] ─┘
  │
  ├─► Down1 (MaxPool2d + ConvBlock: 32 -> 64)   ─► MAMB 1 (64 ch)   ─► Skip to Dec1
  ├─► Down2 (MaxPool2d + ConvBlock: 64 -> 128)  ─► MAMB 2 (128 ch)  ─► Skip to Dec2
  └─► Down3 (MaxPool2d + ConvBlock: 128 -> 256) ─► MAMB 3 (256 ch)  ─► ConvTranspose2d (up2)
                                                                            │
      Decoder: up2 (256->128) + MAMB2 ──► Dec2 (ConvBlock: 256 -> 128) ─────┤
               up1 (128->64)  + MAMB1 ──► Dec1 (ConvBlock: 128 -> 64)  ─────┤
               up0 (64->32)   + Stem  ──► Dec0 (ConvBlock: 64 -> 32)   ─────┤
                                                                            ▼
      Classifier: Conv2d(32, 1, 1x1) + Sigmoid() ─────────────────► [1, 1, 256, 256]
```

1. **Shared Siamese Encoder:** Twin 4-stage convolutional backbone with shared weights across $T_0$ and $T_1$ branches. Features double-conv blocks with BatchNorm and ReLU.
2. **Mixing Mask Attention Blocks (MAMB):** Operates at stages 1, 2, and 3:
   - **Spatial Attention:** $7\times 7$ convolution applied to concatenated average- and max-pooled representations of $|f_0 - f_1|$.
   - **Channel Attention:** Squeeze-and-excitation block ($1\times 1$ convs + ReLU + Sigmoid) on feature differences.
   - **Cross-Modulation:** Features from both temporal branches are modulated by spatial and channel attention masks, concatenated, and fused via $3\times 3$ convolution + BatchNorm + ReLU.
3. **U-Net Decoder:** 3 transpose convolution upsampling stages with skip connections from the MAMB feature maps and the encoder stem.
4. **Classifier Head:** Pointwise $1\times 1$ convolution producing single-channel logit maps followed by `Sigmoid()` activation.

---

## 3. Checkpoint Identity & Parameters

- **Production Checkpoint:** `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`
- **Cryptographic SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`
- **File Size:** `14,326,623` bytes (13.66 MB)
- **Parameter Count:**
  - Total Parameters: **`3,565,034`**
  - Trainable Parameters: **`3,565,034`**
  - Weight Tensors: **`145`**
- **Strict Loading:** Verified with `strict=True` (0 missing, 0 unexpected keys).

---

## 4. Input & Output Contracts

### Inputs:
- `query` (str): Optional natural language query (e.g., *"Detect any newly constructed buildings."*).
- `images` (List[ImageInput]): Exactly two co-registered optical images ($T_0$ and $T_1$).
- Input Resolution: Standardized to $256\times 256$ via bilinear interpolation.
- Channel Format: 3-channel RGB normalized to $[0.0, 1.0]$ float32.

### Outputs:
- Dense probability map $\hat{Y} \in [0.0, 1.0]^{H \times W}$.
- Binary change mask $M \in \{0, 1\}^{H \times W}$ thresholded at $\tau=0.50$.
- Extracted connected components with spatial bounding boxes.
- Visual artifacts: `binary_change_mask.png`, `change_probability_heatmap.png`, `bitemporal_change_composite_*.png`.

---

## 5. Authoritative Performance

Evaluated on the **official standard LEVIR-CD held-out test split** (128 full-scale $1024\times 1024$ parent scenes, $8,388,608$ evaluated pixels):

| Metric | Raw Probability Output ($\tau=0.50$) | Production Postprocessed (Morphology + CC) |
| :--- | :---: | :---: |
| **Overall Accuracy (OA)** | **`97.99%`** | **`97.99%`** |
| **Precision** | **`82.70%`** | **`83.36%`** |
| **Recall** | **`76.62%`** | **`75.63%`** |
| **F1 Score** | **`79.54%`** | **`79.31%`** |
| **IoU (Jaccard Index)** | **`66.03%`** | **`65.71%`** |
| **Dice Coefficient** | `79.54%` | `79.31%` |
| **Specificity** | `99.14%` | `99.19%` |
| **True Positives (TP)** | 327,477 | 323,254 |
| **False Positives (FP)** | 68,505 | 64,540 |
| **False Negatives (FN)** | 99,937 | 104,160 |
| **True Negatives (TN)** | 7,892,689 | 7,896,654 |
| **Ground-Truth Changed Pixels** | 427,414 (`5.0952%`) | 427,414 (`5.0952%`) |
| **Ground-Truth Unchanged Pixels** | 7,961,194 (`94.9048%`) | 7,961,194 (`94.9048%`) |

### Threshold Sensitivity:
Evaluating thresholds between $0.30$ and $0.70$ shows that F1 remains stable across $[0.35, 0.50]$ ($79.4\% \rightarrow 79.9\%$), indicating robust threshold behavior across varying confidence cutoffs without erratic metric drop-off.

---

## 6. Runtime & Hardware

| Environment | Forward Latency | Throughput |
| :--- | :---: | :---: |
| **Apple Silicon Metal (MPS)** | **`27.32 ms`** (Mean) / `26.98 ms` (P50) | **`36.61 FPS`** |
| **Apple Silicon Host CPU** | `156.31 ms` (Mean) / `154.32 ms` (P50) | `6.40 FPS` |
| **NVIDIA Tesla T4 GPU (Historical)** | `15.49 ms` (Mean) | `64.50 FPS` |

---

## 7. Determinism

Evaluated across repeated inference passes on identical test pairs:
- **Maximum Output Difference:** `0.0`
- Deterministic inference is verified under fixed seed and evaluation mode.

---

## 8. Known Limitations & Failure Modes

1. **Thin Boundary Artifacts:** High-contrast rooftop eaves and building perimeter contours occasionally exhibit 1-pixel dilation/erosion offsets between $T_0$ and $T_1$ ground truths.
2. **Illumination and Seasonal Variance:** High solar zenith angle shifts create shadowing adjacent to multi-story buildings that the model occasionally flags as false positive changes.
3. **Severe Class Imbalance:** Building changes represent only `5.0952%` of total pixels; natural background terrain comprises `94.9048%`.
4. **Small Component Dropout:** Postprocessing with `min_region_area=100` filters out isolated pixel noise but suppresses tiny shed or outbuilding emergence under 100 pixels in area.
5. **No Semantic Transition Identification:** The model performs binary change detection only; identifying specific transition types (e.g., farmland to building vs. forest to building) requires downstream semantic classification.
