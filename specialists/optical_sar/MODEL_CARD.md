# Model Card — Division 4 Optical-SAR Specialist Plugin

**Model Name:** `SatQuery-OpticalSAR-CrossFuse`  
**Plugin Identifier:** `optical_sar_cross_modal_specialist`  
**Version:** `1.0.0`  
**Supported Task:** `TaskType.OPTICAL_SAR_ANALYSIS`  

---

## 1. Model Summary & Intended Use

The `OpticalSarSpecialist` is a multimodal vision model designed to perform joint cross-modal inference on co-registered Optical/Multispectral and Synthetic Aperture Radar (SAR) remote-sensing imagery. Cross-modal Optical + SAR inference is demonstrated; SAR radar backscatter provides complementary physical scattering information independent of optical illumination and cloud cover.

### Primary Capabilities:
- **Joint Cross-Modal Reasoning:** Combines optical spectral reflectance (RGB / multispectral bands) with SAR dielectric backscatter (VV/VH polarizations) to disambiguate spectral ambiguities (e.g., distinguishing smooth water bodies from asphalt or shadowed forest).
- **8-Class Land Cover Segmentation:** Produces dense pixel-level classification across 8 classes (`Background`, `Farmland`, `City`, `Village`, `Water`, `Forest`, `Road`, `Others`).
- **Spatial Grounding & Evidence Extraction:** Generates binary segmentation masks, connected-component bounding boxes (`[ymin, xmin, ymax, xmax]`), and calibrated confidence heatmaps.

---

## 2. Architecture & Components

The model uses a dual-encoder architecture with intermediate cross-modal attention and FiLM-modulated task decoding:

```
Optical Raster (3 ch) ──> Truncated ResNet-50 (Stems + L1-3) ──┐
                                                               ├──> CMAF Cross-Attention Neck ──> FiLM Task Head ──> 8-Class Logits
SAR Raster (2 ch)     ──> Truncated ResNet-50 (Stems + L1-3) ──┘
```

1. **Optical Encoder (`OpticalEncoder`):** Torchvision ResNet-50 truncated at `layer3` (strides 4, 8, 16 with channel depths 256, 512, 1024), taking 3 RGB channels. Initialized from ImageNet pretrained weights.
2. **SAR Encoder (`SarEncoder`):** Torchvision ResNet-50 truncated at `layer3` with modified 2-channel `conv1` (adapted for VV/VH dual-polarization SAR backscatter).
3. **Cross-Modal Attention Fusion (`CrossModalAttentionFusion`):** Bidirectional spatial cross-attention neck (CMAF) that projects optical and SAR feature maps to 256 dimensions, exchanges cross-modal spatial context, and combines features via dynamic gating.
4. **Task Decoder (`LandCoverTaskHead`):** Multi-scale convolutional decoder with `FiLMQueryModulator` feature-wise linear modulation conditioned on natural language query keywords, outputting 8-class segmentation logits.

---

## 3. Parameter Counts & Checkpoints

- **Optical Encoder (Truncated ResNet-50):** 8,543,296 parameters
- **SAR Encoder (Truncated ResNet-50, 2 ch):** 8,540,160 parameters
- **CMAF Fusion Neck:** 2,297,344 parameters
- **Task Head & FiLM Modulator:** 374,344 parameters
- **Total Model Parameters:** **19,755,144**

### Checkpoints:
- **Production Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  
  - **SHA-256:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`  
  - **Size:** 79,480,705 bytes (75.80 MB)
- **Preserved Rollback Checkpoint:** `specialists/optical_sar/checkpoints/archive/cmaf_landcover_best_pre_v3.pth`  
  - **SHA-256:** `8a3baac9269db8423a472a6814d7820b8cfea67994305ad2e70541d6a1d1f1c9`  
  - **Size:** 79,473,698 bytes (75.79 MB)

---

## 4. Inputs & Outputs

### Inputs (`ToolRequest`):
- `query` (str): Natural language instruction (e.g., *"Segment water and farmland using optical and SAR sensor rasters."*)
- `images` (List[ImageInput]): 1 Optical/Multispectral raster + 1 SAR raster.

### Outputs (`ToolResult`):
- `status`: `ToolStatus.SUCCESS`
- `answer` (str): Natural language evidence-grounded summary.
- `confidence` (float): Global confidence score $[0.0, 1.0]$ derived from prediction margins and entropy.
- `evidence` (List[Evidence]): Binary PNG masks, connected-component bounding boxes (`[ymin, xmin, ymax, xmax]`), and false-color composite overlay PNGs.
- `artifacts` (List[Artifact]): File references to generated segmentation masks, overlay PNGs, and confidence heatmaps.

---

## 5. Authoritative Evaluation Benchmark

Evaluated on the official Wuhan University WHU-OPT-SAR held-out test split (15 real test scenes, 4,950 paired tiles, 308,687,656 valid pixels):

| Metric | Measured Value |
| :--- | :--- |
| **Overall Accuracy (OA)** | **71.71%** (0.717099) |
| **Mean IoU (mIoU)** | **35.08%** (0.350793) |
| **Macro F1 Score** | **46.62%** (0.466209) |
| **Macro Precision** | **46.58%** (0.465837) |
| **Macro Recall** | **52.48%** (0.524761) |
| **Weighted F1 Score** | **74.18%** (0.741756) |
| **Active Classes** | **8 / 8** |

### Per-Class IoU Breakdown:
- **Forest:** `73.69%` (F1: `84.85%`)
- **Farmland:** `59.20%` (F1: `74.37%`)
- **Water:** `50.71%` (F1: `67.29%`)
- **City:** `44.57%` (F1: `61.66%`)
- **Village:** `33.32%` (F1: `49.99%`)
- **Road:** `11.68%` (F1: `20.91%`)
- **Others:** `7.46%` (F1: `13.89%`)
- **Background:** `0.00%` (F1: `0.00%`)

*Contextual Historical Snapshot:* The intermediate training validation artifact (`final_test_metrics.json`) recorded 56.15% OA and 23.15% mIoU under lower-resolution validation callback subsampling; the fresh authoritative evaluation on full-resolution test tiles establishes the true production performance.

---

## 6. Model Limitations & Documented Weaknesses

- **Underperforming Classes:**
  - **Background (0.00% IoU):** Background constitutes only 0.001% of ground-truth pixels in the WHU dataset and is effectively absorbed into landcover classes.
  - **Road (11.68% IoU):** Fine linear road networks suffer from boundary blurring under 512x512 tile downsampling and SAR speckle noise.
  - **Others (7.46% IoU):** Highly heterogeneous catch-all class with limited training representation.
- **Co-registration Requirement:** Optical and SAR input images must be spatially aligned over identical geographical bounds.
- **Speckle & Terrain Artifacts:** Extreme topographic slopes can produce SAR foreshortening and layover, which require orthorectification in severe relief environments.
