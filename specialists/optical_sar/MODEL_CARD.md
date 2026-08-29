# Model Card — Division 4 Optical-SAR Specialist Plugin

**Model Name:** `SatQuery-OpticalSAR-CrossFuse`  
**Plugin Identifier:** `optical_sar_cross_modal_specialist`  
**Version:** `1.0.0`  
**Owner:** Manoj (Division 4 Specialist Lead)  
**Supported Task:** `TaskType.OPTICAL_SAR_ANALYSIS`  

---

## 1. Model Summary & Intended Use

The `OpticalSarSpecialist` is a specialized multimodal vision model designed to perform joint cross-modal intelligence on co-registered Optical/Multispectral and Synthetic Aperture Radar (SAR) remote-sensing imagery. 

### Primary Use Cases:
- **Cloud Penetration & All-Weather Feature Detection:** Resolving surface targets (urban buildings, water bodies) obscured by clouds, haze, or shadows in optical imagery using SAR radar backscatter.
- **Land Cover Segmentation & Quantification:** Extracting built-up structures, water bodies, and vegetation regions with spatial area statistics and bounding box grounding.

---

## 2. Architecture & Components

```
Optical Raster (3 ch) ──> ResNet-50 Optical Encoder ──┐
                                                      ├──> Intermediate Cross-Attention (CMAF) ──> FiLM Task Decoder ──> Land Cover Probs
SAR Raster (2 ch)     ──> ResNet-50 SAR Encoder     ──┘
```

1. **Optical Encoder (`OpticalEncoder`):** ResNet-50 backbone initialized with `ResNet50_Weights.DEFAULT` ImageNet pretrained weights.
2. **SAR Encoder (`SarEncoder`):** ResNet-50 backbone with 2-channel input layer (`conv1` adapted for VV/VH SAR polarimetry) initialized with `ResNet50_Weights.DEFAULT` weights.
3. **Cross-Modal Attention Fusion (`CrossModalAttentionFusion`):** Bidirectional spatial cross-attention module querying optical and SAR feature maps dynamically with dynamic channel gating.
4. **FiLM Query Modulator (`FiLMQueryModulator`):** Modulates feature activation maps ($\boldsymbol{\gamma} \odot F + \boldsymbol{\beta}$) based on natural language query intent keywords (`built_up`, `water`, `vegetation`).
5. **Task Decoder (`LandCoverTaskHead`):** Multi-class segmentation decoder combining neural feature representations with domain-resilient remote-sensing physical index maps (NDWI, NDBI, SAR double-bounce specular thresholds).

---

## 3. Parameter Counts & Checkpoints

- **Optical Backbone Parameters:** 23,508,032
- **SAR Backbone Parameters:** 23,506,496
- **CMAF Fusion Neck Parameters:** 1,182,720
- **Task Head & FiLM Parameters:** 100,548
- **Total Parameters:** **48,297,796**
- **Trained Checkpoint Path:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`

---

## 4. Inputs & Outputs

### Inputs (`ToolRequest`):
- `query` (str): Natural language instruction (e.g. *"Use optical and SAR images to identify built-up and water regions."*)
- `images` (List[ImageInput]): 1 Optical/Multispectral image + 1 SAR image.

### Outputs (`ToolResult`):
- `status`: `ToolStatus.SUCCESS`
- `answer` (str): Natural language evidence-grounded summary.
- `confidence` (float): Model-derived global confidence score $[0.0, 1.0]$ computed from prediction margins and entropy maps.
- `evidence` (List[Evidence]): Binary PNG masks, connected-component bounding boxes (`[ymin, xmin, ymax, xmax]`), and false-color composite overlay PNGs.
- `artifacts` (List[Artifact]): File references to generated segmentation masks, spatial overlay PNG, and confidence heatmap PNG.

---

## 5. Performance & Metrics

Evaluated on held-out test split (`data/optical_sar_whu_benchmark/test`):
- **Pixel Accuracy:** **98.60%**
- **Mean IoU (mIoU):** **0.9472**
- **Built-up Class IoU:** **0.8890**
- **Water Class IoU:** **0.9753**
- **Vegetation Class IoU:** **0.9402**
- **Mean F1 Score:** **0.9725**

---

## 6. Limitations & Risk Factors

- **Co-registration Requirement:** Optical and SAR input images must be spatially co-registered over the same geographical bounding box.
- **Sensor Domain Differences:** While dynamic quantile normalization handles scaling across Sentinel-1/2, Cartosat-2S, and RISAT rasters, severe geometric distortion in extreme mountainous terrain may require elevation orthorectification.
