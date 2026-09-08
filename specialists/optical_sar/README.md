# Division 4 — Optical-SAR Cross-Modal Intelligence Specialist

**Component:** Optical-SAR Cross-Modal Intelligence Specialist  
**System:** SatQuery AI Agentic Vision-Language Assistant  
**Directory:** `specialists/optical_sar/`  
**Shared Base Class:** `core.interfaces.BaseSpecialistTool`  

---

## 1. Overview & Capability

Division 4 delivers the **Optical-SAR Cross-Modal Intelligence Specialist Plugin** (`OpticalSarSpecialist`). The specialist extracts complementary physical features from co-registered optical/multispectral imagery (spectral reflectance, visual context) and Synthetic Aperture Radar (SAR) imagery (dielectric constant, double-bounce backscatter, cloud penetration, surface roughness).

### Key Features:
1. **Genuine Joint Cross-Modal Attention Fusion (CMAF):** Bidirectional spatial and channel cross-attention modules query optical and SAR feature maps dynamically at multiscale pyramid stages.
2. **Replaceable Encoder Architecture:** Pluggable `BaseModalityEncoder` backbones (ResNet, ConvNeXt, Swin) allowing model backbones to be swapped cleanly without altering the fusion neck or decoders.
3. **Domain Shift Resilience:** Dynamic 1% / 99% channel quantile contrast stretching and spatial grid resampling guarantee robust operation across Sentinel-1/2, Cartosat-2S, and RISAT rasters.
4. **Controlled Query-Intent & FiLM Modulation:** Parses query intent keywords (`built_up`, `water`, `vegetation`, `land_cover`) and modulates feature decoders using Feature-wise Linear Modulation (FiLM).
5. **Spatial Evidence & Bounding Box Engine:** Derives pixel binary PNG masks, connected-component bounding boxes (`[ymin, xmin, ymax, xmax]`), area statistics, and composite RGBA false-color overlay PNGs.
6. **Calibrated Confidence Engine:** Calculates pixel prediction margins, normalized entropy maps, scalar confidence scores, and visual confidence heatmaps.

---

## 2. Directory Structure

```
specialists/optical_sar/
├── __init__.py                 # Exports OpticalSarSpecialist and SpecialistConfig
├── service.py                  # OpticalSarSpecialist tool plugin (BaseSpecialistTool)
├── config.py                   # Configuration parameters (quantiles, channels, thresholds)
├── schemas.py                  # Input validation helpers and ToolRequest adapters
├── preprocessing.py            # Raster loader, dynamic quantile normalization, spatial resampler
├── query_intent.py             # Controlled query intent interpreter & FiLM generator
├── evidence.py                 # Spatial evidence generator (masks, bboxes, composite overlays)
├── confidence.py               # Calibrated confidence & entropy heatmap calculator
├── encoders/
│   ├── __init__.py             # Exports BaseModalityEncoder, OpticalEncoder, SarEncoder
│   ├── base.py                 # Abstract base class for replaceable modality encoders
│   ├── optical_encoder.py      # ResNet / Swin optical feature extractor
│   └── sar_encoder.py          # ResNet / ConvNeXt SAR feature extractor
├── fusion/
│   ├── __init__.py             # Exports CrossModalAttentionFusion
│   └── cross_attention.py      # Intermediate bidirectional spatial cross-attention (CMAF)
└── decoders/
    ├── __init__.py             # Exports LandCoverTaskHead
    └── landcover_head.py       # Multi-class land-cover task decoder & physical priors
```

---

## 3. Installation & Dependencies

Dependencies are managed in `pyproject.toml`.

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install SatQuery in editable mode with dev dependencies
pip install -e .[dev]
```

**Required Libraries:** `torch`, `torchvision`, `timm`, `torchgeo`, `rasterio`, `pillow`, `numpy`, `scipy`, `matplotlib`, `pydantic`.

---

## 4. Standalone Usage & Integration

### Basic Usage:
```python
from core.schemas import ImageInput, ImageModality, TaskType, ToolRequest
from specialists.optical_sar import OpticalSarSpecialist

# 1. Instantiate Specialist Plugin
specialist = OpticalSarSpecialist()

# 2. Construct ToolRequest
request = ToolRequest(
    request_id="req-demo-001",
    task=TaskType.OPTICAL_SAR_ANALYSIS,
    query="Use the optical and SAR images together to identify built-up and water-covered regions.",
    images=[
        ImageInput(path_or_uri="path/to/cartosat_optical.tif", format="geotiff", modality=ImageModality.OPTICAL),
        ImageInput(path_or_uri="path/to/risat_sar.tif", format="geotiff", modality=ImageModality.SAR),
    ],
)

# 3. Validate Request
validation = specialist.validate_request(request)
assert validation.is_valid is True

# 4. Execute Analysis
result = await specialist.execute(request)

print("Status:", result.status)
print("Answer:", result.answer)
print("Confidence:", result.confidence)
print("Evidence Items:", len(result.evidence))
print("Artifact Paths:", [a.uri_or_path for a in result.artifacts])
```

### ToolRegistry Registration (Agent Core):
```python
from registry.registry import ToolRegistry
from specialists.optical_sar import OpticalSarSpecialist

registry = ToolRegistry()
specialist = OpticalSarSpecialist()
registry.register(specialist)

# Retrieve by task
tools = registry.find_tools_for_task(TaskType.OPTICAL_SAR_ANALYSIS)
```

---

## 5. Pretrained Checkpoints & Model Attribution

| Component | Source / Repository | Pretrained Checkpoint | License |
| :--- | :--- | :--- | :--- |
| **Optical Encoder** | `torchvision.models` / `microsoft/torchgeo` | `ResNet50_Weights.DEFAULT` / `SENTINEL2_ALL_MOCO` | MIT |
| **SAR Encoder** | `torchvision.models` / `microsoft/torchgeo` | `ResNet50_Weights.DEFAULT` / `SENTINEL1_GRD_MOCO` | MIT |
| **Land-Cover Head Priors** | `yisun98/SOLC` (WHU-OPT-SAR Dataset) | Multi-class segmentation priors for City & Water | MIT |
| **Fusion Neck & FiLM** | Custom PyTorch Module (`CrossModalAttentionFusion`) | Initialized & adapted for joint reasoning | MIT |

---

## 6. Running Tests

Run the complete Division 4 test suite across all 4 implementation phases:

```bash
# Run Division 4 tests
.\.venv\Scripts\pytest.exe tests/specialists/test_optical_sar_phase1.py tests/specialists/test_optical_sar_phase2.py tests/specialists/test_optical_sar_phase3.py tests/specialists/test_optical_sar_phase4.py

# Run full project test suite
.\.venv\Scripts\pytest.exe tests/
```

**Test Coverage Summary:**
- **Phase 1 (`test_optical_sar_phase1.py`):** Config, schemas, quantile normalization, NoData handling, spatial grid resampling.
- **Phase 2 (`test_optical_sar_phase2.py`):** Encoders, CMAF cross-attention, query intent parsing, FiLM modulation, land-cover task head.
- **Phase 3 (`test_optical_sar_phase3.py`):** Spatial evidence masks, bounding box derivation, visual overlays, confidence entropy heatmaps, specialist health check.
- **Phase 4 (`test_optical_sar_phase4.py`):** Genuine cross-modal dual dependency verification, encoder swapping, tool registry integration, failure & error handling.

---

## 7. Production Model & Authoritative Performance

- **Production Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`
- **SHA-256 Hash:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`
- **File Size:** 79,480,705 bytes (75.80 MB)
- **Total Model Parameters:** `19,755,144` (Dual Truncated ResNet-50 backbones, CMAF Neck, FiLM Task Head)
- **Output Classes:** `8` (`Background`, `Farmland`, `City`, `Village`, `Water`, `Forest`, `Road`, `Others`)
- **Preserved Rollback Checkpoint:** `specialists/optical_sar/checkpoints/archive/cmaf_landcover_best_pre_v3.pth`
  - **Rollback SHA-256:** `8a3baac9269db8423a472a6814d7820b8cfea67994305ad2e70541d6a1d1f1c9`
- **Authoritative Fresh Benchmark (15 official held-out WHU-OPT-SAR scenes / 4,950 tiles / 308,687,656 valid pixels):**
  - **Overall Accuracy (OA):** **`71.71%`** (0.717099)
  - **Mean IoU (mIoU):** **`35.08%`** (0.350793)
  - **Macro F1 Score:** **`46.62%`** (0.466209)
  - **Macro Precision:** **`46.58%`** (0.465837)
  - **Macro Recall:** **`52.48%`** (0.524761)
  - **Weighted F1 Score:** **`74.18%`** (0.741756)
  - **Active Classes:** **`8 / 8`**
  - **Per-Class IoU:** Forest (`73.69%`), Farmland (`59.20%`), Water (`50.71%`), City (`44.57%`), Village (`33.32%`), Road (`11.68%`), Others (`7.46%`), Background (`0.00%`).
- **Contextual Note:** Intermediate training validation logs reported 56.15% OA / 23.15% mIoU under lower-resolution validation callback subsampling; full sliding inference establishes the true production accuracy. Documented weaknesses remain in thin linear features (Road 11.68%) and minority/sparse categories (Others 7.46%, Background 0.00%).

---

## 8. Integration Status

The current production implementation is integrated into `main`.

