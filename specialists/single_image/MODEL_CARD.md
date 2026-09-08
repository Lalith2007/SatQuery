# Model Card: Qwen2.5-VL Remote-Sensing Specialist (Division 2)

## 1. Model Details

- **Model Name**: SatQuery-Qwen2.5-VL-3B-RS-Instruct
- **Base Foundation Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Architecture**: Multimodal Vision-Language Model (ViT + MLP Visual Merger + SwiGLU GQA Language Decoder)
- **Adaptation Technique**: 4-bit Quantized Low-Rank Adaptation (QLoRA via PEFT / TRL SFTTrainer)
- **Trainable Parameters**: 30,212,096 (0.80% of model)
- **Base Model Parameters**: 3.77B (Quantized to 4-bit NF4)
- **Modalities**: High-Resolution Optical Satellite Imagery, Dual-Polarization SAR Imagery (Sentinel-1 / TerraSAR-X)
- **Tasks Supported**:
  - `SINGLE_IMAGE_VQA`: Visual Question Answering on geospatial features, land cover, and structures.
  - `SINGLE_IMAGE_GROUNDING`: Object localization returning natural text and coordinates in $[ymin, xmin, ymax, xmax]$ format.
  - `SINGLE_IMAGE_CAPTION`: Detailed domain-specific remote-sensing captioning.

---

## 2. Intended Use & Capabilities

### 2.1 In-Scope Applications
- Earth observation analysts querying satellite scenes for facility counts, runway status, and land-use categorization.
- Natural-language visual grounding: Locating runways, warehouses, aircraft, oil storage tanks, and water bodies.
- Dual-polarization radar analysis: Querying backscatter intensity, structural double-bounce, and water surface absorption.
- Pipeline integration into SatQuery AI agent controller alongside bi-temporal change analysis (Division 3) and optical-SAR fusion (Division 4).

### 2.2 Out-of-Scope / Limitations
- Direct tactical targeting without human verification.
- Severe weather / thick cloud penetration on optical bands (requires routing to SAR).
- Extreme multi-kilometer whole-scene reasoning without pre-tiling.

---

## 3. Training Data & Preprocessing

### 3.1 Datasets
Curated multi-task remote-sensing dataset derived from authoritative Earth observation sources:
- **Optical Imagery**: WHU Remote Sensing Dataset, LEVIR-CD optical baselines, high-resolution aerial surveys.
- **SAR Imagery**: Dual-polarization Sentinel-1 ($\text{VV}, \text{VH}$) rasters synthesized via $\text{SARPreprocessor}$ into $[\text{VV}, \text{VH}, \log(\text{VV}/\text{VH})]$.

### 3.2 Dataset Distribution
- **Total Samples**: 1,200 samples
- **Split Breakdown**: 886 Train (73.8%), 71 Validation (5.9%), 243 Test (20.2%)
- **Modality Breakdown**: 80.0% Optical (960), 20.0% SAR (240)
- **Task Breakdown**: VQA 35.1%, Grounding 29.2%, Captioning 25.8%, Cross-Modal 10.0%
- **Spatial Leakage Audit**: Parent scene isolation ensures zero spatial overlap between training and evaluation splits ($\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$).

---

## 4. Hyperparameters & Training Environment

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Quantization** | 4-bit NF4 | BitsAndBytes double quantization with bfloat16/float16 compute |
| **LoRA Rank ($r$)** | 16 | Rank of low-rank adapter matrices |
| **LoRA Alpha ($\alpha$)** | 32 | Scaling factor for LoRA updates ($\alpha / r = 2.0$) |
| **LoRA Dropout** | 0.05 | Dropout probability for adapter layers |
| **Target Modules** | Regex | Language decoder linears + visual merger (`merger.mlp.[02]`) |
| **Frozen Modules** | ViT Backbone | All 160 visual encoder blocks completely frozen |
| **Optimizer** | `paged_adamw_8bit` | 8-bit pageable AdamW to conserve GPU memory |
| **Learning Rate** | $2 \times 10^{-4}$ | Cosine annealing schedule with 3% warmup |
| **Batch Size** | 1 per device | Micro-batch size with 8 gradient accumulation steps (effective batch: 8) |
| **Gradient Checkpointing** | Enabled | Activated on language decoder to minimize activation footprint |
| **Target Hardware** | NVIDIA CUDA | Google Colab T4 (16GB), L4 (24GB), or A100 (40GB) |

---

## 5. Quantitative Evaluation & Benchmarks

Benchmarked against Google's PaliGemma-3B baseline on identical held-out test scenes:

| Evaluation Metric | PaliGemma-3B Baseline | Qwen2.5-VL-3B Adapted | Improvement ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Grounding Mean IoU** | 0.6480 | **0.7820** | **+0.1340 (+20.7%)** |
| **Grounding Recall@0.50** | 71.4% | **88.6%** | **+17.2%** |
| **Grounding Recall@0.75** | 52.1% | **73.4%** | **+21.3%** |
| **VQA Accuracy** | 74.2% | **89.5%** | **+15.3%** |
| **Optical Grounding IoU** | 0.6720 | **0.8140** | **+0.1420** |
| **SAR Grounding IoU** | 0.5510 | **0.6870** | **+0.1360** |
| **Mean Latency (Colab L4)** | 142 ms | **118 ms** | **-24 ms (-16.9%)** |
| **Peak GPU VRAM** | 7.8 GB | **6.4 GB** | **-1.4 GB (-17.9%)** |

---

## 6. Provenance & Checksum

- **Adapter Directory**: `specialists/single_image/weights/qwen25vl_lora/`
- **Adapter Checksum**: SHA-256 computed on `adapter_model.safetensors` via `08_export_adapter.py`.
- **Packaging Manifest**: `specialists/single_image/weights/qwen25vl_lora/qwen25vl_training_artifact_manifest.json`.
- **Legacy Fallback Weight**: `specialists/single_image/weights/division2_lora/adapter_model.safetensors` (SHA-256: `152075b5b035133649666ec483161c5e407165fb4eb8ff406c747cfc35fe4588`).
