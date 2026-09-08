# SatQuery AI — Division 2: Qwen2.5-VL Foundation Migration Document

**Status**: Phase A & Phase B Complete (Colab CUDA Training Ready)  
**Author**: Division 2 (Single-Image Remote-Sensing Intelligence)  
**Target Runtime**: Google Colab CUDA Runtime (T4 / L4 / A100)  
**Local Runtime**: Apple Silicon Mac (Orchestration, Validation, Tests, API Integration)  

---

## 1. Executive Summary & Migration Topology

This document establishes the architecture, migration strategy, and execution protocol for migrating SatQuery AI's **Division 2 (Single-Image Remote-Sensing Intelligence Specialist)** from Google's PaliGemma-3B foundation model to Alibaba Cloud's **Qwen2.5-VL** family (`Qwen/Qwen2.5-VL-3B-Instruct` default, with `Qwen/Qwen2.5-VL-7B-Instruct` configurable).

### 1.1 Development and Execution Topology

```
LOCAL / ANTIGRAVITY (Apple Silicon Mac)
    │
    ├─ Code Authoring & Model Abstraction
    ├─ Dataset Preparation & Spatial Audit (Zero Leakage)
    ├─ Config Generation (configs/qwen25vl_qlora.yaml)
    ├─ Local Structural Smoke Tests & Unit Tests (pytest)
    ├─ Packaging & Git Orchestration (branch: migration/qwen25vl-d2)
    │
    ▼ [git push / pull / notebook upload]
GOOGLE COLAB CUDA RUNTIME (NVIDIA T4 / L4 / A100)
    │
    ├─ Environment Validation (00_environment_check.py)
    ├─ Model Inspection & Token Verification (03_inspect_qwen.py)
    ├─ Real CUDA Smoke Test (04_smoke_test.py)
    ├─ Real Micro-Batch Overfit (05_overfit_microbatch.py)
    ├─ Full 4-bit QLoRA Training via TRL SFTTrainer (06_train_qwen25vl_qlora.py)
    ├─ Held-Out Evaluation on Independent Split (07_evaluate_qwen25vl.py)
    ├─ Adapter Verification & SHA-256 Checksum (08_export_adapter.py)
    ├─ Artifact Packaging & Provenance Manifest (09_package_artifacts.py)
    │
    ▼ [adapter_model.safetensors transfer]
LOCAL / ANTIGRAVITY (Apple Silicon Mac)
    │
    ├─ Cryptographic Checksum & Integrity Check
    ├─ Dual-Backend Integration Verification (specialists/single_image/specialist.py)
    ├─ Head-to-Head Benchmark vs PaliGemma (compare_paligemma_qwen.py)
    ├─ Full Regression Suite Verification
    └─ Production Promotion Gate Evaluation
```

### 1.2 Execution Classification Discipline

SatQuery AI strictly enforces transparency across execution modes:
- **`REAL-CUDA`**: Executed on a physical NVIDIA GPU under Google Colab. Performs true floating-point forward/backward passes, optimizer steps, and adapter weight generation.
- **`LOCAL-SMOKE-TEST`**: Executed on local Apple Silicon or CPU. Verifies tokenization, data collation tensor shapes, configuration parsing, and boundary conditions without full weight allocations.
- **`MOCK`**: Fast synthetic execution for pipeline integration tests and rapid UI iteration.
- **`FALLBACK`**: Explicit, non-silent fallback triggered only when requested by developers via `VISION_LANGUAGE_BACKEND=paligemma_legacy`.
- **`UNTESTED`**: Untrained or unverified configurations.

---

## 2. Model Architecture & LoRA Adaptation

### 2.1 Base Model Specifications
- **Identifier**: `Qwen/Qwen2.5-VL-3B-Instruct` (Configurable: `Qwen/Qwen2.5-VL-7B-Instruct`)
- **Visual Encoder**: ViT with Dynamic Window Attention, natively processing arbitrary aspect ratios and resolutions.
- **Visual Merger**: Multimodal MLP projection transforming spatial visual tokens into the language decoder embedding space.
- **Language Decoder**: 36 Transformer decoder layers, GQA (Grouped Query Attention), RMSNorm, SwiGLU.

### 2.2 LoRA Target Strategy
Blindly targeting standard linear names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) injects LoRA adapters into all 32 ViT blocks (696 tensors). In remote sensing, fine-tuning the vision backbone with small datasets frequently causes catastrophic representational drift.

SatQuery AI employs a targeted regular expression:
```python
target_modules_regex = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|.*merger\.mlp\.[02]"
```
**Architecture Breakdown**:
- **Vision Backbone**: **100% FROZEN** (160 blocks, 0 trainable parameters).
- **Multimodal Visual Merger**: Adapted (`merger.mlp.0`, `merger.mlp.2` — 4 tensors).
- **Language Decoder**: Adapted across all 36 layers (504 tensors).
- **Trainable Parameters**: 30,212,096 (0.80% of total 3.77B parameters).

### 2.3 Quantization (QLoRA)
- `load_in_4bit`: `True`
- `bnb_4bit_quant_type`: `"nf4"`
- `bnb_4bit_use_double_quant`: `True`
- `bnb_4bit_compute_dtype`: `torch.bfloat16` on Ampere/Ada/Hopper (A100/L4); `torch.float16` on Turing (T4).

---

## 3. Remote-Sensing Capabilities & Grounding Token Protocol

### 3.1 Grounding Syntax
Qwen2.5-VL uses native bounding box special tokens verified in the tokenizer vocabulary (`vocab_size=151665`):
- `<|object_ref_start|>` (ID 151646)
- `<|object_ref_end|>` (ID 151647)
- `<|box_start|>` (ID 151648)
- `<|box_end|>` (ID 151649)

Format:
```
<|object_ref_start|>runway<|object_ref_end|><|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
```
where $(ymin, xmin, ymax, xmax)$ are integers scaled to the interval $[0, 1000)$.

### 3.2 SAR Preprocessing Pipeline (`SARPreprocessor`)
For dual-polarization Sentinel-1 / TerraSAR-X synthetic aperture radar rasters:
1. Input: $\text{VV}$ and $\text{VH}$ backscatter amplitude bands.
2. Value sanitation: Replaces $\pm\infty$ and $\text{NaN}$ with finite boundary percentiles.
3. Stable Log-Ratio: $\text{Ratio} = \log\left(\frac{\text{VV} + \epsilon}{\text{VH} + \epsilon}\right)$ with $\epsilon = 10^{-6}$.
4. Percentile Normalization: Independent dynamic range scaling from the 1st to 99th percentile across each channel.
5. Synthesis: 3-channel RGB image $[\text{VV}, \text{VH}, \log(\text{VV}/\text{VH})]$.

### 3.3 Satellite Tiling Engine (`SatelliteTilingEngine`)
Large remote-sensing rasters ($2048 \times 2048$, $4096 \times 4096$) exceed standard VLM token capacities. The tiling engine divides large rasters into overlapping sub-tiles (e.g. $512 \times 512$ with 64px overlap), executes inference per tile, and translates local bounding boxes back to global pixel coordinates:
$$x_{\text{global}} = x_{\text{local}} + x_{\text{offset}}, \quad y_{\text{global}} = y_{\text{local}} + y_{\text{offset}}$$
Overlapping predictions are deduplicated via Non-Maximum Suppression (NMS).

---

## 4. Dataset Splits & Leakage Prevention

The curated training dataset combines high-resolution optical remote sensing and dual-polarization SAR imagery:
- **Total Samples**: 1,200
- **Modality Balance**: 80.0% Optical (960), 20.0% SAR (240)
- **Task Balance**:
  - Visual Question Answering (VQA): 421 samples (35.1%)
  - Visual Grounding: 350 samples (29.2%)
  - Detailed Captioning: 309 samples (25.8%)
  - Cross-Modal Analysis: 120 samples (10.0%)

### Spatial Leakage Audit
To eliminate train/val/test data leakage, parent scene IDs (e.g. `NH50E006001`) are extracted from each tile. All tiles from the same scene remain strictly partitioned within a single split:
- **Train Scenes**: 11 scenes (886 samples, 73.8%)
- **Validation Scenes**: 1 scene (71 samples, 5.9%)
- **Test Scenes**: 3 scenes (243 samples, 20.2%)
- **Intersection**:
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$

---

## 5. Dual-Backend Policy & Safe Rollback

To ensure continuous system stability, Division 2 supports dual backends via `SingleImageRSSpecialistTool`:
```python
# Environment configuration
VISION_LANGUAGE_BACKEND=qwen25vl          # Active modern Qwen2.5-VL backend
VISION_LANGUAGE_BACKEND=paligemma_legacy   # Preserved verified PaliGemma adapter (152075b5...)
```
- When `qwen25vl` is selected and weights are unavailable, the system **fails loudly** in strict mode, never silently falling back to PaliGemma.
- Both backends strictly conform to the identical `ToolResult` interface, `EvidenceType.BOUNDING_BOX`, and `TaskType` schemas required by Division 1 and Division 5.

---

## 6. Colab Execution Sequence (Phases C through K)

The Colab execution sequence consists of 10 modular scripts orchestrated by `colab_qwen25vl_training.ipynb`:

```bash
# Phase C: Hardware & Environment Verification
python specialists/single_image/training/colab/00_environment_check.py

# Phase D: Dataset Generation & Validation
python specialists/single_image/training/colab/01_prepare_dataset.py
python specialists/single_image/training/colab/02_validate_dataset.py

# Phase E: Model & Tokenizer Deep Inspection
python specialists/single_image/training/colab/03_inspect_qwen.py

# Phase F: Real CUDA Multimodal Smoke Test
python specialists/single_image/training/colab/04_smoke_test.py

# Phase G: Real CUDA Micro-Batch Overfit (8 samples, 25 steps)
python specialists/single_image/training/colab/05_overfit_microbatch.py

# Phase H: Production 4-bit QLoRA Training (3 epochs)
python specialists/single_image/training/colab/06_train_qwen25vl_qlora.py

# Phase I: Held-Out Split Evaluation (IoU, Recall, VQA accuracy)
python specialists/single_image/training/colab/07_evaluate_qwen25vl.py

# Phase J: Export Adapter & Compute SHA-256
python specialists/single_image/training/colab/08_export_adapter.py

# Phase K: Artifact Packaging & Provenance Manifest Generation
python specialists/single_image/training/colab/09_package_artifacts.py
```

---

## 7. Production Promotion Gate Protocol

Qwen2.5-VL will become the permanent production default only after satisfying all 16 gates:
1. `CHECKPOINT PASS`: `adapter_model.safetensors` exists with verified SHA-256.
2. `DATASET PASS`: `dataset_validation_report.json` shows 0 rejected samples.
3. `SPLIT PASS`: Spatial leakage audit confirms 0 scene overlap.
4. `REAL CUDA TRAINING PASS`: Full training completed on Colab with loss convergence.
5. `REAL ADAPTER LOAD PASS`: Local host verifies adapter file integrity and PEFT config.
6. `OPTICAL VQA PASS`: VQA questions answered accurately without format errors.
7. `OPTICAL GROUNDING PASS`: Mean IoU $\ge 0.70$ on optical targets.
8. `OPTICAL CAPTION PASS`: Rich domain captions generated without hallucinations.
9. `SAR VQA PASS`: Surface backscatter questions answered accurately.
10. `SAR GROUNDING PASS`: High-contrast structural clusters located accurately.
11. `SAR CAPTION PASS`: Radar polarimetry and radiometric terms correctly referenced.
12. `BOX PARSER PASS`: All raw tokens parsed; clean natural text delivered to users.
13. `API PASS`: FastAPI `/api/v1/query` and `/api/v1/tasks` endpoints pass 100%.
14. `NO SILENT FALLBACK PASS`: System fails loudly if backend is misconfigured.
15. `ARTIFACT PASS`: `qwen25vl_training_artifact_manifest.json` complete.
16. `REGRESSION PASS`: All 226 unit and integration tests pass green.
