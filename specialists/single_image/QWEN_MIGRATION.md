# SatQuery AI — Division 2: Qwen2.5-VL Foundation Migration Document

**Status**: Ready for Colab CUDA Execution (Local Validation PASS)  
**Author**: Division 2 (Single-Image Remote-Sensing Intelligence)  
**Training Runtime**: Google Colab NVIDIA CUDA Runtime (T4 / L4 / A100) — STRICTLY ENFORCED  
**Local Runtime**: Apple Silicon Mac (Development, Static Validation, Unit Tests, Config/Notebook Authoring ONLY)  

---

## 1. Executive Summary & Migration Topology

This document establishes the architecture, migration strategy, and execution protocol for migrating SatQuery AI's **Division 2 (Single-Image Remote-Sensing Intelligence Specialist)** from Google's PaliGemma-3B foundation model to Alibaba Cloud's **Qwen2.5-VL** family (`Qwen/Qwen2.5-VL-3B-Instruct` default).

> [!CRITICAL]
> **STRICT EXECUTION REQUIREMENT: GOOGLE COLAB CUDA ONLY**
> - **Actual model training MUST happen in Google Colab on physical NVIDIA CUDA GPUs.**
> - **QLoRA training is NEVER performed on the local Mac / Apple Silicon / MPS.**
> - **The local repository is ONLY for: code development, static validation, dataset structure validation, notebook generation, configuration audits, and unit tests.**
> - **Reporting Semantics & Status Integrity**:
>   - **Local execution certifies ONLY**: `IMPLEMENTATION VALIDATION = PASS`, `REAL TRAINING STATUS = NOT COMPLETE`.
>   - **Phases A–E (Structural/Implementation)**: `PASS`.
>   - **Phases F–K (Training & Verification)**: Prior to physical execution on Google Colab with CUDA:
>     - `Phase F (Full Stage 1 QLoRA Training) = NOT EXECUTED`
>     - `Phase G (Held-out Evaluation) = BLOCKED / NOT EXECUTED`
>     - `Phase H (Adapter Verification) = BLOCKED / NOT EXECUTED`
>     - `Phase I (Full Checkpoint Merge) = BLOCKED / NOT EXECUTED`
>     - `Phase J (Merged Checkpoint Independent Inference) = BLOCKED / NOT EXECUTED`
>     - `Phase K (Artifact Packaging & Google Drive Export) = BLOCKED / NOT EXECUTED`
>   - **PASS states for Phases F–K are NEVER fabricated from mocks, unit tests, schema tests, or placeholder artifacts.**
>   - **Only after certified REAL-CUDA execution in Colab does the report state**:
>     - `REAL-CUDA TRAINING = PASS`
>     - `ADAPTER = PASS`
>     - `MERGED FULL CHECKPOINT = PASS`
>     - `INDEPENDENT INFERENCE = PASS`
>     - `CHECKPOINT INTEGRITY = PASS`
>     - `PERSISTENT ARTIFACT = PASS`
> - **Deliverables: Produces BOTH a LoRA adapter (`adapter/`) AND a complete standalone merged model checkpoint (`merged_full/`) stored in safetensors format.**
> - **The merged model is the PRIMARY deployment artifact and must be loadable independently without PEFT.**
> - **Multi-GB model weights are stored outside Git** (in persistent Google Drive `/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/` or optional Hugging Face Hub). Git stores code, configs, manifests, documentation, and SHA-256 hashes.

### 1.1 Development and Execution Topology

```
LOCAL / ANTIGRAVITY (Apple Silicon Mac)
    │
    ├─ Code Authoring & Model Abstraction
    ├─ Stage 1 BigEarthNet Manifest Audit (8,000 pairs, 16,000 examples)
    ├─ Sensor Conversion Audit (S1 SAR: VV, VH, VV-VH; S2 MSI: B04, B03, B02)
    ├─ Config Generation (configs/qwen25vl_qlora.yaml)
    ├─ Colab Notebook Generation (colab_qwen25vl_training.ipynb)
    ├─ Local Unit & Structural Tests (pytest)
    │
    ▼ [git push / notebook upload to Google Colab]
GOOGLE COLAB CUDA RUNTIME (NVIDIA T4 / L4 / A100)
    │
    ├─ Phase A: Environment Check (Strict CUDA Guard) (00_environment_check.py)
    ├─ Phase B: Stage 1 Dataset Validation Gate (01_prepare_dataset.py, 02_validate_dataset.py)
    ├─ Phase C: Architecture & Grounding Token Inspection (03_inspect_qwen.py)
    ├─ Phase D: Real CUDA Smoke Test (04_smoke_test.py)
    ├─ Phase E: Micro-Batch Overfit Convergence Test (05_overfit_microbatch.py)
    ├─ Phase F: Production 4-bit QLoRA Training (06_train_qwen25vl_qlora.py)
    ├─ Phase G: Held-Out Authoritative Evaluation (07_evaluate_qwen25vl.py)
    ├─ Phase H: Adapter Verification & SHA-256 Calculation (08_export_adapter.py)
    ├─ Phase I: Full Checkpoint Merging (Safetensors Shards) (08_export_adapter.py)
    ├─ Phase J: Standalone Merged Independent Inference Validation (08_export_adapter.py)
    └─ Phase K: Packaging, SHA-256 Manifest, Archive & Google Drive Export (09_package_artifacts.py)
    │
    ▼ [Persistent Artifact Export]
GOOGLE DRIVE PERSISTENCE & REPRODUCIBILITY ARCHIVE
    /content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/
    ├── adapter/ (adapter_model.safetensors, adapter_config.json)
    ├── merged_full/ (config.json, preprocessor_config.json, model.safetensors shards)
    ├── evaluation/ (results.json, results.md)
    ├── checkpoint_manifest.json (SHA-256 checksums across all shards)
    ├── CHECKPOINT_CARD.md
    └── qwen25vl_stage1_full_checkpoint.tar.zst
```

### 1.2 Execution Classification Discipline

SatQuery AI strictly enforces transparency across execution modes:
- **`REAL-CUDA`**: Executed on a physical NVIDIA GPU under Google Colab. Performs true floating-point forward/backward passes, optimizer steps, checkpoint merging, and verified weight generation.
- **`LOCAL-SMOKE-TEST`**: Executed on local Apple Silicon or CPU. Verifies tokenization, data collation tensor shapes, configuration parsing, and boundary conditions without full weight allocations.
- **`MOCK`**: Fast synthetic execution for pipeline integration tests and rapid UI iteration.
- **`FALLBACK`**: Explicit, non-silent fallback triggered only when requested by developers via `paligemma_legacy`. If Qwen is selected and fails, it FAILS LOUDLY; it never silently falls back to PaliGemma.
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

## 6. Colab Execution Sequence (Mandatory Phases A through K)

The Colab execution sequence executes all 11 phases in sequential order via `specialists/single_image/training/colab/colab_qwen25vl_training.ipynb` (or individual modules):

```bash
# PHASE A — Environment Diagnostics & Hardware Verification (Strict CUDA Guard)
python specialists/single_image/training/colab/00_environment_check.py

# PHASE B — Stage 1 BigEarthNet Dataset Preparation & Strict Validation Gate
python specialists/single_image/training/colab/01_prepare_dataset.py --manifest_path data/curated_mixture/bigearthnet_stage1_manifest.jsonl
python specialists/single_image/training/colab/02_validate_dataset.py --manifest_path data/curated_mixture/bigearthnet_stage1_manifest.jsonl

# PHASE C — Qwen2.5-VL Architecture & Grounding Token Inspection
python specialists/single_image/training/colab/03_inspect_qwen.py --model_id Qwen/Qwen2.5-VL-3B-Instruct

# PHASE D — Real CUDA Multimodal Smoke Test
python specialists/single_image/training/colab/04_smoke_test.py --model_id Qwen/Qwen2.5-VL-3B-Instruct

# PHASE E — Micro-Batch Overfit Convergence Test (8 samples, 25 steps)
python specialists/single_image/training/colab/05_overfit_microbatch.py --model_id Qwen/Qwen2.5-VL-3B-Instruct --steps 25

# PHASE F — Full Stage 1 QLoRA Production Training (16,000 examples, 3 epochs)
python specialists/single_image/training/colab/06_train_qwen25vl_qlora.py --config configs/qwen25vl_qlora.yaml

# PHASE G — Held-Out Authoritative Evaluation (Visual Grounding IoU, VQA Accuracy, Captioning)
python specialists/single_image/training/colab/07_evaluate_qwen25vl.py --test_file data/qwen_dataset/test.jsonl

# PHASE H — LoRA Adapter Verification & SHA-256 Calculation
python specialists/single_image/training/colab/08_export_adapter.py --skip_merge

# PHASE I — Full Checkpoint Merging (Base Model + LoRA -> Standalone Safetensors Shards)
python specialists/single_image/training/colab/08_export_adapter.py --base_model Qwen/Qwen2.5-VL-3B-Instruct --merged_dir artifacts/qwen25vl_stage1/merged_full

# PHASE J — Checkpoint Integrity & Standalone Independent Inference Validation (NO PEFT Loaded)
python specialists/single_image/training/colab/08_export_adapter.py --validate_only --merged_dir artifacts/qwen25vl_stage1/merged_full

# PHASE K — Artifact Packaging, Comprehensive Manifest, tar.zst Archive & Google Drive Export
python specialists/single_image/training/colab/09_package_artifacts.py --google_drive_dir /content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run
```

---

## 7. Production Promotion Gate Protocol

Qwen2.5-VL will become the permanent production default only after satisfying all 16 gates:
1. `CHECKPOINT PASS`: `adapter_model.safetensors` and `merged_full/*.safetensors` exist with verified SHA-256.
2. `DATASET PASS`: `dataset_validation_report.json` shows 0 rejected samples on 16,000 records / 8,000 pairs.
3. `SPLIT PASS`: Spatial leakage audit confirms 0 parent-granule overlap.
4. `REAL CUDA TRAINING PASS`: Full training completed on Colab with loss convergence (`EXECUTION_MODE=REAL-CUDA`).
5. `STANDALONE MERGE PASS`: Merged checkpoint verified via independent inference without PEFT.
6. `OPTICAL VQA PASS`: VQA questions answered accurately without format errors.
7. `OPTICAL GROUNDING PASS`: Mean IoU $\ge 0.70$ on optical targets.
8. `OPTICAL CAPTION PASS`: Rich domain captions generated without hallucinations.
9. `SAR VQA PASS`: Surface backscatter questions answered accurately.
10. `SAR GROUNDING PASS`: High-contrast structural clusters located accurately.
11. `SAR CAPTION PASS`: Radar polarimetry and radiometric terms correctly referenced.
12. `BOX PARSER PASS`: All raw tokens parsed; clean natural text delivered to users.
13. `API PASS`: FastAPI `/api/v1/query` and `/api/v1/tasks` endpoints pass 100%.
14. `NO SILENT FALLBACK PASS`: System fails loudly if backend is misconfigured.
15. `PERSISTENCE PASS`: Verified Google Drive export to `/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/`.
16. `REGRESSION PASS`: All unit and integration tests pass green.
