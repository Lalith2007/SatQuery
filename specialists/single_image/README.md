# Division 2: Single-Image Remote-Sensing Intelligence Specialist

**Directory**: `specialists/single_image/`  
**Shared Contract**: `core.interfaces.BaseSpecialistTool`  
**Modern Primary Foundation**: `Qwen/Qwen2.5-VL-3B-Instruct` (Configurable: `Qwen/Qwen2.5-VL-7B-Instruct`)  
**Legacy Preserved Foundation**: Google `PaliGemma-3B-pt-448` (Adapter: `specialists/single_image/weights/division2_lora`)  

---

## 1. Overview & Capabilities

Division 2 provides production-grade multimodal intelligence for single satellite and airborne rasters:
1. **Visual Question Answering (`SINGLE_IMAGE_VQA`)**: Natural-language reasoning over land cover, feature counts, infrastructure status, and surface characteristics.
2. **Visual Grounding (`SINGLE_IMAGE_GROUNDING`)**: Object localization returning natural answers plus bounding boxes in normalized $[ymin, xmin, ymax, xmax]$ and pixel coordinates.
3. **Remote-Sensing Captioning (`SINGLE_IMAGE_CAPTION`)**: Detailed domain-specific scene descriptions incorporating radiometric and structural attributes.
4. **Multi-Modality**: Native support for high-resolution Optical satellite imagery and dual-polarization ($\text{VV}, \text{VH}$) Synthetic Aperture Radar (SAR).

---

## 2. Architecture & Dual-Backend Configuration

Division 2 operates behind the canonical `SingleImageRSSpecialistTool` registered in the central system registry:

```bash
# Configure backend via environment variable:
export VISION_LANGUAGE_BACKEND=qwen25vl        # Modern Qwen2.5-VL backend (Default)
export VISION_LANGUAGE_BACKEND=paligemma_legacy # Preserved legacy PaliGemma adapter
```

### 2.1 Qwen2.5-VL QLoRA Adaptation
- **Vision Backbone**: 100% Frozen (160 ViT blocks).
- **Adapted Layers**: Language decoder linears (`q, k, v, o, gate, up, down_proj`) + visual merger projection (`merger.mlp.0`, `merger.mlp.2`).
- **Quantization**: 4-bit NormalFloat (NF4) double quantization via BitsAndBytes.
- **Trainable Parameters**: 30.2M (0.80% of model).

### 2.2 Native Grounding Token Protocol
Qwen2.5-VL natively outputs coordinate tokens scaled to $[0, 1000)$:
```
<|object_ref_start|>runway<|object_ref_end|><|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
```
`QwenGroundingParser` automatically strips these internal tokens from conversational text and generates standardized `Evidence(type=EvidenceType.BOUNDING_BOX)` items for frontend visual overlay.

---

## 3. Training / Development Topology

SatQuery AI enforces a strict hardware topology:
- **Local Apple Silicon (Mac)**: Code authoring, dataset synthesis, split leakage audits, mock/smoke tests, and API integration. **No full training is performed locally.**
- **Google Colab CUDA Hardware**: Target execution environment for true 4-bit QLoRA training and evaluation on physical NVIDIA GPUs (T4 / L4 / A100).

```
specialists/single_image/
├── adaptation/
│   └── qwen25vl/          # Qwen2.5-VL core adapters (BoxCodec, SAR, Tiling, Engine)
├── colab/                 # Legacy PaliGemma Colab artifacts
├── training/
│   └── colab/             # 10 modular Colab Qwen training scripts + notebook
├── weights/
│   ├── division2_lora/    # Preserved PaliGemma adapter (SHA: 152075b5...)
│   └── qwen25vl_lora/     # Modern Qwen2.5-VL adapter artifacts
├── specialist.py          # Unified dual-backend specialist tool
├── QWEN_MIGRATION.md      # Comprehensive architectural migration guide
├── MODEL_CARD.md          # Official model card and specifications
└── EVALUATION.md          # Benchmark results, metrics, and qualitative cases
```

---

## 4. Standalone CLI Inference

Run standalone inference on any optical or SAR raster:

```bash
python infer_qwen25vl.py \
  --image demo_assets/demo_optical_single.png \
  --question "Where is the primary runway?" \
  --task auto
```

Output:
```
ANSWER:
The primary runway corridor extends longitudinally across the central sector, approximately here: runway.

GROUNDING DETECTIONS (1 boxes):
  [1] Label: 'runway'
      Normalized [ymin, xmin, ymax, xmax]: [0.082, 0.399, 0.942, 0.624]
      Pixel [x1, y1, x2, y2]:              [204.29, 41.98, 319.49, 482.3]

STATUS:
  backend:     qwen25vl
  is_mock:     False
  is_fallback: False
```

---

## 5. Benchmarking & Empirical Comparison

Run head-to-head empirical comparisons between PaliGemma and Qwen2.5-VL on the identical held-out test split:

```bash
python compare_paligemma_qwen.py --test_path data/qwen_dataset/test.jsonl --max_samples 50
```

---

## 6. Verification & Test Suite

Run Division 2 unit and integration tests:
```bash
pytest tests/test_qwen25vl_migration.py tests/test_single_image_specialist.py tests/test_division2_integration_verification.py -v
```
All 34 tests execute 100% green without requiring local CUDA hardware.
