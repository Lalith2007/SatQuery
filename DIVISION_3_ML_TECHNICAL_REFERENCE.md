# SatQuery AI — Division 3 ML Technical Reference & Developer Guide

**Module**: `specialists/temporal_change/` (Division 3 — Bi-Temporal Change Intelligence)  
**Model Architecture**: TinyCD (Siamese U-Net + MAMB Space-Time Attention Block)  
**Checkpoint Path**: `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`  
**License**: Non-commercial and research purposes only (Andrea Codegoni et al.).  

---

## 1. Directory Structure

```
specialists/temporal_change/
├── adaptation/
│   ├── models/
│   │   └── tinycd.py                  # TinyCD PyTorch architecture (731K params)
│   ├── dataset_loader.py              # LEVIR-CD loader with parent-scene isolation
│   ├── generate_benchmark_corpus.py   # LEVIR-CD paired corpus generator
│   ├── train_detector.py              # AdamW + BCE + SoftDice training pipeline
│   ├── smoke_test_detector.py         # Gradient smoke test & SHA-256 verification
│   └── evaluate_detector.py           # Benchmark evaluator (F1, IoU, Precision, Recall)
├── weights/
│   ├── ChangeDetector-TinyCD.pth      # Best trained neural checkpoint
│   ├── training_metrics.json          # Multi-epoch loss and convergence logs
│   ├── smoke_test_proof.json          # Mathematical backpropagation verification
│   └── smoke_test_checkpoint.pth      # Smoke test checkpoint
├── evaluation/
│   ├── benchmark_report.json          # Official test split metrics
│   ├── test_manifest.json             # Sample-by-sample test trace manifest
│   └── qualitative_panels/            # Comparison panels (T0 | T1 | GT | Pred)
├── colab/
│   ├── SatQuery_Division3_Colab_Training.ipynb # 19-step Colab GPU training notebook
│   └── reproducibility_manifest.json  # Complete reproducibility manifest
├── specialist.py                      # BiTemporalChangeSpecialistTool entry point
├── model_adapter.py                   # TinyCDAdapter & MockChangeModel
├── validation.py                      # 6-Point geospatial validation
├── preprocessing.py                   # Array resizing and normalization
├── postprocessing.py                  # Morphology & connected components
├── semantic_reasoning.py              # SpatialMetricSynthesizer
├── evidence.py                        # Standardized Evidence & Artifact packaging
└── config.py                          # Environment variable configuration
```

---

## 2. Configuration & Environment Variables

| Variable Name | Type | Default Value | Description |
|---|---|---|---|
| `SATQUERY_TC_USE_MOCK` | bool | `"true"` | When `"false"`, executes real neural TinyCD inference. |
| `SATQUERY_TC_MODEL_ARCH` | str | `"tinycd"` | Model architecture (`"tinycd"`, `"changeformer"`). |
| `SATQUERY_TC_MODEL_CHECKPOINT`| str | `"specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"` | Path to trained `.pth` state dictionary. |
| `SATQUERY_TC_CHANGE_THRESHOLD`| float| `0.5` | Probability decision boundary for change pixels. |
| `SATQUERY_TC_MIN_REGION_AREA` | int | `100` | Minimum connected component area in pixels. |
| `SATQUERY_TC_MORPH_KERNEL`    | int | `3` | Morphological ellipse structuring element size. |
| `SATQUERY_TC_MAX_REGIONS`     | int | `20` | Maximum number of bounding boxes returned. |

---

## 3. Developer Commands & Workflows

### 1. Run Gradient Smoke Test:
```bash
source .venv/bin/activate
python specialists/temporal_change/adaptation/smoke_test_detector.py
```

### 2. Run Local Training:
```bash
source .venv/bin/activate
python -c "
from specialists.temporal_change.adaptation.dataset_loader import LEVIRCDDatasetLoader
from specialists.temporal_change.adaptation.train_detector import train_tinycd

root = 'specialists/temporal_change/data/levir_cd'
train_samples = LEVIRCDDatasetLoader.discover_split_samples(root, 'train')
val_samples = LEVIRCDDatasetLoader.discover_split_samples(root, 'val')
train_tinycd(train_samples, val_samples, epochs=5, batch_size=4)
"
```

### 3. Run Benchmark Evaluation:
```bash
source .venv/bin/activate
python -c "
from specialists.temporal_change.adaptation.dataset_loader import LEVIRCDDatasetLoader
from specialists.temporal_change.adaptation.evaluate_detector import evaluate_test_split

root = 'specialists/temporal_change/data/levir_cd'
test_samples = LEVIRCDDatasetLoader.discover_split_samples(root, 'test')
evaluate_test_split(test_samples, 'specialists/temporal_change/weights/ChangeDetector-TinyCD.pth')
"
```

### 4. Run Full PyTest Suite:
```bash
source .venv/bin/activate
pytest -v --tb=short
```

---

## 4. Tensor Signatures & Model Card

- **Input Tensors**:
  - `T0`: $(B, 3, 256, 256)$ `torch.float32` normalized to $[0.0, 1.0]$.
  - `T1`: $(B, 3, 256, 256)$ `torch.float32` normalized to $[0.0, 1.0]$.
- **Output Tensor**:
  - `Probability Map`: $(B, 1, 256, 256)$ `torch.float32` spanning $[0.0, 1.0]$ via sigmoid activation.
- **Total Parameters**: $731,136$ (Trainable: $731,136$).
- **FLOPs**: $\sim 1.34\text{ GFLOPs}$ per pair.
- **Inference Latency**: $\sim 3.01\text{ ms}$ (warm median on Apple Silicon / MPS).

---

## 5. Troubleshooting & Error Codes

| Error Code / Exception | Cause | Resolution |
|---|---|---|
| `ChangeModelLoadError` | Strict mode enabled and checkpoint missing at configured path | Verify checkpoint path or set `SATQUERY_TC_USE_MOCK=true`. |
| `IncompatibleDimensionsError`| $T_0$ and $T_1$ dimensions differ without geospatial metadata | Ensure rasters are pre-aligned or enable reprojection. |
| `ImageReadError` | File not found or raster header $< 8$ bytes | Verify input paths in `ImageInput.path_or_uri`. |
