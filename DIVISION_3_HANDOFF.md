# SatQuery AI — Division 3 Developer Handoff

**Division**: Division 3 — Bi-Temporal Change Intelligence  
**Author**: Dheeraj Reddy (`dheeraj-7ty` / `dheeraj12237@gmail.com`)  
**Status**: **`DIVISION_3_CLOSED`** (`COMPLETE — SCIENTIFICALLY VERIFIED`)  
**Date**: August 26, 2026  

---

### Key Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Neural Model Architecture** | **`TinyCD` (Siamese U-Net + MAMB Space-Time Attention Block)** |
| **Model Parameters** | **`3,565,034` parameters** (145 PyTorch weight tensors, 100% trainable) |
| **Checkpoint Path** | [`specialists/temporal_change/weights/ChangeDetector-TinyCD.pth`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/weights/ChangeDetector-TinyCD.pth) |
| **Checkpoint SHA-256** | `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0` |
| **Checkpoint Size** | **`14,326,623` bytes** ($13.66\text{ MB}$) |
| **Target Dataset** | **Official LEVIR-CD Benchmark** (`satellite-image-deep-learning/LEVIR-CD`) |
| **Total Benchmark Pairs** | **`985` pairs** ($256 \times 256 \times 3$) |
| **Partitioning (Zero Leakage)** | **`542` Train / `95` Validation / `348` Held-Out Test** |
| **Training Loss Recipe** | **`nn.BCELoss()` (Binary Cross-Entropy in float32)** |
| **Optimizer & Schedule** | **`AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR(T_max=20)`** |
| **Training Duration** | **20 Epochs on NVIDIA Tesla T4 GPU** ($\sim 18.0\text{ minutes}$) |
| **Best Validation $F_1$** | **`0.7952`** ($79.52\%$, Validation IoU: `0.6418`) |
| **Held-Out Test $F_1$** | **`0.6654`** ($66.54\%$, Held-Out Test IoU: `0.4986`, Precision: `0.6726`, Recall: `0.6583`, OA: `0.9730`) |
| **Inference Latency** | **`15.49 ms`** per bi-temporal pair ($\sim 64.5\text{ FPS}$ on GPU) |
| **Strict Checkpoint Load** | **`load_state_dict(strict=True)` verified** (0 missing / 0 unexpected keys) |
| **Neural Inference Mode** | **Active** (`is_mock=False`, `model_name="ChangeDetector-TinyCD"`) |
| **Mock Fallback Policy** | **Forbidden during scientific evaluation** (raises structured `ChangeModelLoadError`) |
| **Division 1 Integration** | **Verified** (`AgentController` single-step `CHANGE_VQA` $\to$ `ToolStatus.SUCCESS`) |
| **Composite Workflow** | **Verified** (`CHANGE_VQA` $\to$ `SINGLE_IMAGE_GROUNDING` with $T_1$ image) |
| **Test Suite Pass Rate** | **`130 / 130 tests passed`** (56/56 Division 3 tests passed) |

---

### Primary Code Artifacts
* Specialist Tool: [`specialists/temporal_change/specialist.py`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/specialist.py)
* Model Architecture: [`specialists/temporal_change/adaptation/models/tinycd.py`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/adaptation/models/tinycd.py)
* Model Adapter: [`specialists/temporal_change/model_adapter.py`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/model_adapter.py)
* Dataset Loader: [`specialists/temporal_change/adaptation/dataset_loader.py`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/adaptation/dataset_loader.py)
* Colab Training Notebook: [`specialists/temporal_change/colab/SatQuery_Division3_Colab_Training.ipynb`](file:///Users/lalith/Desktop/SatQuery/specialists/temporal_change/colab/SatQuery_Division3_Colab_Training.ipynb)
* Full Authoritative Status Report: [`DIVISION_3_FINAL_STATUS.md`](file:///Users/lalith/Desktop/SatQuery/DIVISION_3_FINAL_STATUS.md)
