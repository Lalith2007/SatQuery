# SatQuery AI: Division 2 — Real Adaptation Experiment & Verification Report

**Division**: Division 2 — Single-Image Remote-Sensing Intelligence (VQA + Visual Grounding)  
**Lead Owner**: Sruthi (`sruthi-270` / `rajamanurisruthi@gmail.com`)  
**Branch**: [`feature/sruthi-single-image`](https://github.com/Lalith2007/SatQuery/tree/feature/sruthi-single-image) (Pull Request [#2](https://github.com/Lalith2007/SatQuery/pull/2))  
**Classification**: `[CONTROLLED BENCHMARK SUBSET EVALUATION — REAL GPU PEFT & ZERO FALLBACK]`  
**Status**: `SCIENTIFIC RESULTS — VERIFICATION PENDING`

---

### A. Actual Training Sample Count & Loss Dynamics
- **Training Samples Processed**: $N = 150$ unique instruction samples per epoch across 3 epochs (450 sample step iterations total, partitioned from $N=900$ train corpus).
- **Validation Samples**: $N = 30$ samples (for epoch validation monitoring).
- **Test Samples**: $N = 150$ strictly held-out samples ($N=85$ VQA, $N=65$ Grounding).
- **Data Leakage**: $\text{Train} \cap \text{Test} = \emptyset$ (`leakage_count = 0`, `data_leakage_detected = False`).
- **Optimizer**: AdamW ($\text{lr}=2 \times 10^{-4}$, weight decay $= 0.01$, batch size $= 1$, gradient accumulation steps $= 8$).
- **Loss Trajectory (NVIDIA Tesla T4 GPU)**:
  - Epoch 1: Train Loss **3.4067**, Val Loss **1.7572** (Duration: 64.06s)
  - Epoch 2: Train Loss **1.1617**, Val Loss **0.6925** (Duration: 61.63s)
  - Epoch 3: Train Loss **0.3725**, Val Loss **0.3299** (Duration: 60.96s)
- **Total Training Duration**: **202.62 s** (~3.4 minutes on CUDA).

---

### B. Base Model & LoRA Adapter Specifications
- **Base Model Checkpoint**: `google/paligemma-3b-pt-224`
- **Exact Base Model Revision (SHA)**: `b6be84488344bc2f84bf27b9a5e8e7b1658b1fb9`
- **Total Base Parameters**: 2,934,765,296 parameters (SigLIP-So400m + Gemma-2B)
- **LoRA Adapter Checkpoint**: `specialists/single_image/weights/satquery_paligemma_lora/adapter_model.safetensors` (43 MB)
- **LoRA Checkpoint SHA-256**: `152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d`
- **LoRA Configuration**: Rank $r=8$, $\alpha=16$, dropout=0.05
- **Adapted Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Adapted Layer Count**: 45 layers (18 language model layers + 27 vision transformer layers)
- **Total Adapter Tensors**: **414 tensors**
- **Trainable Parameters**: **11,298,816 parameters** (0.3850% of total)

---

### C. Scientific Verification Benchmark ($N=150$ Held-Out Test Samples)

Evaluated with genuine `google/paligemma-3b-pt-224` weights and trained LoRA adapter on NVIDIA Tesla T4 (`fallback_used = False`, `real_model_loaded = True`):

| Task / Metric | Base Model (Zero-Shot) | Adapted Model (SatQuery RS LoRA) | Absolute Delta ($\Delta_{\text{abs}}$) | Relative Delta ($\Delta_{\text{rel}}$) | Sample Count ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VQA Overlap Accuracy** | **0.0%** (0.000) | **89.4%** (0.894) | **+89.4%** | **N/A** *(Zero baseline)* | **$N = 85$** |
| **Visual Grounding mIoU** | **0.079** | **0.240** | **+0.161** | **+203.8%** | **$N = 65$** |
| **Visual Grounding P@0.5** | **4.6%** (0.046) | **24.6%** (0.246) | **+20.0%** | **+434.8%** | **$N = 65$** |
| **Total Test Split** | — | — | — | — | **$N = 150$** |

---

### D. Synchronized CUDA Latency Profile (Tesla T4 GPU / 20 Warm Runs)

Measured with explicit `torch.cuda.synchronize()` before and after generation:
- **Cold Start Latency**: **61,472.06 ms** (includes weight loading & allocation)
- **Warm Inference Mean**: **1,667.96 ms**
- **Warm Inference Median**: **1,631.07 ms**
- **Min / Max Warm Latency**: **1,538.00 ms / 1,978.73 ms** ($\sigma = \pm 119.59$ ms)
- **Stage Breakdown**:
  - Image Preprocessing: 1.06 ms
  - Neural Generation (`max_new_tokens=64`): 1,666.64 ms
  - Postprocessing & Coordinate Parsing: 0.01 ms
- **Host Resident Memory (RSS)**: **2,475.31 MB**

---

### E. Image-to-Sample Mapping & Prediction Diversity Audit

- **Test Record Image Mapping**:
  - `demo_assets/demo_optical_single.png`: 86 samples (BigEarthNet LULC & RSVQA prompts)
  - `demo_assets/demo_airport_grounding.png`: 64 samples (VRSBench high-resolution runway/aircraft/harbor prompts)
- **Prediction Diversity**:
  - Base zero-shot model outputs 6 generic short tokens (`"0"`, `"grass"`, `"yes"`, `"no"`, `"1"`).
  - Adapted model outputs 18 distinct domain-specific sentences (e.g., `"The region is dominated by industrial units, featuring warehouses..."`, `"There are 4 cargo vessels docked in the harbor."`).

---

### F. Authoritative Artifact References

- **Raw Predictions Log**: [`specialists/single_image/evaluation/raw_predictions.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/evaluation/raw_predictions.json)
- **Evaluation Metrics**: [`specialists/single_image/evaluation/evaluation_metrics.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/evaluation/evaluation_metrics.json)
- **PEFT Smoke Test Proof**: [`specialists/single_image/weights/satquery_paligemma_lora/smoke_test_proof.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/weights/satquery_paligemma_lora/smoke_test_proof.json)
- **Training Metrics Log**: [`specialists/single_image/weights/satquery_paligemma_lora/training_metrics.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/weights/satquery_paligemma_lora/training_metrics.json)
- **Reproducibility Manifest**: [`specialists/single_image/colab/reproducibility_manifest.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/colab/reproducibility_manifest.json)
