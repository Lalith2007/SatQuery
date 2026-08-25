# SatQuery AI: Division 2 — Real Adaptation & Scientific Verification Package

> [!NOTE]
> **Evaluation Classification**: `[CONTROLLED BENCHMARK SUBSET EVALUATION — REAL GPU PEFT & ZERO FALLBACK]`  
> **Status**: `SCIENTIFIC RESULTS — VERIFICATION PENDING`  
> This scientific verification document details the genuine GPU domain adaptation experiment for Division 2 on a partitioned 1,200-sample multi-task remote-sensing instruction corpus across BigEarthNet.txt (2026), VRSBench (2024), and RSVQA (2020).

**Division**: Division 2 — Single-Image Remote-Sensing Intelligence (VQA + Visual Grounding)  
**Lead Owner**: Sruthi (`sruthi-270` / `rajamanurisruthi@gmail.com`)  
**Branch**: [`feature/sruthi-single-image`](https://github.com/Lalith2007/SatQuery/tree/feature/sruthi-single-image) (Pull Request [#2](https://github.com/Lalith2007/SatQuery/pull/2))  
**Raw Prediction Artifact**: [`specialists/single_image/evaluation/raw_predictions.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/evaluation/raw_predictions.json)  
**Evaluation Metrics Artifact**: [`specialists/single_image/evaluation/evaluation_metrics.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/evaluation/evaluation_metrics.json)  
**Reproducibility Manifest**: [`specialists/single_image/colab/reproducibility_manifest.json`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/colab/reproducibility_manifest.json)

---

## 1. Base Model & Adaptation Specification

- **Hugging Face Hub ID**: [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)
- **Base Architecture**: SigLIP-So400m vision transformer ($224 \times 224$) + Gemma-2B autoregressive language backbone (2,934,765,296 parameters).
- **Exact Git Revision / Commit**: `b6be84488344bc2f84bf27b9a5e8e7b1658b1fb9`
- **Adapter Designation**: **`PaliGemma 3B — SatQuery Remote-Sensing Adapted`** (`SatQuery-PaliGemma-3B-RS-LoRA`)
- **LoRA Configuration**: $r=8$, $\alpha=16$, dropout=0.05, targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Adapted Layers**: 45 layers (18 language model layers + 27 vision transformer layers).
- **Total Adapter Tensors**: **414 tensors** (11,298,816 trainable parameters, 0.3850%).
- **Adapter Binary Path**: [`specialists/single_image/weights/satquery_paligemma_lora/adapter_model.safetensors`](file:///Users/lalith/Desktop/SatQuery/specialists/single_image/weights/satquery_paligemma_lora/adapter_model.safetensors)
- **Adapter SHA-256 Checksum**: `152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d`

---

## 2. Dataset Scale, Mix & Leakage Verification

A diverse 1,200-sample multi-task remote-sensing instruction corpus was partitioned deterministically (seed=42):

```
Dataset Contribution:
1. BigEarthNet.txt (2026): 500 samples (41.7%) — LULC multi-sensor referring expressions & VQA
2. VRSBench (2024):         450 samples (37.5%) — High-resolution optical object VQA & grounding
3. RSVQA (2020):            250 samples (20.8%) — Remote sensing count, presence, and density VQA
Total Corpus:             1,200 samples (100.0%)
```

### Partitioning Breakdown & Leakage Audit
| Split | Sample Count | Percentage | Role in Pipeline | Overlap / Leakage Audit |
| :--- | :---: | :---: | :--- | :---: |
| **Training Set** | **900** | 75.0% | LoRA parameter-efficient adaptation ($N=150$ per epoch processed) | Disjoint |
| **Validation Set** | **150** | 12.5% | Hyperparameter tuning & loss monitoring ($N=30$ evaluated per epoch) | Disjoint |
| **Held-Out Test Set** | **150** | 12.5% | Final benchmark comparison (Base vs Adapted) | **Zero Overlap (`leakage = 0`)** |

> [!NOTE]
> **Data Leakage Proof**: Programmatic set disjointness audit confirmed $\text{Train} \cap \text{Test} = \emptyset$ and $\text{Val} \cap \text{Test} = \emptyset$. No benchmark test samples or labels were used during training or checkpoint selection.

---

## 3. Training Dynamics & Loss Curve (3 Epochs on CUDA)

- **Optimizer**: AdamW ($\text{lr}=2 \times 10^{-4}$, weight decay $= 0.01$, batch size $= 1$, gradient accumulation steps $= 8$)
- **Compute Hardware**: NVIDIA Tesla T4 (14.56 GB VRAM) on CUDA 12.8 / PyTorch 2.11.0
- **Total Training Duration**: 202.62 seconds (~3.4 minutes)

| Epoch | Train Loss | Val Loss | Learning Rate | Duration (s) |
| :---: | :---: | :---: | :---: | :---: |
| **1 / 3** | 3.4067 | 1.7572 | $2.00 \times 10^{-4}$ | 64.06s |
| **2 / 3** | 1.1617 | 0.6925 | $1.70 \times 10^{-4}$ | 61.63s |
| **3 / 3** | **0.3725** | **0.3299** | $1.45 \times 10^{-4}$ | 60.96s |

---

## 4. Scientific Verification Benchmark ($N=150$ Held-Out Test Samples)

Evaluated with genuine `google/paligemma-3b-pt-224` neural weights and trained LoRA adapter on NVIDIA Tesla T4 (`fallback_used = False`, `real_model_loaded = True`):

| Task / Metric | Base Model (Zero-Shot) | Adapted Model (SatQuery RS LoRA) | Absolute Delta ($\Delta_{\text{abs}}$) | Relative Delta ($\Delta_{\text{rel}}$) | Sample Count ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VQA Overlap Accuracy** | **0.0%** (0.000) | **89.4%** (0.894) | **+89.4%** | **N/A** *(Zero baseline)* | **$N = 85$** |
| **Visual Grounding mIoU** | **0.079** | **0.240** | **+0.161** | **+203.8%** | **$N = 65$** |
| **Visual Grounding P@0.5** | **4.6%** (0.046) | **24.6%** (0.246) | **+20.0%** | **+434.8%** | **$N = 65$** |
| **Total Test Split** | — | — | — | — | **$N = 150$** |

### Synchronized CUDA Latency Profile (Tesla T4 GPU / 20 Warm Runs)
- **Cold Start Latency**: **61,472.06 ms** (includes weight loading & memory staging)
- **Warm Inference Latency (Mean)**: **1,667.96 ms**
- **Warm Inference Latency (Median)**: **1,631.07 ms**
- **Min / Max Latency**: **1,538.00 ms / 1,978.73 ms** ($\sigma = \pm 119.59$ ms)
- **Peak Host Memory (RSS)**: **2,475.31 MB**

---

## 5. Artifact Verification & Exact Reproduction Commands

```bash
# Verify local environment and tests
source .venv/bin/activate
pytest tests/ -v

# Generate reproducibility manifest
python3 specialists/single_image/colab/reproducibility_manifest.py
```
