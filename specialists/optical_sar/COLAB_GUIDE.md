# Google Colab GPU Re-Training Package — Division 4 Optical-SAR Specialist

**System:** SatQuery AI  
**Division:** Division 4 — Optical-SAR Cross-Modal Joint Specialist  
**Dataset:** Complete WHU-OPT-SAR Dataset (**100 Image Pairs, ~33,000 paired 256x256 tiles**)  
**Splitting:** Image-level 70/15/15 split **BEFORE** tiling (70 train, 15 validation, 15 strictly held-out test scenes)  
**Notebook Entrypoint:** [`train_optical_sar_colab.ipynb`](../../train_optical_sar_colab.ipynb) or [`specialists/optical_sar/run_colab_training.ipynb`](run_colab_training.ipynb)  
**Training Script:** [`specialists/optical_sar/train_colab.py`](train_colab.py)  
**Candidate Checkpoint:** `specialists/optical_sar/checkpoints/cmaf_landcover_best_v2.pth`  

---

## 1. Package Overview

This package provides a complete, scientifically defensible re-training and quantitative evaluation engine for the Division 4 Optical-SAR Specialist on NVIDIA GPU hardware in Google Colab (Tesla T4, V100, or A100):

1. **Synchronized Spatial Augmentations:**
   Applies simultaneous physical transformations (random horizontal flip, random vertical flip, random 90° rotations) identically across optical, SAR, and segmentation masks.
2. **Dynamic Class Imbalance Handling:**
   Computes exact training-set pixel frequencies and applies inverse square-root class loss weights to prevent majority class collapse (`Farmland` and `Forest` dominating).
3. **Bounded Minority-Aware Tile Sampling:**
   `WeightedRandomSampler` boosts exposure to minority classes (`Water`, `Village`, `Road`, `Others`, `City`) with a bounded weight cap ($w_{\max} \le 3.5$) to prevent common class starvation.
4. **50-Epoch Differential Schedule:**
   - **Stage 1 (Warmup):** 5 epochs with frozen ResNet-50 backbones, training CMAF neck and LandCover head ($lr = 1\times 10^{-3}$).
   - **Stage 2 (Fine-Tuning):** 45 epochs unfreezing `layer3` and `layer4` of optical and SAR backbones with differential learning rates:
     - Head / Neck: $lr = 5\times 10^{-4}$
     - Backbones: $lr = 2\times 10^{-5}$
     - Cosine Annealing scheduler down to $1\times 10^{-6}$.
5. **Early Stopping & Primary Metric:**
   Validation mIoU is the primary selection metric with early stopping patience of 10 epochs.
6. **Non-Destructive Checkpointing:**
   Saves candidate model to `cmaf_landcover_best_v2.pth` without overwriting the production path.
7. **Post-Training Evaluation & Ablation:**
   Automatic evaluation on untouched 15-scene test split, Old vs New comparison, and 3-way modality ablation (Dual vs Opt-only vs SAR-only).

---

## 2. Quickstart Instructions for Google Colab

### Step 1: Open Notebook in Google Colab
Upload [`train_optical_sar_colab.ipynb`](../../train_optical_sar_colab.ipynb) or [`run_colab_training.ipynb`](run_colab_training.ipynb) to Google Colab ([colab.research.google.com](https://colab.research.google.com)).

### Step 2: Enable GPU Accelerator
In Colab:
`Runtime` $\to$ `Change runtime type` $\to$ `Hardware accelerator` $\to$ **`GPU`** (T4 / V100 / A100).

### Step 3: Execute Training Command
Run Cell 12 in the notebook, or execute from terminal:
```bash
python specialists/optical_sar/train_colab.py \
    --data_dir data/official_whu_opt_sar \
    --epochs 50 \
    --warmup_epochs 5 \
    --batch_size 16 \
    --lr_head 0.001 \
    --ft_lr_head 0.0005 \
    --ft_lr_backbone 0.00002 \
    --num_workers 2 \
    --patience 10 \
    --seed 42
```

Alternatively, run the master all-in-one script:
```bash
python specialists/optical_sar/run_all_colab.py
```

---

## 3. Architecture & Contract Preservation

- **Optical Encoder:** Pretrained ResNet-50 taking 3 RGB channels (8.54M params).
- **SAR Encoder:** Pretrained ResNet-50 taking 2 VV/VH polarization channels (8.54M params).
- **CMAF Neck:** Bidirectional spatial cross-attention fusion (2.30M params).
- **Task Head:** `LandCoverTaskHead` producing 8-class logits and probabilities (0.37M params).
- **Total Model Parameters:** 19,755,144 (exact match to production specialist).
- **Production Compatibility:** `OpticalSarSpecialist(checkpoint_path="specialists/optical_sar/checkpoints/cmaf_landcover_best_v2.pth")` loads with `strict=True` with zero missing and zero unexpected keys.
