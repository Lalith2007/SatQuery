# Division 4 — Training Documentation & Reproduction Guide

**Model:** SatQuery Optical-SAR Specialist Plugin (`OpticalSarSpecialist`)  
**Directory:** `specialists/optical_sar/`  
**Checkpoint Path:** `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth`  

---

## 1. Training Command & Environment

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run Stage 1 & Stage 2 Training Pipeline
python specialists/optical_sar/train.py --data_dir data/optical_sar_whu_benchmark --epochs 8 --fine_tune_epochs 2 --batch_size 8 --lr 1e-3
```

- **Python Version:** 3.11.9
- **PyTorch Version:** 2.6.0
- **Hardware:** Intel Core CPU / NVIDIA GPU (CUDA supported)

---

## 2. Dataset Specification & Leakage Prevention

- **Dataset Name:** WHU-OPT-SAR Paired Remote Sensing Benchmark Structure
- **Dataset Location:** `data/optical_sar_whu_benchmark`
- **Total Tiles:** 100 paired $256 \times 256$ rasters
- **Split Strategy:** Leakage-resistant spatial split
  - **Train:** 70 paired tiles (`data/optical_sar_whu_benchmark/train`)
  - **Val:** 15 paired tiles (`data/optical_sar_whu_benchmark/val`)
  - **Test:** 15 paired tiles (`data/optical_sar_whu_benchmark/test`)
- **Modality Channels:**
  - Optical RGB (3 channels, 8-bit reflectance)
  - SAR VV/VH Polarimetric (2 channels, 8-bit backscatter intensity)
  - Ground-Truth Mask (4 classes: `0=built_up`, `1=water`, `2=vegetation`, `3=background`)

---

## 3. Training Strategy & Phasing

### Stage 1: Frozen Encoders (Epochs 1–8)
- **Optical ResNet-50:** FROZEN (`requires_grad = False`)
- **SAR ResNet-50:** FROZEN (`requires_grad = False`)
- **Trainable Parameters:**
  - `CrossModalAttentionFusion` (`fusion_neck`): 1,182,720 parameters
  - `LandCoverTaskHead` & `FiLMQueryModulator`: 100,548 parameters
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Loss Function:** `CrossEntropyLoss()` + `0.5 * DiceLoss()`

### Stage 2: Top-Layer Encoder Fine-Tuning (Epochs 9–10)
- **Unfrozen Layers:** `layer3` of `OpticalEncoder` and `SarEncoder`
- **Optimizer:** `AdamW` with differential learning rates:
  - Fusion Neck & Task Head: `lr = 5e-4`
  - Encoder `layer3`: `lr = 1e-5`

---

## 4. Empirical Training Log & Loss Trajectory

```
2026-08-25 22:24:34 | INFO | Optical & SAR ResNet-50 Encoders: FROZEN.
2026-08-25 22:24:34 | INFO | Starting Division 4 Training for 8 epochs...
2026-08-25 22:25:02 | INFO | Epoch 01/8 [28.0s] | Train Loss: 1.2466 | Val Loss: 2.2761 | Val mIoU: 0.2343
2026-08-25 22:25:16 | INFO | Epoch 02/8 [13.7s] | Train Loss: 0.9351 | Val Loss: 1.4140 | Val mIoU: 0.6067
2026-08-25 22:25:28 | INFO | Epoch 03/8 [12.3s] | Train Loss: 0.7694 | Val Loss: 0.6545 | Val mIoU: 0.7631
2026-08-25 22:25:42 | INFO | Epoch 04/8 [13.9s] | Train Loss: 0.6332 | Val Loss: 0.5488 | Val mIoU: 0.8121
2026-08-25 22:25:57 | INFO | Epoch 05/8 [14.2s] | Train Loss: 0.5172 | Val Loss: 0.4859 | Val mIoU: 0.8453
2026-08-25 22:26:23 | INFO | Epoch 06/8 [25.8s] | Train Loss: 0.4212 | Val Loss: 0.3588 | Val mIoU: 0.8770
2026-08-25 22:26:50 | INFO | Epoch 07/8 [27.3s] | Train Loss: 0.3462 | Val Loss: 0.3215 | Val mIoU: 0.9061
2026-08-25 22:27:18 | INFO | Epoch 08/8 [27.4s] | Train Loss: 0.2877 | Val Loss: 0.2720 | Val mIoU: 0.9402
--- Starting Stage 2: Fine-Tuning Top Encoder Layers ---
2026-08-25 22:27:35 | INFO | FT Epoch 09 [17.7s] | Train Loss: 0.2561 | Val Loss: 0.2281 | Val mIoU: 0.9270
2026-08-25 22:28:00 | INFO | FT Epoch 10 [24.7s] | Train Loss: 0.2186 | Val Loss: 0.2238 | Val mIoU: 0.9447
--> Saved new BEST fine-tuned checkpoint at 'specialists/optical_sar/checkpoints/cmaf_landcover_best.pth'
```

---

## 5. Checkpoint Verification

The production service `OpticalSarSpecialist` automatically locates and loads `specialists/optical_sar/checkpoints/cmaf_landcover_best.pth` at startup:
- State dict keys verified: `optical_encoder_state_dict`, `sar_encoder_state_dict`, `fusion_neck_state_dict`, `task_head_state_dict`.
- Model placed in `eval()` mode.
- Fails clearly if checkpoint file is missing or corrupt.
