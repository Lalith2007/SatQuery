# Google Colab GPU Training Package — Division 4 Optical-SAR Specialist

**Owner:** Manoj (Division 4 Specialist Lead)  
**System:** SatQuery AI  
**Dataset Version:** Official WHU-OPT-SAR 52-Pair Development Subset (**17,160 paired 256x256 tiles**)  
**Benchmark Note:** *Development training run on 52 official pairs (17,160 tiles); not claimed as the final 100-pair benchmark.*  
**Notebook Location:** [`specialists/optical_sar/run_colab_training.ipynb`](run_colab_training.ipynb)
**Training Script:** [`specialists/optical_sar/train_colab.py`](train_colab.py)

---

## 1. Package Contents

This Google Colab GPU package provides everything needed to train the custom **CMAF Cross-Modal Attention Fusion Neck** and **8-Class LandCoverTaskHead** on GPU hardware (Tesla T4, V100, or A100):

1. **`specialists/optical_sar/train_colab.py`**: Standalone GPU training script with PyTorch Automatic Mixed Precision (AMP `cuda.amp`), 2-stage training, loss calculation (`CrossEntropyLoss` + `DiceLoss`), validation mIoU tracking, and best checkpoint saving.
2. **`specialists/optical_sar/run_colab_training.ipynb`**: Interactive Jupyter Notebook ready for Google Colab.
3. **`data/official_whu_opt_sar/`**: The 17,160 spatially aligned 256x256 paired tiles (`train`: 11,880, `val`: 2,310, `test`: 2,970).

---

## 2. Quickstart Instructions for Google Colab

### Step 1: Open Notebook in Google Colab
Upload [`run_colab_training.ipynb`](run_colab_training.ipynb) to Google Colab ([colab.research.google.com](https://colab.research.google.com)).

### Step 2: Enable GPU Accelerator
In Colab, go to:
`Runtime` $\to$ `Change runtime type` $\to$ `Hardware accelerator` $\to$ **`GPU`** (T4 / V100 / A100).

### Step 3: Run Training Command
Execute the training cell:
```bash
!python specialists/optical_sar/train_colab.py --data_dir data/official_whu_opt_sar --epochs 15 --ft_epochs 5 --batch_size 16 --lr 0.001
```

### Step 4: Run Held-Out Test Evaluation
```bash
!python specialists/optical_sar/evaluate.py --ckpt specialists/optical_sar/checkpoints/cmaf_landcover_best.pth --data_dir data/official_whu_opt_sar
```

---

## 3. Pretrained Encoders & Architecture

- **Optical Encoder:** Pretrained ResNet-50 (`torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`) taking 3 RGB spectral channels.
- **SAR Encoder:** Pretrained ResNet-50 adapted for 2 SAR polarization channels (VV/VH).
- **CMAF Neck:** Cross-Modal Attention Fusion connecting Optical and SAR ResNet-50 feature maps.
- **8-Class Task Decoder:** `LandCoverTaskHead(num_classes=8)` producing 8-class logits and probabilities.
- **Query Aggregation:** Applied **post-softmax** to map 8-class probabilities to SatQuery intents (`BUILT_UP`, `WATER`, `VEGETATION`, `OTHERS`, `BACKGROUND`).

---

## 4. Metadata Integrity Statement

Every checkpoint saved by this Colab training package explicitly embeds the following metadata:
```json
{
  "dataset_version": "Official WHU-OPT-SAR 52-Pair Development Subset (17,160 tiles)",
  "benchmark_note": "Development training run on 52 official pairs; not the final 100-pair full benchmark",
  "num_classes": 8
}
```
*This guarantees strict scientific honesty in all presentation materials.*
