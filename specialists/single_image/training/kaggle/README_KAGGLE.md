# SatQuery AI — Kaggle Production Training Guide

## 1. Problem Diagnosis: Why Did Kaggle Run Out of Disk Previously?

* **Kaggle Disk Ceiling**: Kaggle standard notebooks provide a **20 GB quota** for `/kaggle/working` and an ephemeral disk limit of **~73 GB**.
* **Archive Footprint**:
  * `BigEarthNet-S1.tar.gzaa` (48.3 GB) + `BigEarthNet-S1.tar.gzab` (6.5 GB) = **54.8 GB**
  * `BigEarthNet-S2.tar.gzaa` (48.3 GB) + `BigEarthNet-S2.tar.gzab` (15.2 GB) = **63.5 GB**
* **The Crash**: In standard download mode, downloading the 63.5 GB Sentinel-2 archive onto Kaggle's local disk alongside existing packages, system files, and extracted Sentinel-1 rasters pushed total disk usage to 100% (~73 GB), causing the kernel to crash at `Materialized 1000/7952 Sentinel-2 pairs...`.

---

## 2. The Solution: Zero-Archive Direct HTTP Streaming

The Kaggle Edition (`kaggle_qwen25vl_training.ipynb`) replaces disk-based archive downloads with **direct in-memory HTTP streaming**:

1. **0 GB Archive Storage**: `materialize_bigearthnet_kaggle.py` streams the multi-part `.tar.gz` files from Hugging Face directly into an in-memory streaming reader. **No multi-gigabyte `.tar.gz` files are ever saved to disk.**
2. **Minimal Footprint**: Only the exact 120×120 GeoTIFFs for the 8,000 required pairs are written to disk:
   * Sentinel-1 (2 bands float32, ~116 KB/pair): **~928 MB**
   * Sentinel-2 (3 bands uint16, ~86 KB/pair): **~688 MB**
   * **Total Dataset Footprint: ~1.6 GB** (leaving **~18.4 GB free** on `/kaggle/working` for training checkpoints, LoRA adapters, evaluation JSONs, and test visualizations).
3. **Resilient Provenance**: Generates identical `checksums.sha256`, `materialization_manifest.jsonl`, and enforces the exact same **Hard 8,000-Pair Materialization Gate**.

---

## 3. How to Run on Kaggle

### Step 1: Create or Import Notebook
1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Import Notebook**.
3. Select `specialists/single_image/training/kaggle/kaggle_qwen25vl_training.ipynb`.

### Step 2: Configure Kaggle Runtime Settings
* **Accelerator**: Select **GPU T4 x 2** or **GPU P100** (Settings panel on right).
* **Internet**: Turn **Internet ON** (Required to stream BigEarthNet rasters and download base model weights).
* **Persistence**: Files in `/kaggle/working` are automatically retained when you save version / commit.

### Step 3: (Optional) Set Hugging Face Token
To avoid Hugging Face rate limits on public endpoints:
1. Go to **Add-ons** -> **Secrets** in the Kaggle top menu.
2. Add a secret named `HF_TOKEN` with your Hugging Face user access token.
3. Check the box to grant access to the notebook.

### Step 4: Run
Click **Run All** (or run cells sequentially from top to bottom).
The notebook will:
1. Verify Kaggle GPU and working disk quotas (Phase A).
2. Stream and assemble all 8,000 real S1/S2 pairs in ~8-10 minutes with 0 disk overflow (Phase B).
3. Format ChatML records and partition into Train (14,304), Val (846), Test (850) with parent-granule spatial isolation (Phase C).
4. Run grounding token and model architecture inspection (Phase D).
5. Execute CUDA multimodal smoke test and micro-batch overfit (Phases E & F).
6. Train 4-bit QLoRA on the 14,304 real training records (Phase G).
7. Run zero-fallback evaluation on 850 held-out test records (Phase H).
8. Export LoRA adapter and merged model directly into `/kaggle/working/SatQueryAI_Qwen25VL` (Phases I, J, K, L).
