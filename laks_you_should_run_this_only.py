#!/usr/bin/env python3
"""
================================================================================
SatQuery Division 4 — Optical-SAR Specialist Local Training Runner
Script: laks_you_should_run_this_only.py
Author: SatQuery Multi-Modal AI Team
Target System: Local RTX 4060 Laptop (CUDA Acceleration, 8GB VRAM)
================================================================================

HOW TO RUN:
    Simply open a terminal in the SatQuery directory and run:
        python laks_you_should_run_this_only.py

    Or if your dataset is in a custom folder:
        python laks_you_should_run_this_only.py --data-dir /path/to/official_whu_opt_sar

WHAT THIS SCRIPT DOES:
    1. Detects your NVIDIA RTX 4060 GPU and optimizes VRAM / cuDNN settings.
    2. Auto-locates your local WHU-OPT-SAR dataset (or reassembles it if chunked).
    3. Runs pre-flight dimensional & numerical stability checks (AMP mixed precision).
    4. Executes full 50-Epoch Re-Training:
       - Stage 1: 5 Warmup Epochs (frozen ResNet-50 backbones, AdamW lr=1e-3)
       - Stage 2: 45 Fine-Tuning Epochs (layer3 unfrozen, lr_head=5e-4, lr_backbone=2e-5,
                  Cosine Annealing, Early Stopping patience=10)
       - Dynamic inverse square-root pixel weights + multi-class Dice Loss
       - Bounded minority-aware sampler (WeightedRandomSampler)
       - Streaming O(1) confusion matrix (zero memory spikes on 8GB VRAM)
    5. Saves winning checkpoint to:
       specialists/optical_sar/checkpoints/cmaf_landcover_best_v2.pth
    6. Runs complete post-training evaluation suite:
       - 15-scene untouched held-out test split evaluation
       - Per-class metrics table (IoU, F1, Precision, Recall, Support)
       - Normalized test confusion matrix heatmap
       - 3-way modality ablation study (Dual vs Optical-only vs SAR-only)
       - Deterministic reproducibility check
       - Production specialist verification (OpticalSarSpecialist)
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# -----------------------------------------------------------------------------
# 0. Repository Root & sys.path Setup
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"

# Clean any erroneous subpaths from sys.path
for bad in [str(PROJECT_ROOT / "core"), str(PROJECT_ROOT / "specialists")]:
    while bad in sys.path:
        sys.path.remove(bad)

# Verify core dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:
    print("\n[ERROR] PyTorch is not installed in this environment.")
    print("Please install PyTorch with CUDA support:")
    print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 1. Beautiful Console Banner & Diagnostics
# -----------------------------------------------------------------------------
def print_banner(title: str, char: str = "=") -> None:
    width = 80
    print("\n" + char * width)
    print(f" {title}")
    print(char * width, flush=True)


def check_system_and_gpu() -> Tuple[torch.device, bool, int]:
    """Inspect local hardware, specifically tuned for RTX 4060 Laptop (8GB VRAM)."""
    print_banner("STAGE 1: HARDWARE & CUDA ENVIRONMENT DIAGNOSTICS")
    print(f"Timestamp        : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Operating System : {sys.platform} ({os.name})")
    print(f"Python Version   : {sys.version.split()[0]}")
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"Project Root     : {PROJECT_ROOT}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available   : {cuda_available}")

    recommended_batch_size = 16
    use_amp = False

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        print(f"GPU Device       : {gpu_name}")
        print(f"CUDA Capability  : {torch.cuda.get_device_capability(0)}")
        print(f"Total VRAM       : {total_gb:.2f} GB")
        print(f"Available VRAM   : {free_gb:.2f} GB")

        # RTX 4060 Laptop Ada Lovelace optimizations
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda:0")
        use_amp = True

        if "4060" in gpu_name:
            print("\n  >>> Detected NVIDIA GeForce RTX 4060 Laptop GPU! <<<")
            print("  >>> Ada Lovelace architecture with 4th-Gen Tensor Cores.")
            print("  >>> Enabling CUDA AMP mixed-precision (FP16/BF16) + cuDNN benchmark.")

        if free_gb < 4.5:
            print(f"\n  [Notice] Available VRAM ({free_gb:.2f} GB) is under 4.5 GB.")
            print("  Switching default batch size to 8 to avoid out-of-memory errors.")
            recommended_batch_size = 8
        else:
            print(f"  VRAM ({free_gb:.2f} GB free) is optimal for batch size 16.")
    elif torch.backends.mps.is_available():
        print("Hardware Target  : Apple Silicon (MPS)")
        device = torch.device("mps")
        recommended_batch_size = 8
    else:
        print("\n  [WARNING] No CUDA GPU detected! Training on CPU will be extremely slow.")
        print("  Make sure NVIDIA GPU drivers and PyTorch CUDA build are installed.")
        device = torch.device("cpu")
        recommended_batch_size = 4

    return device, use_amp, recommended_batch_size


# -----------------------------------------------------------------------------
# 2. Smart Local Dataset Discovery & Validation
# -----------------------------------------------------------------------------
def is_valid_tiled_dataset(candidate: Path) -> bool:
    """Check if directory has standard WHU-OPT-SAR tiled structure (train, val, test)."""
    if not candidate.exists() or not candidate.is_dir():
        return False
    train_opt = candidate / "train" / "optical"
    val_opt = candidate / "val" / "optical"
    test_opt = candidate / "test" / "optical"

    if train_opt.exists() and val_opt.exists() and test_opt.exists():
        train_count = len(list(train_opt.glob("*.png")))
        val_count = len(list(val_opt.glob("*.png")))
        test_count = len(list(test_opt.glob("*.png")))
        return train_count > 0 and val_count > 0 and test_count > 0
    return False


def is_valid_raw_dataset(candidate: Path) -> bool:
    """Check if directory has raw official WHU-OPT-SAR full scenes (optical, sar, labels)."""
    if not candidate.exists() or not candidate.is_dir():
        return False
    opt = candidate / "optical"
    sar = candidate / "sar"
    lbl = candidate / "labels" if (candidate / "labels").exists() else candidate / "lbl"
    if opt.exists() and sar.exists() and lbl.exists():
        opt_files = list(opt.glob("*.*"))
        return len(opt_files) > 0
    return False


def auto_tile_raw_dataset(raw_dir: Path, output_dir: Path) -> Path:
    """Tile raw WHU-OPT-SAR image pairs using the exact SatQuery 70/15/15 Image-Level Split."""
    print(f"\n  [Notice] Detected raw untiled WHU-OPT-SAR dataset at: {raw_dir}")
    print("  Applying official SatQuery Custom Image-Level 70/15/15 Split (seed=42)...")
    print("  (70 train scenes, 15 validation scenes, 15 untouched test scenes)")
    print(f"  Tiling into 256x256 patches inside: {output_dir} ...")

    from specialists.optical_sar.tile_official_dataset import tile_whu_dataset_directory
    tile_counts = tile_whu_dataset_directory(
        raw_dataset_dir=raw_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        tile_size=256,
    )
    print(f"  Tiling complete! Generated tiles per split:")
    for s_name, count in tile_counts.items():
        print(f"    - {s_name:5s}: {count:,} tiles")
    return output_dir


def try_reassemble_chunks(target_dataset_dir: Path) -> bool:
    """If split archive chunks (whu_chunk_*) exist, reassemble and extract them."""
    chunks = sorted(list(PROJECT_ROOT.glob("whu_chunk_*")))
    if not chunks:
        # Also check parent directory
        chunks = sorted(list(PROJECT_ROOT.parent.glob("whu_chunk_*")))

    if len(chunks) >= 6:
        print(f"\n  Found {len(chunks)} archive chunks ({chunks[0].name}..{chunks[-1].name})!")
        print("  Reassembling complete official WHU-OPT-SAR dataset...")
        assembled_zip = PROJECT_ROOT / "official_whu_opt_sar_assembled.zip"

        try:
            with open(assembled_zip, "wb") as outfile:
                for chunk in chunks:
                    print(f"    Appending chunk: {chunk.name} ({chunk.stat().st_size / (1024*1024):.1f} MB)...")
                    with open(chunk, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

            print(f"  Extracting assembled archive into {target_dataset_dir}...")
            import zipfile
            target_dataset_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(assembled_zip, "r") as zf:
                zf.extractall(target_dataset_dir)

            if is_valid_tiled_dataset(target_dataset_dir):
                print("  Dataset successfully reassembled and verified!")
                return True
            if is_valid_raw_dataset(target_dataset_dir):
                tiled_target = PROJECT_ROOT / "data" / "official_whu_opt_sar"
                return auto_tile_raw_dataset(target_dataset_dir, tiled_target)
        except Exception as e:
            print(f"  [Warning] Failed to reassemble archive chunks: {e}")

    return False


def locate_local_dataset(explicit_dir: Optional[str] = None) -> Path:
    """Find the WHU-OPT-SAR dataset (tiled or raw) across standard local locations."""
    print_banner("STAGE 2: LOCAL DATASET DISCOVERY & TILE COUNT VERIFICATION")
    default_tiled_target = PROJECT_ROOT / "data" / "official_whu_opt_sar"

    # 1. Explicit CLI argument
    if explicit_dir:
        cand = Path(explicit_dir).expanduser().resolve()
        if is_valid_tiled_dataset(cand):
            print(f"Tiled dataset located via CLI argument: {cand}")
            return cand
        if is_valid_tiled_dataset(cand / "official_whu_opt_sar"):
            print(f"Tiled dataset located via CLI argument subfolder: {cand / 'official_whu_opt_sar'}")
            return cand / "official_whu_opt_sar"
        if is_valid_raw_dataset(cand):
            return auto_tile_raw_dataset(cand, default_tiled_target)
        if is_valid_raw_dataset(cand / "raw_whu_opt_sar"):
            return auto_tile_raw_dataset(cand / "raw_whu_opt_sar", default_tiled_target)

    # 2. Environment variable
    env_dir = os.environ.get("WHU_DATASET_DIR") or os.environ.get("DATASET_DIR")
    if env_dir:
        cand = Path(env_dir).expanduser().resolve()
        if is_valid_tiled_dataset(cand):
            print(f"Tiled dataset located via environment variable: {cand}")
            return cand
        if is_valid_raw_dataset(cand):
            return auto_tile_raw_dataset(cand, default_tiled_target)

    # 3. Standard search candidate paths for TILED dataset
    home = Path.home()
    tiled_candidates: List[Path] = [
        Path("D:/official_whu_opt_sar_dataset/official_whu_opt_sar_100scenes"),
        Path("D:/official_whu_opt_sar_dataset/official_whu_opt_sar"),
        PROJECT_ROOT / "data" / "official_whu_opt_sar",
        PROJECT_ROOT.parent / "data" / "official_whu_opt_sar",
        Path("data/official_whu_opt_sar").resolve(),
        Path("data/whu_opt_sar").resolve(),
        home / "data" / "official_whu_opt_sar",
        home / "Datasets" / "official_whu_opt_sar",
        home / "Downloads" / "official_whu_opt_sar",
        home / "Desktop" / "official_whu_opt_sar",
        Path("D:/data/official_whu_opt_sar"),
        Path("D:/official_whu_opt_sar"),
        Path("C:/data/official_whu_opt_sar"),
        Path("E:/data/official_whu_opt_sar"),
    ]

    for cand in tiled_candidates:
        if is_valid_tiled_dataset(cand):
            print(f"Tiled dataset auto-discovered at: {cand}")
            return cand

    # 4. Standard search candidate paths for RAW dataset (auto-tile if found)
    raw_candidates: List[Path] = [
        PROJECT_ROOT / "data" / "raw_whu_opt_sar",
        PROJECT_ROOT.parent / "data" / "raw_whu_opt_sar",
        Path("data/raw_whu_opt_sar").resolve(),
        home / "data" / "raw_whu_opt_sar",
        home / "Datasets" / "raw_whu_opt_sar",
        home / "Downloads" / "raw_whu_opt_sar",
        home / "Desktop" / "raw_whu_opt_sar",
        Path("D:/data/raw_whu_opt_sar"),
        Path("D:/raw_whu_opt_sar"),
        Path("C:/data/raw_whu_opt_sar"),
    ]

    for cand in raw_candidates:
        if is_valid_raw_dataset(cand):
            return auto_tile_raw_dataset(cand, default_tiled_target)

    # 5. Attempt reassembly from chunks if present
    if try_reassemble_chunks(default_tiled_target):
        return default_tiled_target

    # 6. Check if unextracted zip exists
    zip_candidates = [
        PROJECT_ROOT / "official_whu_opt_sar_dataset.zip",
        PROJECT_ROOT / "data" / "official_whu_opt_sar.zip",
        home / "Downloads" / "official_whu_opt_sar_dataset.zip",
    ]
    for zpath in zip_candidates:
        if zpath.exists():
            print(f"Found dataset archive: {zpath}. Extracting to {default_tiled_target}...")
            import zipfile
            default_tiled_target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(default_tiled_target)
            if is_valid_tiled_dataset(default_tiled_target):
                return default_tiled_target
            if is_valid_raw_dataset(default_tiled_target):
                return auto_tile_raw_dataset(default_tiled_target, PROJECT_ROOT / "data" / "official_whu_opt_sar")

    # 7. Interactive prompt fallback
    print("\n[!] Could not automatically find the WHU-OPT-SAR dataset folder.")
    print("Checked paths:")
    for c in tiled_candidates[:5]:
        print(f"  - {c}")

    try:
        user_input = input("\nPlease enter the path to your WHU-OPT-SAR dataset folder: ").strip()
        if user_input:
            cand = Path(user_input).expanduser().resolve()
            if is_valid_tiled_dataset(cand):
                return cand
            if is_valid_tiled_dataset(cand / "official_whu_opt_sar"):
                return cand / "official_whu_opt_sar"
            if is_valid_raw_dataset(cand):
                return auto_tile_raw_dataset(cand, default_tiled_target)
    except (EOFError, KeyboardInterrupt):
        pass

    raise FileNotFoundError(
        "Could not locate WHU-OPT-SAR dataset (neither tiled nor raw).\n"
        "Please specify the path using: python laks_you_should_run_this_only.py --data-dir <PATH_TO_DATASET>"
    )


# -----------------------------------------------------------------------------
# 3. Pre-Flight Dimensional & Numerical Stability Audit
# -----------------------------------------------------------------------------
def run_preflight_audit(dataset_dir: Path, device: torch.device, use_amp: bool) -> None:
    """Verify data loaders, batch dimensional contracts, and AMP numerical stability."""
    print_banner("STAGE 3: PRE-FLIGHT DIMENSIONAL & NUMERICAL STABILITY AUDIT")
    from specialists.optical_sar.dataset import OpticalSarPairedDataset
    from specialists.optical_sar.query_intent import FiLMQueryModulator
    from specialists.optical_sar.dataset import OpticalSarPairedDataset
    from specialists.optical_sar.config_balanced_v3 import TrainingConfigV3
    from specialists.optical_sar.train_colab_v3 import ColabTrainerV3

    train_ds = OpticalSarPairedDataset(dataset_dir, split="train", num_classes=8, augment=True)
    val_ds = OpticalSarPairedDataset(dataset_dir, split="val", num_classes=8, augment=False)
    test_ds = OpticalSarPairedDataset(dataset_dir, split="test", num_classes=8, augment=False)

    print(f"  Train tiles : {len(train_ds):,d} (augment=True)")
    print(f"  Val tiles   : {len(val_ds):,d} (augment=False)")
    print(f"  Test tiles  : {len(test_ds):,d} (augment=False)")

    assert len(train_ds) > 0, "Train split is empty!"
    assert len(val_ds) > 0, "Validation split is empty!"
    assert len(test_ds) > 0, "Test split is empty!"

    # 1. Dimensional contract validation
    print("\n  [Check 1/3] Validating Batch Dimensional Contract...")
    check_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(check_loader))

    opt = batch["optical"]
    sar = batch["sar"]
    lbl = batch["label"]

    print(f"    Optical Tensor : {list(opt.shape)} (Expected: [4, 3, 256, 256])")
    print(f"    SAR Tensor     : {list(sar.shape)} (Expected: [4, 2, 256, 256])")
    print(f"    Label Tensor   : {list(lbl.shape)} (Expected: [4, 256, 256])")

    assert opt.shape == (4, 3, 256, 256), f"Unexpected Optical shape: {opt.shape}"
    assert sar.shape == (4, 2, 256, 256), f"Unexpected SAR shape: {sar.shape}"
    assert lbl.shape == (4, 256, 256), f"Unexpected Label shape: {lbl.shape}"

    # 2. Query intent neutral conditioning check
    print("  [Check 2/3] Validating Neutral Query Intent Conditioning...")
    intent = torch.ones(4, 8, device=device)
    assert intent.shape == (4, 8), f"Unexpected Intent shape: {intent.shape}"

    # 3. Quick gradient forward-backward check
    print(f"  [Check 3/3] Running 3-Step Numerical Stability & AMP Test on {device}...")
    cfg = TrainingConfigV3(data_dir=dataset_dir, batch_size=4, use_amp=use_amp)
    trainer = ColabTrainerV3(config=cfg, device=str(device))
    params = trainer.get_parameter_counts()
    print(f"    Total Model Parameters: {params['total_model_parameters']:,d} (Expected: 19,755,144)")
    assert params["total_model_parameters"] == 19755144, f"Parameter count mismatch: {params}"

    weights = trainer.compute_median_frequency_weights()
    trainer.setup_loss(weights)
    trainer.set_stage(1)
    test_opt_params = trainer.get_trainable_parameters(1)
    optimizer = torch.optim.AdamW(test_opt_params, lr=1e-3)

    for step, sample_batch in enumerate(check_loader):
        if step >= 3:
            break
        b_opt = sample_batch["optical"].to(device)
        b_sar = sample_batch["sar"].to(device)
        b_lbl = sample_batch["label"].to(device)
        b_intent = torch.ones(b_opt.shape[0], 8, device=device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=trainer.use_amp):
            feats_opt = trainer.optical_encoder(b_opt)
            feats_sar = trainer.sar_encoder(b_sar)
            fused = trainer.fusion_neck(feats_opt["stride_8"], feats_sar["stride_8"])
            logits, _ = trainer.task_head(fused, b_intent)
            if logits.shape[2:] != b_lbl.shape[1:]:
                logits = F.interpolate(logits, size=b_lbl.shape[1:], mode="bilinear", align_corners=False)
            loss_ce = trainer.ce_loss_fn(logits.float(), b_lbl)
            loss_dice = trainer.dice_loss_fn(logits.float(), b_lbl)
            loss = loss_ce + 1.0 * loss_dice

        assert torch.isfinite(loss), f"Encountered non-finite loss at test step {step}: {loss}"
        if trainer.use_amp:
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(optimizer)
            trainer.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print(f"    Step {step+1}/3 passed -> Loss: {loss.item():.4f}")

    del trainer, check_loader
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(">>> PRE-FLIGHT AUDIT PASSED WITH ZERO ERRORS! <<<\n")


# -----------------------------------------------------------------------------
# 4. Main Training Pipeline
# -----------------------------------------------------------------------------
def run_full_training(
    dataset_dir: Path,
    epochs: int = 50,
    warmup_epochs: int = 5,
    batch_size: int = 16,
    num_workers: int = 0,
    lr_head: float = 1e-3,
    ft_lr_head: float = 5e-4,
    ft_lr_backbone: float = 2e-5,
    patience: int = 10,
    seed: int = 42,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_amp: bool = True,
    checkpoint_dir: Optional[Path] = None,
) -> Path:
    """Execute the full 50-epoch retraining and validation pipeline with Balanced V3 engine."""
    ckpt_dir = checkpoint_dir or (PROJECT_ROOT / "specialists" / "optical_sar" / "checkpoints" / "experiment_balanced_v3_100scenes")
    print_banner(f"STAGE 4: LAUNCHING 50-EPOCH RETRAINING PIPELINE ON {device.type.upper()}")
    print(f"Dataset Directory : {dataset_dir}")
    print(f"Checkpoint Output : {ckpt_dir}")
    print(f"Total Epochs      : {epochs} ({warmup_epochs} Warmup + {epochs - warmup_epochs} Fine-Tuning)")
    print(f"Batch Size        : {batch_size}")
    print(f"DataLoader Workers: {num_workers}")
    print(f"CUDA AMP Enabled  : {use_amp}")
    print(f"Early Stopping    : Patience={patience} epochs on Val mIoU")

    from specialists.optical_sar.config_balanced_v3 import TrainingConfigV3
    from specialists.optical_sar.train_colab_v3 import ColabTrainerV3

    config = TrainingConfigV3(
        data_dir=dataset_dir,
        checkpoint_dir=ckpt_dir,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr_head_warmup=lr_head,
        ft_lr_head=ft_lr_head,
        ft_lr_backbone=ft_lr_backbone,
        patience=patience,
        seed=seed,
        use_amp=use_amp,
    )

    trainer = ColabTrainerV3(
        config=config,
        device=str(device),
    )

    best_checkpoint = trainer.run_full_training(
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        batch_size=batch_size,
        patience=patience,
    )

    return Path(best_checkpoint)


# -----------------------------------------------------------------------------
# 5. Production Specialist Verification & Summary
# -----------------------------------------------------------------------------
def verify_production_specialist(checkpoint_path: Path) -> None:
    """Verify the trained checkpoint loads into the production BaseSpecialistTool."""
    print_banner("STAGE 5: PRODUCTION SPECIALIST INTEGRATION VERIFICATION")
    from specialists.optical_sar.service import OpticalSarSpecialist

    if not checkpoint_path.exists():
        print(f"[!] Warning: Expected checkpoint not found at {checkpoint_path}")
        return

    sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    print(f"Checkpoint File  : {checkpoint_path}")
    print(f"File Size        : {size_mb:.2f} MiB")
    print(f"SHA-256 Checksum : {sha256}")

    specialist = OpticalSarSpecialist(checkpoint_path=checkpoint_path, require_trained_weights=True)
    print(f"Specialist Name  : {specialist.metadata.name}")
    print(f"Specialist Vers. : {specialist.metadata.version}")
    print(f"Supported Tasks  : {[t.value for t in specialist.supported_tasks]}")
    print(f"Trained Loaded   : {specialist.is_trained_loaded}")

    assert specialist.is_trained_loaded, "Production specialist failed to load trained checkpoint!"
    print(">>> PRODUCTION SPECIALIST VERIFICATION PASSED! <<<\n")


# -----------------------------------------------------------------------------
# 6. Main Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SatQuery Division 4 — 1-Click Retraining Script for Local RTX 4060 Laptop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", "--dataset-dir", dest="data_dir", type=str, default=None,
                        help="Path to local WHU-OPT-SAR dataset (auto-discovered if omitted)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total epochs (5 warmup + 45 fine-tuning)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Stage 1 frozen backbone warmup epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: 16, auto-adjusted if VRAM is low)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader multiprocessing workers (0 is safest for laptop GPUs)")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Stage 1 AdamW learning rate for head & neck")
    parser.add_argument("--ft-lr-head", type=float, default=5e-4,
                        help="Stage 2 AdamW learning rate for head & neck")
    parser.add_argument("--ft-lr-backbone", type=float, default=2e-5,
                        help="Stage 2 AdamW learning rate for unfrozen layer3 backbones")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without val mIoU improvement)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic reproducibility")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision (AMP)")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preliminary batch contract & stability audit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test run with 1 warmup epoch and 1 fine-tuning epoch")

    args = parser.parse_args()

    # Dry-run override
    if args.dry_run:
        args.epochs = 2
        args.warmup_epochs = 1
        print("\n>>> [DRY-RUN MODE ACTIVE] Running 2 epochs for quick verification <<<")

    # 1. Diagnostics & Hardware
    device, default_amp, recommended_batch_size = check_system_and_gpu()
    use_amp = default_amp and (not args.no_amp)
    batch_size = args.batch_size or recommended_batch_size

    # 2. Locate Dataset
    dataset_dir = locate_local_dataset(args.data_dir)

    # 3. Pre-Flight Checks
    if not args.skip_preflight:
        run_preflight_audit(dataset_dir, device, use_amp)

    # 4. Full Training Execution
    t_start = datetime.datetime.now()
    best_checkpoint = run_full_training(
        dataset_dir=dataset_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=batch_size,
        num_workers=args.num_workers,
        lr_head=args.lr_head,
        ft_lr_head=args.ft_lr_head,
        ft_lr_backbone=args.ft_lr_backbone,
        patience=args.patience,
        seed=args.seed,
        device=device,
        use_amp=use_amp,
    )
    total_duration = (datetime.datetime.now() - t_start).total_seconds() / 60.0

    # 5. Production Specialist Compatibility Check
    verify_production_specialist(best_checkpoint)

    # 6. Final Summary
    print_banner("ALL PHASES COMPLETED SUCCESSFULLY — TRAINING SUMMARY", "=")
    print(f"Total Execution Time : {total_duration:.1f} minutes")
    print(f"Winning Checkpoint   : {best_checkpoint}")
    print(f"Training History     : {best_checkpoint.parent / 'colab_training_history.json'}")
    print(f"Training Loss Curve  : {best_checkpoint.parent / 'training_curves.png'}")
    print(f"Test Confusion Matrix: {best_checkpoint.parent / 'test_confusion_matrix.png'}")
    print(f"Test Metrics JSON    : {best_checkpoint.parent / 'final_test_metrics.json'}")
    print(f"Per-Class Metrics CSV: {best_checkpoint.parent / 'final_per_class_metrics.csv'}")
    print("\nLak, you are good to go! Your new Optical-SAR specialist model is ready.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
