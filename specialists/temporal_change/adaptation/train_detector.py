"""PyTorch Training Pipeline for TinyCD Bi-Temporal Change Detector on Large-Scale LEVIR-CD.

Features:
- CLI configurable arguments (dataset root, mode full/dev, epochs, batch size, gradient accumulation, lr)
- Memory-safe DataLoader with configurable num_workers
- Mixed precision training (fp16 / bf16 / none) on CUDA
- Binary Cross-Entropy (BCELoss) optimization for stable probability learning
- AdamW optimizer with CosineAnnealingLR scheduling
- Programmatic data leakage verification before epoch 1
- Saves both Best Checkpoint (by validation F1) and Last Checkpoint
- Comprehensive training telemetry recording in training_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from specialists.temporal_change.adaptation.models.tinycd import TinyCD, count_parameters
from specialists.temporal_change.adaptation.dataset_loader import (
    LEVIRCDDatasetLoader,
    TemporalChangeDataset,
)
from specialists.temporal_change.adaptation.smoke_test_detector import compute_file_sha256


class CombinedBCEDiceLoss(nn.Module):
    """Hybrid loss combining Binary Cross-Entropy and Soft Dice Loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)

        intersection = (pred * target).sum(dim=(1, 2, 3))
        cardinality = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute binary classification metrics (Precision, Recall, F1, IoU) at threshold."""
    bin_pred = (pred >= threshold).float()
    bin_target = (target >= 0.5).float()

    tp = float((bin_pred * bin_target).sum().item())
    fp = float((bin_pred * (1.0 - bin_target)).sum().item())
    fn = float(((1.0 - bin_pred) * bin_target).sum().item())
    tn = float(((1.0 - bin_pred) * (1.0 - bin_target)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    oa = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "oa": oa,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tinycd_pipeline(
    dataset_root: Path | str,
    output_dir: Path | str = "specialists/temporal_change/weights",
    mode: str = "full",
    epochs: int = 20,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    image_size: int = 256,
    mixed_precision: str = "auto",
    resume_checkpoint: Optional[str] = None,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete PyTorch training for TinyCD model on real bi-temporal datasets."""
    set_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = device

    print("=" * 60)
    print("SATQUERY AI — DIVISION 3: TINYCD TRAINING PIPELINE")
    print(f"Dataset Root:        {dataset_root}")
    print(f"Dataset Mode:        {mode.upper()}")
    print(f"Compute Device:      {dev}")
    print(f"Epochs:              {epochs}")
    print(f"Batch Size:          {batch_size} (Grad Accum: {gradient_accumulation_steps})")
    print(f"Learning Rate:       {learning_rate}")
    print("=" * 60)

    # 1. Discover split samples dynamically
    train_samples = LEVIRCDDatasetLoader.discover_split_samples(dataset_root, "train", mode=mode)
    val_samples = LEVIRCDDatasetLoader.discover_split_samples(dataset_root, "val", mode=mode)
    test_samples = LEVIRCDDatasetLoader.discover_split_samples(dataset_root, "test", mode=mode)

    if not train_samples:
        raise FileNotFoundError(
            f"No training samples found in '{dataset_root}/train'. "
            "Please ensure LEVIR-CD is properly mounted or downloaded."
        )

    # 2. Mandatory Data Leakage Audit
    leakage_audit = LEVIRCDDatasetLoader.audit_split_leakage(train_samples, val_samples, test_samples, check_file_hashes=False)
    audit_file = eval_dir / "dataset_audit.json"
    with open(audit_file, "w") as f:
        json.dump(leakage_audit, f, indent=2)

    print("\nLEVIR-CD DATASET AUDIT")
    print("-" * 30)
    print(f"Train Parent Scenes:      {leakage_audit['train_parent_count']}")
    print(f"Train Patch Pairs:        {leakage_audit['train_samples_count']}")
    print(f"Val Parent Scenes:        {leakage_audit['val_parent_count']}")
    print(f"Val Patch Pairs:          {leakage_audit['val_samples_count']}")
    print(f"Test Parent Scenes:       {leakage_audit['test_parent_count']}")
    print(f"Test Patch Pairs:         {leakage_audit['test_samples_count']}")
    print(f"Leakage Free:             {leakage_audit['is_leakage_free']}")

    if not leakage_audit["is_leakage_free"]:
        raise RuntimeError(
            f"CRITICAL: Spatial data leakage detected between dataset splits! Audit: {leakage_audit}"
        )

    # 3. Generate Official Dataset Manifest & Class Balance Analysis
    manifest_file = eval_dir / "levir_cd_manifest.json"
    manifest = LEVIRCDDatasetLoader.generate_dataset_manifest(
        dataset_root, train_samples, val_samples, test_samples, output_path=manifest_file, mode=mode
    )
    print(f"Dataset Manifest Saved:   {manifest_file}")
    if "train_class_balance" in manifest and "class_imbalance_ratio" in manifest["train_class_balance"]:
        print(f"Class Balance Ratio:      {manifest['train_class_balance']['class_imbalance_ratio']}")

    # 4. Lazy DataLoaders
    train_ds = TemporalChangeDataset(train_samples, image_size=image_size, is_training=True)
    val_ds = TemporalChangeDataset(val_samples, image_size=image_size, is_training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if dev != "mps" else 0,
        drop_last=True if len(train_ds) > batch_size else False,
        pin_memory=True if dev == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if dev != "mps" else 0,
        pin_memory=True if dev == "cuda" else False,
    )

    # 5. Model, Loss, Optimizer, Scheduler
    model = TinyCD(in_channels=3, base_features=32)
    start_epoch = 1

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt_state = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt_state)

    model.to(dev)
    total_params, trainable_params = count_parameters(model)
    print(f"\nTinyCD Parameters: Total = {total_params:,}, Trainable = {trainable_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Mixed precision setup
    use_amp = False
    scaler = None
    if mixed_precision == "auto":
        use_amp = (dev == "cuda")
    elif mixed_precision in ["fp16", "bf16"]:
        use_amp = (dev == "cuda")

    if use_amp and dev == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed Precision: Enabled (torch.cuda.amp.GradScaler)")

    best_f1 = -1.0
    best_epoch = -1
    best_checkpoint_path = out_dir / "ChangeDetector-TinyCD.pth"
    last_checkpoint_path = out_dir / "ChangeDetector-TinyCD-last.pth"
    history: List[Dict[str, Any]] = []

    start_total_time = time.perf_counter()
    print("\nStarting Training Execution...")

    for epoch in range(start_epoch, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        train_loss_accum = 0.0
        train_steps = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            t0 = batch["t0"].to(dev, non_blocking=True)
            t1 = batch["t1"].to(dev, non_blocking=True)
            mask = batch["mask"].to(dev, non_blocking=True)

            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(t0, t1)
                    loss = criterion(pred.float(), mask.float())
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                pred = model(t0, t1)
                loss = criterion(pred.float(), mask.float())
                loss = loss / gradient_accumulation_steps
                loss.backward()


                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss_accum += (loss.item() * gradient_accumulation_steps)
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss_accum / max(train_steps, 1)

        # Validation phase (frozen weights)
        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        all_metrics: List[Dict[str, float]] = []

        with torch.no_grad():
            for batch in val_loader:
                t0 = batch["t0"].to(dev, non_blocking=True)
                t1 = batch["t1"].to(dev, non_blocking=True)
                mask = batch["mask"].to(dev, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(t0, t1)
                        loss = criterion(pred, mask)
                else:
                    pred = model(t0, t1)
                    loss = criterion(pred, mask)

                val_loss_accum += loss.item()
                val_steps += 1

                m = calculate_batch_metrics(pred, mask, threshold=0.5)
                all_metrics.append(m)

        avg_val_loss = val_loss_accum / max(val_steps, 1)
        avg_val_f1 = float(np.mean([m["f1"] for m in all_metrics])) if all_metrics else 0.0
        avg_val_iou = float(np.mean([m["iou"] for m in all_metrics])) if all_metrics else 0.0
        avg_val_precision = float(np.mean([m["precision"] for m in all_metrics])) if all_metrics else 0.0
        avg_val_recall = float(np.mean([m["recall"] for m in all_metrics])) if all_metrics else 0.0

        epoch_duration = time.perf_counter() - t_epoch_start
        curr_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {avg_val_f1:.4f} | "
            f"Val IoU: {avg_val_iou:.4f} | "
            f"Time: {epoch_duration:.1f}s | "
            f"LR: {curr_lr:.2e}"
        )

        epoch_log = {
            "epoch": epoch,
            "train_samples_count": len(train_samples),
            "train_loss": round(avg_train_loss, 6),
            "val_samples_count": len(val_samples),
            "val_loss": round(avg_val_loss, 6),
            "val_f1": round(avg_val_f1, 4),
            "val_iou": round(avg_val_iou, 4),
            "val_precision": round(avg_val_precision, 4),
            "val_recall": round(avg_val_recall, 4),
            "duration_sec": round(epoch_duration, 2),
            "learning_rate": round(curr_lr, 8),
        }
        history.append(epoch_log)

        # Save last checkpoint
        torch.save(model.state_dict(), last_checkpoint_path)

        # Save best checkpoint (selected on validation F1)
        if avg_val_f1 > best_f1 or epoch == 1:
            best_f1 = avg_val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)

    total_duration = time.perf_counter() - start_total_time
    chk_sha256 = compute_file_sha256(best_checkpoint_path) if best_checkpoint_path.exists() else ""

    summary = {
        "status": "REAL_MODEL_TRAINED",
        "dataset_mode": mode.upper(),
        "model_name": "TinyCD",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_f1, 4),
        "total_training_duration_sec": round(total_duration, 2),
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_sha256": chk_sha256,
        "training_history": history,
    }

    metrics_file = out_dir / "training_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING RUN COMPLETE")
    print(f"Best Validation F1:    {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Total Duration:        {total_duration:.1f}s")
    print(f"Best Checkpoint Saved: {best_checkpoint_path}")
    print(f"Checkpoint SHA-256:    {chk_sha256}")
    print("=" * 60)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyCD on LEVIR-CD Bi-Temporal Change Detection.")
    parser.add_argument("--dataset-root", type=str, default=os.getenv("SATQUERY_TC_DATASET_ROOT", "specialists/temporal_change/data/levir_cd"), help="Root path of LEVIR-CD dataset")
    parser.add_argument("--mode", type=str, default=os.getenv("SATQUERY_TC_DATA_MODE", "full"), choices=["full", "dev"], help="Dataset split mode: full or dev")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--image-size", type=int, default=256, help="Square patch dimension")
    parser.add_argument("--mixed-precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "none"], help="Mixed precision mode")
    parser.add_argument("--checkpoint-dir", type=str, default="specialists/temporal_change/weights", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda, mps, cpu, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tinycd_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.checkpoint_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        mixed_precision=args.mixed_precision,
        resume_checkpoint=args.resume,
        device=args.device,
        seed=args.seed,
    )
