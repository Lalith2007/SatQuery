"""Google Colab GPU Re-Training & Quantitative Evaluation Pipeline for Division 4 Optical-SAR Specialist.


Optimized for NVIDIA GPU in Google Colab (T4 / V100 / A100).
Trains Cross-Modal Attention Fusion (CMAF), FiLM query modulator, and 8-Class LandCoverTaskHead
on the complete WHU-OPT-SAR dataset using an image-level 70/15/15 split (70 train, 15 val, 15 test scenes).

Implements:
1. Dynamic training pixel frequency calculation and calibrated inverse square-root class weights.
2. Synchronized spatial data augmentations (horizontal flip, vertical flip, 90-degree rotations).
3. Bounded tile-level minority class sampling (WeightedRandomSampler) with logged effective distribution.
4. 50-epoch differential schedule: 5 warmup epochs (frozen backbones) + 45 fine-tuning epochs (unfreezing layer3 only — deepest exposed stage).
5. AdamW with differential learning rates and CosineAnnealingLR scheduler.
6. CUDA AMP mixed precision with gradient clipping and numerical assertions.
7. Validation tracking: OA, mIoU, Macro Precision/Recall/F1, per-class metrics, confusion matrix, and active predicted class monitoring.
8. Early stopping (patience=10 on val mIoU) and non-destructive checkpointing to cmaf_landcover_best_v2.pth.
9. Post-training single evaluation on untouched 15-scene held-out test split, Old vs New comparison, and 3-way modality ablation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from specialists.optical_sar.config import SpecialistConfig
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import OpticalEncoder, SarEncoder
from specialists.optical_sar.fusion import CrossModalAttentionFusion
from specialists.optical_sar.tile_official_dataset import LABEL_VALUE_TO_INDEX, OFFICIAL_LABEL_VALUES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("colab_train_v2")

CLASS_NAMES = ["background", "farmland", "city", "village", "water", "forest", "road", "others"]
MINORITY_CLASS_INDICES = [0, 2, 3, 4, 6, 7]  # Background, City, Village, Water, Road, Others


class DiceLoss(nn.Module):
    """Multi-class Dice Loss with present-class masking and ignore_index=255 border handling."""

    def __init__(self, num_classes: int = 8, ignore_index: int = 255, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return (logits * 0.0).sum()

        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        probs = torch.softmax(logits.float(), dim=1)
        targets_onehot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * mask_expanded
        targets_onehot = targets_onehot * mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        target_sum = torch.sum(targets_onehot, dim=dims)

        present_mask = (target_sum > 0)
        if not present_mask.any():
            return (logits * 0.0).sum()

        dice_score = (2.0 * intersection[present_mask] + self.smooth) / (cardinality[present_mask] + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return torch.clamp(dice_loss, 0.0, 1.0)


def compute_dynamic_class_weights(
    pixel_counts: np.ndarray,
    min_weight: float = 1.0,
    max_weight: float = 15.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute inverse square root class weights from actual training pixel frequencies."""
    counts = np.maximum(pixel_counts.astype(np.float64), epsilon)
    max_count = np.max(counts)
    raw_weights = np.sqrt(max_count / counts)
    clipped_weights = np.clip(raw_weights, min_weight, max_weight)
    norm_weights = clipped_weights * (len(pixel_counts) / np.sum(clipped_weights))
    return torch.tensor(norm_weights, dtype=torch.float32)


def build_minority_aware_sampler(
    dataset: OpticalSarPairedDataset,
    max_weight_ratio: float = 3.5,
) -> Tuple[WeightedRandomSampler, Dict[str, Any]]:
    """Construct a bounded minority-aware tile sampler and compute effective sampling distribution."""
    num_samples = len(dataset)
    presence_matrix = dataset.get_tile_class_presence()  # shape (N, 8) boolean

    rarity_boosts = {
        0: 1.0,  # Background
        2: 0.8,  # City
        3: 1.2,  # Village
        4: 1.2,  # Water
        6: 1.5,  # Road
        7: 1.5,  # Others
    }

    tile_weights = np.ones(num_samples, dtype=np.float64)
    for c_idx, boost in rarity_boosts.items():
        has_class = presence_matrix[:, c_idx]
        tile_weights[has_class] += boost

    tile_weights = np.clip(tile_weights, 1.0, max_weight_ratio)
    sampling_probs = tile_weights / np.sum(tile_weights)

    original_presence_fraction = {}
    effective_presence_fraction = {}

    for k, name in enumerate(CLASS_NAMES):
        orig_frac = float(np.mean(presence_matrix[:, k]))
        eff_frac = float(np.sum(sampling_probs[presence_matrix[:, k]]))
        original_presence_fraction[name] = orig_frac
        effective_presence_fraction[name] = eff_frac

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(tile_weights).float(),
        num_samples=num_samples,
        replacement=True,
    )

    sampling_info = {
        "max_weight_ratio": max_weight_ratio,
        "mean_tile_weight": float(np.mean(tile_weights)),
        "min_tile_weight": float(np.min(tile_weights)),
        "max_tile_weight": float(np.max(tile_weights)),
        "original_tile_presence": original_presence_fraction,
        "effective_tile_presence": effective_presence_fraction,
    }

    return sampler, sampling_info


class ColabTrainer:
    """Manages Google Colab GPU training, differential learning rates, metrics, and checkpointing."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/official_whu_opt_sar",
        checkpoint_dir: Union[str, Path] = "specialists/optical_sar/checkpoints",
        config: Optional[SpecialistConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        use_amp: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or SpecialistConfig()
        self.device = torch.device(device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        # 1. Initialize Encoders, Fusion Neck, and LandCover Head
        self.optical_encoder = OpticalEncoder(
            backbone_type=self.config.model.optical_encoder_type.split("_")[-1],
            pretrained_weights=self.config.model.optical_pretrained_weights,
            in_channels=self.config.model.in_channels_optical,
        ).to(self.device)

        self.sar_encoder = SarEncoder(
            backbone_type=self.config.model.sar_encoder_type.split("_")[-1],
            pretrained_weights=self.config.model.sar_pretrained_weights,
            in_channels=self.config.model.in_channels_sar,
        ).to(self.device)

        opt_out_ch = self.optical_encoder.out_channels["stride_8"]
        sar_out_ch = self.sar_encoder.out_channels["stride_8"]

        self.fusion_neck = CrossModalAttentionFusion(
            optical_channels=opt_out_ch,
            sar_channels=sar_out_ch,
            out_channels=self.config.model.fusion_channels,
        ).to(self.device)

        self.task_head = LandCoverTaskHead(
            feature_channels=self.config.model.fusion_channels,
            num_classes=8,
        ).to(self.device)

        self.dice_loss_fn = DiceLoss(num_classes=8, ignore_index=255, smooth=1.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.class_weights = None
        self.ce_loss_fn = None

    def get_parameter_counts(self) -> Dict[str, int]:
        """Compute exact parameter counts per component."""
        opt_params = sum(p.numel() for p in self.optical_encoder.parameters())
        sar_params = sum(p.numel() for p in self.sar_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_neck.parameters())
        head_params = sum(p.numel() for p in self.task_head.parameters())
        total_params = opt_params + sar_params + fusion_params + head_params
        return {
            "optical_encoder": opt_params,
            "sar_encoder": sar_params,
            "fusion_neck": fusion_params,
            "task_head": head_params,
            "total_model_parameters": total_params,
        }

    def freeze_encoders(self) -> None:
        """Freeze optical and SAR encoder backbones (Stage 1 Warmup)."""
        for param in self.optical_encoder.parameters():
            param.requires_grad = False
        for param in self.sar_encoder.parameters():
            param.requires_grad = False
        logger.info("Encoders: Optical and SAR ResNet-50 FROZEN for Stage 1 warmup.")

    def unfreeze_encoder_top_layers(self) -> None:
        """Unfreeze layer3 of both encoders for Stage 2 fine-tuning.

        The encoder architecture terminates at layer3 (stride-16). There is no
        deeper stage exposed, so only layer3 gradients are re-enabled here.
        """
        for param in self.optical_encoder.layer3.parameters():
            param.requires_grad = True
        for param in self.sar_encoder.layer3.parameters():
            param.requires_grad = True
        logger.info("Encoders: Unfroze layer3 of Optical & SAR backbones for Stage 2 fine-tuning.")

    def setup_loss(self, class_weights: torch.Tensor) -> None:
        """Set up weighted CrossEntropyLoss with inverse square root weights."""
        self.class_weights = class_weights.to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)

    def metrics_from_confusion_matrix(self, conf_matrix: np.ndarray) -> Dict[str, Any]:
        """Compute all classification and segmentation metrics directly from an 8x8 confusion matrix in O(1) time."""
        ious, precisions, recalls, f1s = [], [], [], []
        support_pixels, pred_pixels = [], []

        for k in range(8):
            tp = int(conf_matrix[k, k])
            fp = int(np.sum(conf_matrix[:, k]) - tp)
            fn = int(np.sum(conf_matrix[k, :]) - tp)
            support = int(np.sum(conf_matrix[k, :]))
            predicted = int(np.sum(conf_matrix[:, k]))

            support_pixels.append(support)
            pred_pixels.append(predicted)

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        total_support = sum(support_pixels)
        total_correct = int(np.trace(conf_matrix))
        oa = float(total_correct / total_support) if total_support > 0 else 1.0

        macro_iou = float(np.mean(ious))
        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        if total_support > 0:
            weighted_f1 = float(sum(f1s[k] * support_pixels[k] for k in range(8)) / total_support)
            weighted_recall = float(sum(recalls[k] * support_pixels[k] for k in range(8)) / total_support)
            weighted_precision = float(sum(precisions[k] * support_pixels[k] for k in range(8)) / total_support)
        else:
            weighted_f1, weighted_recall, weighted_precision = 0.0, 0.0, 0.0

        # Minority mean IoU (excluding Forest and Farmland)
        minority_ious = [ious[k] for k in MINORITY_CLASS_INDICES]
        minority_mIoU = float(np.mean(minority_ious))

        # Active predicted classes
        active_classes = [CLASS_NAMES[k] for k in range(8) if pred_pixels[k] > 0]
        zero_prediction_classes = [CLASS_NAMES[k] for k in range(8) if pred_pixels[k] == 0]

        metrics = {
            "overall_accuracy": oa,
            "mIoU": macro_iou,
            "minority_mIoU": minority_mIoU,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "active_class_count": len(active_classes),
            "zero_prediction_classes": zero_prediction_classes,
        }

        for k, name in enumerate(CLASS_NAMES):
            metrics[f"iou_{name}"] = float(ious[k])
            metrics[f"f1_{name}"] = float(f1s[k])
            metrics[f"precision_{name}"] = float(precisions[k])
            metrics[f"recall_{name}"] = float(recalls[k])
            metrics[f"support_{name}"] = support_pixels[k]
            metrics[f"predicted_{name}"] = pred_pixels[k]

        return metrics

    def compute_detailed_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute metrics using fast vectorized PyTorch bincount without Python loops or memory explosion."""
        valid_mask = (targets >= 0) & (targets < 8) & (preds >= 0) & (preds < 8)
        if valid_mask.any():
            indices = (targets[valid_mask] * 8 + preds[valid_mask]).to(torch.int64)
            conf_matrix = torch.bincount(indices, minlength=64).reshape(8, 8).cpu().numpy()
        else:
            conf_matrix = np.zeros((8, 8), dtype=np.int64)

        metrics = self.metrics_from_confusion_matrix(conf_matrix)
        return metrics, conf_matrix

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Execute one training epoch with mixed precision, NaN assertions, and gradient clipping."""
        self.fusion_neck.train()
        self.task_head.train()

        total_loss = 0.0
        num_batches = len(dataloader)
        batch_idx = 0

        for batch in dataloader:
            batch_idx += 1
            opt = batch["optical"].to(self.device, non_blocking=True)
            sar = batch["sar"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            intent = batch["intent_vector"].to(self.device, non_blocking=True)

            if not torch.isfinite(opt).all():
                raise ValueError(f"Epoch {epoch} Batch {batch_idx}: NaN/Inf in optical input!")
            if not torch.isfinite(sar).all():
                raise ValueError(f"Epoch {epoch} Batch {batch_idx}: NaN/Inf in SAR input!")

            valid_mask = (targets != 255)
            if not valid_mask.any():
                continue

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                opt_feats = self.optical_encoder(opt)
                sar_feats = self.sar_encoder(sar)

                fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                logits, _ = self.task_head(fused, intent)

                if logits.shape[2:] != targets.shape[1:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                    )

                ce_loss = self.ce_loss_fn(logits.float(), targets)
                dice_loss = self.dice_loss_fn(logits.float(), targets)
                loss = ce_loss + 0.5 * dice_loss

                if not torch.isfinite(loss):
                    raise ValueError(f"Epoch {epoch} Batch {batch_idx}: Loss exploded to {loss.item()}!")

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.fusion_neck.parameters()) + list(self.task_head.parameters()),
                    max_norm=1.0,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.fusion_neck.parameters()) + list(self.task_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0 or batch_idx == num_batches:
                print(
                    f"  [Epoch {epoch:02d}/{total_epochs}] Batch {batch_idx:03d}/{num_batches} | "
                    f"Avg Loss: {total_loss / batch_idx:.4f} | CE: {ce_loss.item():.4f} | Dice: {dice_loss.item():.4f}",
                    flush=True,
                )

        return total_loss / max(1, batch_idx)

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray]:
        """Validate model performance across full validation set using streaming O(1) memory."""
        self.fusion_neck.eval()
        self.task_head.eval()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        total_loss = 0.0
        conf_matrix = np.zeros((8, 8), dtype=np.int64)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                opt = batch["optical"].to(self.device, non_blocking=True)
                sar = batch["sar"].to(self.device, non_blocking=True)
                targets = batch["label"].to(self.device, non_blocking=True)
                intent = batch["intent_vector"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(self.use_amp and self.device.type == "cuda")):
                    opt_feats = self.optical_encoder(opt)
                    sar_feats = self.sar_encoder(sar)

                    fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                    logits, _ = self.task_head(fused, intent)

                    if logits.shape[2:] != targets.shape[1:]:
                        logits = F.interpolate(
                            logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                        )

                    ce_loss = self.ce_loss_fn(logits.float(), targets)
                    dice_loss = self.dice_loss_fn(logits.float(), targets)
                    loss = ce_loss + 0.5 * dice_loss
                    total_loss += loss.item()

                    preds = logits.argmax(dim=1)

                # Streaming O(1) RAM Confusion Matrix update via PyTorch bincount
                valid_mask = (targets >= 0) & (targets < 8) & (preds >= 0) & (preds < 8)
                if valid_mask.any():
                    indices = (targets[valid_mask] * 8 + preds[valid_mask]).to(torch.int64)
                    batch_cm = torch.bincount(indices, minlength=64).reshape(8, 8).cpu().numpy()
                    conf_matrix += batch_cm

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"    Validation Batch {batch_idx + 1:03d}/{len(dataloader)} | Running Loss: {total_loss / (batch_idx + 1):.4f}", flush=True)

        metrics = self.metrics_from_confusion_matrix(conf_matrix)
        avg_loss = total_loss / max(1, len(dataloader))

        return avg_loss, metrics, conf_matrix

    def run_colab_training(
        self,
        epochs: int = 50,
        warmup_epochs: int = 5,
        batch_size: int = 16,
        lr_head: float = 1e-3,
        ft_lr_head: float = 5e-4,
        ft_lr_backbone: float = 2e-5,
        num_workers: int = 0,
        patience: int = 10,
        seed: int = 42,
    ) -> Path:
        """Run complete 50-epoch retraining pipeline with validation selection and early stopping."""
        # Reproducibility seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Environment & Diagnostic Banner
        param_counts = self.get_parameter_counts()
        print("\n" + "=" * 80)
        print("SATQUERY DIVISION 4: COMPLETE OPTICAL-SAR RETRAINING PIPELINE (GOOGLE COLAB)")
        print("=" * 80)
        print(f"Start Timestamp   : {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        print(f"Device Name       : {self.device}")
        if self.device.type == "cuda":
            print(f"GPU Model         : {torch.cuda.get_device_name(0)}")
            print(f"Total VRAM        : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GiB")
            print(f"CUDA Version      : {torch.version.cuda}")
        print(f"PyTorch Version   : {torch.__version__}")
        print(f"AMP Enabled       : {self.use_amp}")
        print(f"Total Parameters  : {param_counts['total_model_parameters']:,d}")
        print(f"  - Optical ResNet: {param_counts['optical_encoder']:,d}")
        print(f"  - SAR ResNet    : {param_counts['sar_encoder']:,d}")
        print(f"  - CMAF Neck     : {param_counts['fusion_neck']:,d}")
        print(f"  - Task Head+FiLM: {param_counts['task_head']:,d}")
        print("=" * 80 + "\n", flush=True)

        # 1. Instantiate Datasets
        print(">>> [Phase 2] Loading Complete WHU-OPT-SAR Dataset Splits...")
        train_ds = OpticalSarPairedDataset(self.dataset_dir, split="train", num_classes=8, augment=True)
        val_ds = OpticalSarPairedDataset(self.dataset_dir, split="val", num_classes=8, augment=False)
        test_ds = OpticalSarPairedDataset(self.dataset_dir, split="test", num_classes=8, augment=False)

        print(f"  - Training tiles  : {len(train_ds):,d} (with synchronized spatial augmentations)")
        print(f"  - Validation tiles: {len(val_ds):,d} (deterministic evaluation)")
        print(f"  - Test tiles      : {len(test_ds):,d} (held-out untouched evaluation)\n", flush=True)

        # 2. Dynamic Training Class Frequencies & Weights
        print(">>> [Phase 5] Calculating Exact Training Pixel Frequencies...")
        pixel_counts = train_ds.compute_class_frequencies()
        total_pixels = max(1, int(np.sum(pixel_counts)))

        print("  Training Class Distribution:")
        for k, name in enumerate(CLASS_NAMES):
            pct = (pixel_counts[k] / total_pixels) * 100.0
            print(f"    Class {k} ({name:10s}): {pixel_counts[k]:12,d} pixels ({pct:5.2f}%)")

        class_weights = compute_dynamic_class_weights(pixel_counts)
        self.setup_loss(class_weights)

        print("\n  Computed Inverse Square-Root Class Weights (Clipped & Normalized):")
        for k, name in enumerate(CLASS_NAMES):
            print(f"    Class {k} ({name:10s}): {self.class_weights[k].item():.4f}")
        print(flush=True)

        # 3. Bounded Minority-Aware Sampler
        print(">>> [Phase 6] Constructing Bounded Minority-Aware Tile Sampler...")
        train_sampler, sampling_info = build_minority_aware_sampler(train_ds, max_weight_ratio=3.5)
        print("  Effective Tile Presence Comparison:")
        for name in CLASS_NAMES:
            orig = sampling_info["original_tile_presence"][name] * 100.0
            eff = sampling_info["effective_tile_presence"][name] * 100.0
            print(f"    {name:10s} -> Original: {orig:5.1f}% tiles | Effective Sampling: {eff:5.1f}% tiles")
        print(flush=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # 4. Optimizer and Scheduler setup
        best_checkpoint_path = self.checkpoint_dir / "cmaf_landcover_best_v2.pth"
        latest_checkpoint_path = self.checkpoint_dir / "cmaf_landcover_latest.pth"

        best_val_mIoU = -1.0
        best_epoch = 0
        best_metrics = {}
        epochs_no_improve = 0
        history = []

        # STAGE 1: Warmup (Frozen Encoders)
        print("\n" + "=" * 80)
        print(f"STAGE 1: WARMUP ({warmup_epochs} EPOCHS) — BACKBONES FROZEN")
        print("=" * 80, flush=True)

        self.freeze_encoders()
        stage1_params = list(self.fusion_neck.parameters()) + list(self.task_head.parameters())
        optimizer = torch.optim.AdamW(stage1_params, lr=lr_head, weight_decay=1e-4)

        for epoch in range(1, warmup_epochs + 1):
            t0 = datetime.datetime.now()
            train_loss = self.train_epoch(train_loader, optimizer, epoch, epochs)

            import gc
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"  [Epoch {epoch:02d}/{epochs:02d}] Running Validation (4,950 tiles, streaming O(1) memory)...", flush=True)
            val_loss, val_metrics, val_cm = self.validate(val_loader)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            dur = (datetime.datetime.now() - t0).total_seconds()

            val_mIoU = val_metrics["mIoU"]
            active_count = val_metrics["active_class_count"]

            print(
                f"[Stage 1] Epoch {epoch:02d}/{epochs:02d} ({dur:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val OA: {val_metrics['overall_accuracy']*100:.2f}% | "
                f"Val mIoU: {val_mIoU*100:.2f}% | Macro F1: {val_metrics['macro_f1']*100:.2f}% | "
                f"Active Classes: {active_count}/8 | LR: {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

            history.append({
                "epoch": epoch,
                "stage": "stage1_warmup",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics": val_metrics,
                "lr": optimizer.param_groups[0]['lr'],
            })

            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                best_epoch = epoch
                best_metrics = val_metrics
                self.save_checkpoint(best_checkpoint_path, epoch, val_loss, val_metrics, history, is_best=True)
                print(f"  --> Saved NEW BEST Model [v2] (Val mIoU: {val_mIoU*100:.2f}%)", flush=True)

        # STAGE 2: Fine-Tuning (Unfreezing Layer 3 & Layer 4)
        fine_tune_epochs = epochs - warmup_epochs
        if fine_tune_epochs > 0:
            print("\n" + "=" * 80)
            print(f"STAGE 2: FINE-TUNING ({fine_tune_epochs} EPOCHS) — TOP LAYERS UNFROZEN")
            print("=" * 80, flush=True)

            self.unfreeze_encoder_top_layers()

            encoder_params = (
                list(self.optical_encoder.layer3.parameters())
                + list(self.sar_encoder.layer3.parameters())
            )
            head_params = list(self.fusion_neck.parameters()) + list(self.task_head.parameters())

            ft_param_groups = [
                {"params": head_params, "lr": ft_lr_head},
                {"params": encoder_params, "lr": ft_lr_backbone},
            ]
            ft_optimizer = torch.optim.AdamW(ft_param_groups, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ft_optimizer, T_max=fine_tune_epochs, eta_min=1e-6
            )

            for epoch in range(warmup_epochs + 1, epochs + 1):
                t0 = datetime.datetime.now()
                train_loss = self.train_epoch(train_loader, ft_optimizer, epoch, epochs)

                import gc
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Running Validation (4,950 tiles, streaming O(1) memory)...", flush=True)
                val_loss, val_metrics, val_cm = self.validate(val_loader)
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                dur = (datetime.datetime.now() - t0).total_seconds()

                val_mIoU = val_metrics["mIoU"]
                active_count = val_metrics["active_class_count"]
                current_head_lr = ft_optimizer.param_groups[0]['lr']
                current_backbone_lr = ft_optimizer.param_groups[1]['lr']

                print(
                    f"[Stage 2] Epoch {epoch:02d}/{epochs:02d} ({dur:.1f}s) | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val OA: {val_metrics['overall_accuracy']*100:.2f}% | "
                    f"Val mIoU: {val_mIoU*100:.2f}% | Macro F1: {val_metrics['macro_f1']*100:.2f}% | "
                    f"Active Classes: {active_count}/8 | Head LR: {current_head_lr:.2e}, BB LR: {current_backbone_lr:.2e}",
                    flush=True,
                )

                scheduler.step()

                history.append({
                    "epoch": epoch,
                    "stage": "stage2_finetune",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "metrics": val_metrics,
                    "head_lr": current_head_lr,
                    "backbone_lr": current_backbone_lr,
                })

                # Always save latest
                self.save_checkpoint(latest_checkpoint_path, epoch, val_loss, val_metrics, history, is_best=False)

                # Check for best
                if val_mIoU > best_val_mIoU:
                    best_val_mIoU = val_mIoU
                    best_epoch = epoch
                    best_metrics = val_metrics
                    epochs_no_improve = 0
                    self.save_checkpoint(best_checkpoint_path, epoch, val_loss, val_metrics, history, is_best=True)
                    print(f"  --> Saved NEW BEST Model [v2] (Val mIoU: {val_mIoU*100:.2f}%)", flush=True)
                else:
                    epochs_no_improve += 1
                    print(f"  [Early Stopping Tracker] No improvement for {epochs_no_improve}/{patience} epochs.", flush=True)
                    if epochs_no_improve >= patience:
                        print(f"\n>>> EARLY STOPPING TRIGGERED AT EPOCH {epoch:02d} (Patience={patience}) <<<", flush=True)
                        break

        # Save complete training history
        with open(self.checkpoint_dir / "colab_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Plot training curves
        self.plot_training_curves(history, self.checkpoint_dir / "training_curves.png")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print(f"Best Checkpoint: {best_checkpoint_path}")
        print(f"Best Epoch     : {best_epoch:02d}")
        print(f"Best Val OA    : {best_metrics.get('overall_accuracy', 0.0)*100:.2f}%")
        print(f"Best Val mIoU  : {best_val_mIoU*100:.2f}%")
        print(f"Best Macro F1  : {best_metrics.get('macro_f1', 0.0)*100:.2f}%")
        print("=" * 80 + "\n", flush=True)

        # 5. Held-Out Untouched Test Evaluation
        print(">>> [Phase 14] Evaluating Best Checkpoint on Untouched Held-Out Test Set (15 scenes)...")
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # Load best checkpoint strictly
        ckpt = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
        self.optical_encoder.load_state_dict(ckpt["optical_encoder_state_dict"], strict=True)
        self.sar_encoder.load_state_dict(ckpt["sar_encoder_state_dict"], strict=True)
        self.fusion_neck.load_state_dict(ckpt["fusion_neck_state_dict"], strict=True)
        self.task_head.load_state_dict(ckpt["task_head_state_dict"], strict=True)

        test_loss, test_metrics, test_cm = self.validate(test_loader)

        # Plot and save confusion matrix
        self.plot_confusion_matrix(test_cm, self.checkpoint_dir / "test_confusion_matrix.png")

        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS (UNTOUCHED 15-SCENE HELD-OUT SPLIT)")
        print("=" * 80)
        print(f"Overall Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
        print(f"Mean IoU (mIoU) : {test_metrics['mIoU']*100:.2f}%")
        print(f"Macro F1 Score  : {test_metrics['macro_f1']*100:.2f}%")
        print(f"Weighted F1     : {test_metrics['weighted_f1']*100:.2f}%")
        print(f"Active Classes  : {test_metrics['active_class_count']}/8")
        print("-" * 80)
        print(f"{'Class':<12} | {'IoU (%)':<9} | {'F1 (%)':<9} | {'Recall (%)':<11} | {'Precision (%)':<13} | {'Support':<10}")
        print("-" * 80)
        for name in CLASS_NAMES:
            iou_v = test_metrics[f"iou_{name}"] * 100.0
            f1_v = test_metrics[f"f1_{name}"] * 100.0
            rec_v = test_metrics[f"recall_{name}"] * 100.0
            prec_v = test_metrics[f"precision_{name}"] * 100.0
            supp_v = test_metrics[f"support_{name}"]
            print(f"{name:<12} | {iou_v:7.2f}%  | {f1_v:7.2f}%  | {rec_v:9.2f}%  | {prec_v:11.2f}%  | {supp_v:10,d}")
        print("=" * 80 + "\n", flush=True)

        # Export test metrics JSON and CSV
        with open(self.checkpoint_dir / "final_test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        self.export_per_class_csv(test_metrics, self.checkpoint_dir / "final_per_class_metrics.csv")

        # Old vs New comparison
        self.run_old_vs_new_comparison(test_metrics)

        # Modality Ablation
        self.run_modality_ablation(test_loader)

        # Reproducibility check
        self.verify_reproducibility(test_ds)

        return best_checkpoint_path

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        history: list,
        is_best: bool = False,
    ) -> None:
        """Save state_dict, parameters, metrics, and complete provenance metadata."""
        git_commit = "unknown"
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics,
            "num_classes": 8,
            "class_names": CLASS_NAMES,
            "class_mapping": LABEL_VALUE_TO_INDEX,
            "class_weights": self.class_weights.cpu().tolist() if self.class_weights is not None else None,
            "dataset_version": "Complete Official WHU-OPT-SAR (100 pairs, 70/15/15 image-level split)",
            "benchmark_note": "Production re-training package for Division 4 Optical-SAR Specialist",
            "is_best": is_best,
            "optical_encoder_state_dict": self.optical_encoder.state_dict(),
            "sar_encoder_state_dict": self.sar_encoder.state_dict(),
            "fusion_neck_state_dict": self.fusion_neck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "config": json.loads(json.dumps(self.config.to_dict(), default=str)),
            "git_commit": git_commit,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "history": history,
        }
        torch.save(checkpoint, path)

    def plot_training_curves(self, history: list, save_path: Path) -> None:
        """Plot loss and mIoU progression curves across epochs."""
        try:
            epochs = [h["epoch"] for h in history]
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]
            val_mious = [h["metrics"]["mIoU"] * 100.0 for h in history]
            macro_f1s = [h["metrics"]["macro_f1"] * 100.0 for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(epochs, train_losses, label="Train Loss", color="tab:blue", lw=2)
            ax1.plot(epochs, val_losses, label="Val Loss", color="tab:orange", lw=2)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            ax2.plot(epochs, val_mious, label="Val mIoU (%)", color="tab:green", lw=2)
            ax2.plot(epochs, macro_f1s, label="Val Macro F1 (%)", color="tab:purple", lw=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score (%)")
            ax2.set_title("Validation Segmentation Metrics")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Training curves saved to: {save_path}")
        except Exception as e:
            logger.warning(f"Could not generate training curves plot: {e}")

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path) -> None:
        """Plot and save normalized confusion matrix heatmap."""
        try:
            cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)

            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Ground-Truth Class")
            ax.set_title("Normalized Confusion Matrix (Held-Out Test Set)")

            for i in range(8):
                for j in range(8):
                    val = cm_norm[i, j]
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"  Confusion matrix saved to: {save_path}")
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {e}")

    def export_per_class_csv(self, metrics: Dict[str, Any], save_path: Path) -> None:
        """Export per-class metrics to structured CSV."""
        import csv
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class Name", "IoU", "F1", "Precision", "Recall", "Support Pixels", "Predicted Pixels"])
            for name in CLASS_NAMES:
                writer.writerow([
                    name,
                    f"{metrics[f'iou_{name}']:.4f}",
                    f"{metrics[f'f1_{name}']:.4f}",
                    f"{metrics[f'precision_{name}']:.4f}",
                    f"{metrics[f'recall_{name}']:.4f}",
                    metrics[f"support_{name}"],
                    metrics[f"predicted_{name}"],
                ])
        print(f"  Per-class metrics CSV saved to: {save_path}")

    def run_old_vs_new_comparison(self, new_metrics: Dict[str, Any]) -> None:
        """Compare new checkpoint results against the baseline weak checkpoint."""
        old_metrics_file = Path("reports/optical_sar_evaluation_metrics.json")
        if not old_metrics_file.exists():
            print("  Note: Baseline metrics JSON not found; skipping Old vs New delta comparison.")
            return

        try:
            with open(old_metrics_file, "r") as f:
                old_metrics = json.load(f)

            old_oa = old_metrics.get("overall_accuracy", 0.2756) * 100.0
            old_miou = old_metrics.get("mean_iou", 0.0553) * 100.0
            old_f1 = old_metrics.get("macro_f1", 0.0925) * 100.0

            new_oa = new_metrics["overall_accuracy"] * 100.0
            new_miou = new_metrics["mIoU"] * 100.0
            new_f1 = new_metrics["macro_f1"] * 100.0

            print("\n" + "=" * 80)
            print("CHECKPOINT COMPARISON: OLD (cmaf_landcover_best.pth) vs NEW (v2)")
            print("=" * 80)
            print(f"{'Metric':<20} | {'Old Checkpoint':<16} | {'New Checkpoint':<16} | {'Delta':<10}")
            print("-" * 80)
            print(f"{'Overall Accuracy':<20} | {old_oa:14.2f}% | {new_oa:14.2f}% | {new_oa - old_oa:+6.2f}%")
            print(f"{'Mean IoU (mIoU)':<20} | {old_miou:14.2f}% | {new_miou:14.2f}% | {new_miou - old_miou:+6.2f}%")
            print(f"{'Macro F1 Score':<20} | {old_f1:14.2f}% | {new_f1:14.2f}% | {new_f1 - old_f1:+6.2f}%")
            print("-" * 80)

            old_ious = old_metrics.get("per_class_iou", {})
            print(f"{'Class':<20} | {'Old IoU':<16} | {'New IoU':<16} | {'Delta':<10}")
            print("-" * 80)
            for name in CLASS_NAMES:
                o_iou = old_ious.get(name, 0.0) * 100.0
                n_iou = new_metrics[f"iou_{name}"] * 100.0
                print(f"{name:<20} | {o_iou:14.2f}% | {n_iou:14.2f}% | {n_iou - o_iou:+6.2f}%")
            print("=" * 80 + "\n", flush=True)
        except Exception as e:
            logger.warning(f"Could not complete Old vs New comparison: {e}")

    def run_modality_ablation(self, dataloader: DataLoader) -> None:
        """Run 3-way ablation study: Dual (Optical+SAR), Optical-only, SAR-only."""
        print(">>> [Phase 16] Running Post-Training Multimodal Ablation Study...")
        self.fusion_neck.eval()
        self.task_head.eval()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        results = {}
        for mode in ["dual", "optical_only", "sar_only"]:
            conf_matrix = np.zeros((8, 8), dtype=np.int64)
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    if idx >= 50:  # Sample first 50 batches for ablation speed
                        break
                    opt = batch["optical"].to(self.device)
                    sar = batch["sar"].to(self.device)
                    targets = batch["label"].to(self.device)
                    intent = batch["intent_vector"].to(self.device)

                    if mode == "optical_only":
                        sar = torch.zeros_like(sar)
                    elif mode == "sar_only":
                        opt = torch.zeros_like(opt)

                    with torch.amp.autocast("cuda", enabled=(self.use_amp and self.device.type == "cuda")):
                        opt_feats = self.optical_encoder(opt)
                        sar_feats = self.sar_encoder(sar)
                        fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                        logits, _ = self.task_head(fused, intent)

                        if logits.shape[2:] != targets.shape[1:]:
                            logits = F.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False)

                        preds = logits.argmax(dim=1)

                    valid_mask = (targets >= 0) & (targets < 8) & (preds >= 0) & (preds < 8)
                    if valid_mask.any():
                        indices = (targets[valid_mask] * 8 + preds[valid_mask]).to(torch.int64)
                        batch_cm = torch.bincount(indices, minlength=64).reshape(8, 8).cpu().numpy()
                        conf_matrix += batch_cm

            results[mode] = self.metrics_from_confusion_matrix(conf_matrix)

        print("\n" + "=" * 70)
        print("MULTIMODAL ABLATION RESULTS (Sampled Evaluation)")
        print("=" * 70)
        print(f"{'Condition':<20} | {'Overall Accuracy':<18} | {'mIoU':<10} | {'Macro F1':<10}")
        print("-" * 70)
        for mode, label in [("dual", "Dual (Optical + SAR)"), ("optical_only", "Optical Only (Zero SAR)"), ("sar_only", "SAR Only (Zero Optical)")]:
            m = results[mode]
            print(f"{label:<20} | {m['overall_accuracy']*100:16.2f}% | {m['mIoU']*100:8.2f}% | {m['macro_f1']*100:8.2f}%")
        print("=" * 70 + "\n", flush=True)

    def verify_reproducibility(self, dataset: OpticalSarPairedDataset) -> None:
        """Verify deterministic inference: two duplicate passes yield identical outputs."""
        print(">>> [Phase 17] Verifying Deterministic Reproducibility...")
        self.fusion_neck.eval()
        self.task_head.eval()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        sample = dataset[0]
        opt = sample["optical"].unsqueeze(0).to(self.device)
        sar = sample["sar"].unsqueeze(0).to(self.device)
        intent = sample["intent_vector"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            fused1 = self.fusion_neck(self.optical_encoder(opt)["stride_8"], self.sar_encoder(sar)["stride_8"])
            logits1, probs1 = self.task_head(fused1, intent)

            fused2 = self.fusion_neck(self.optical_encoder(opt)["stride_8"], self.sar_encoder(sar)["stride_8"])
            logits2, probs2 = self.task_head(fused2, intent)

        max_diff = float(torch.max(torch.abs(logits1 - logits2)).item())
        pred_match = bool(torch.equal(logits1.argmax(dim=1), logits2.argmax(dim=1)))

        print(f"  Reproducibility Check -> Max logit diff: {max_diff:.2e} | Exact Prediction Match: {pred_match}")
        assert pred_match, "Reproducibility failure: duplicate runs produced different predictions!"
        assert max_diff < 1e-5, f"Reproducibility failure: logit tolerance exceeded ({max_diff})!"
        print("  >>> DETERMINISTIC REPRODUCIBILITY VERIFIED! <<<\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Colab GPU Complete Re-Training for Division 4")
    parser.add_argument("--data_dir", type=str, default="data/official_whu_opt_sar", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (warmup + fine-tune)")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Stage 1 frozen warmup epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Stage 1 head learning rate")
    parser.add_argument("--ft_lr_head", type=float, default=5e-4, help="Stage 2 head learning rate")
    parser.add_argument("--ft_lr_backbone", type=float, default=2e-5, help="Stage 2 backbone learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    trainer = ColabTrainer(dataset_dir=args.data_dir)
    trainer.run_colab_training(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        ft_lr_head=args.ft_lr_head,
        ft_lr_backbone=args.ft_lr_backbone,
        num_workers=args.num_workers,
        patience=args.patience,
        seed=args.seed,
    )
