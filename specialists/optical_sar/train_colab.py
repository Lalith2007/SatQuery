"""Google Colab GPU Training & Evaluation Pipeline for Division 4 Optical-SAR Specialist.

Optimized for Colab T4 / V100 / A100 GPUs.
Trains CMAF cross-attention fusion neck and 8-class LandCoverTaskHead on the
17,160 real paired 256x256 tiles from the 52 official WHU-OPT-SAR image pairs.

Metadata explicitly records:
'dataset_version': 'Official WHU-OPT-SAR 52-Pair Development Subset (17,160 tiles)'
'benchmark_note': 'Development training run on 52 official pairs; not the final 100-pair full benchmark'
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from specialists.optical_sar.config import SpecialistConfig
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import OpticalEncoder, SarEncoder
from specialists.optical_sar.fusion import CrossModalAttentionFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("colab_train")

VERSION_MARKER = "TRAIN_COLAB_V5_CMAF_ATTN_NORM_2026_08_28"
DEFAULT_NUM_WORKERS = 0


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

        # Compute softmax in float32 for AMP precision safety
        probs = torch.softmax(logits.float(), dim=1)
        targets_onehot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * mask_expanded
        targets_onehot = targets_onehot * mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        target_sum = torch.sum(targets_onehot, dim=dims)

        # Mask out classes not present in the batch ground-truth to eliminate absent-class gradient singularities
        present_mask = (target_sum > 0)
        if not present_mask.any():
            return (logits * 0.0).sum()

        dice_score = (2.0 * intersection[present_mask] + self.smooth) / (cardinality[present_mask] + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return torch.clamp(dice_loss, 0.0, 1.0)


class ColabTrainer:
    """Manages Colab GPU training, mixed-precision AMP, evaluation metrics, and checkpointing."""

    def __init__(
        self,
        dataset_dir: Union[str, Path] = "data/official_whu_opt_sar",
        checkpoint_dir: Union[str, Path] = "specialists/optical_sar/checkpoints",
        config: Optional[SpecialistConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or SpecialistConfig()
        self.device = torch.device(device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        logger.info(f"Initializing ColabTrainer [{VERSION_MARKER}] on device='{self.device}' (AMP={self.use_amp})...")

        # Encoders
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

        # Calibrated inverse sqrt class weights for WHU-OPT-SAR full split (Forest=48%, Farmland=32%, Water=10%, Village=5%, City=3%, Others=1.3%, Road=0.6%)
        self.class_weights = torch.tensor([5.0, 1.0, 3.5, 2.8, 2.0, 1.0, 7.0, 5.0], dtype=torch.float32, device=self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
        self.dice_loss_fn = DiceLoss(num_classes=8, ignore_index=255, smooth=1.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def freeze_encoders(self) -> None:
        for param in self.optical_encoder.parameters():
            param.requires_grad = False
        for param in self.sar_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders_top_layers(self) -> None:
        for param in self.optical_encoder.layer3.parameters():
            param.requires_grad = True
        for param in self.sar_encoder.layer3.parameters():
            param.requires_grad = True

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        valid_mask = (targets_np != 255)
        preds_valid = preds_np[valid_mask]
        targets_valid = targets_np[valid_mask]

        correct = (preds_valid == targets_valid).sum()
        total = targets_valid.size
        pixel_acc = float(correct / total) if total > 0 else 1.0

        class_names = ["background", "farmland", "city", "village", "water", "forest", "road", "others"]
        metrics = {"pixel_acc": pixel_acc}
        ious = []

        for k in range(8):
            pred_k = (preds_valid == k)
            target_k = (targets_valid == k)
            intersection = (pred_k & target_k).sum()
            union = (pred_k | target_k).sum()
            iou = float(intersection / union) if union > 0 else 1.0
            ious.append(iou)
            metrics[f"iou_{class_names[k]}"] = iou

        metrics["mIoU"] = float(np.mean(ious))
        return metrics

    def train_epoch(
        self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int = 1, total_epochs: int = 15
    ) -> float:
        self.fusion_neck.train()
        self.task_head.train()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        total_loss = 0.0
        batch_idx = 0
        num_batches = len(dataloader)

        for batch in dataloader:
            batch_idx += 1
            opt = batch["optical"].to(self.device, non_blocking=True)
            sar = batch["sar"].to(self.device, non_blocking=True)
            targets = batch["label"].to(self.device, non_blocking=True)
            intent = batch["intent_vector"].to(self.device, non_blocking=True)

            # Precise diagnostic assertions
            if not torch.isfinite(opt).all():
                raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in optical input tensor!")
            if not torch.isfinite(sar).all():
                raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in SAR input tensor!")

            # Validate target labels are in 0..7 or 255
            unique_labels = torch.unique(targets)
            valid_label_set = {0, 1, 2, 3, 4, 5, 6, 7, 255}
            for ul in unique_labels.tolist():
                if ul not in valid_label_set:
                    raise ValueError(f"Batch {batch_idx}: Invalid target class label value: {ul}")

            valid_mask = (targets != 255)
            if not valid_mask.any():
                # Skip batches that contain only padded border ignore_index pixels
                continue

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                with torch.no_grad():
                    opt_feats = self.optical_encoder(opt)
                    sar_feats = self.sar_encoder(sar)

                if not torch.isfinite(opt_feats["stride_8"]).all():
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in optical encoder features!")
                if not torch.isfinite(sar_feats["stride_8"]).all():
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in SAR encoder features!")

                fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                if not torch.isfinite(fused).all():
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in CMAF fused features!")

                logits, probs = self.task_head(fused, intent)
                if not torch.isfinite(logits).all():
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in task_head logits!")

                if logits.shape[2:] != targets.shape[1:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                    )

                # Compute loss in float32 for numerical stability
                ce_loss = self.ce_loss_fn(logits.float(), targets)
                if not torch.isfinite(ce_loss):
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in CrossEntropy loss computation!")

                dice_loss = self.dice_loss_fn(logits.float(), targets)
                if not torch.isfinite(dice_loss):
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in Dice loss computation!")

                loss = ce_loss + 0.5 * dice_loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Batch {batch_idx}: NaN/Inf detected in total loss computation! Loss={loss.item()}")

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
                    f"Avg Loss: {total_loss / batch_idx:.4f} | Current CE: {ce_loss.item():.4f}, Dice: {dice_loss.item():.4f}",
                    flush=True,
                )

        return total_loss / max(1, batch_idx)

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.fusion_neck.eval()
        self.task_head.eval()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                opt = batch["optical"].to(self.device, non_blocking=True)
                sar = batch["sar"].to(self.device, non_blocking=True)
                targets = batch["label"].to(self.device, non_blocking=True)
                intent = batch["intent_vector"].to(self.device, non_blocking=True)

                opt_feats = self.optical_encoder(opt)
                sar_feats = self.sar_encoder(sar)

                fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                logits, probs = self.task_head(fused, intent)

                if logits.shape[2:] != targets.shape[1:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                    )

                ce_loss = self.ce_loss_fn(logits, targets)
                dice_loss = self.dice_loss_fn(logits, targets)
                loss = ce_loss + 0.5 * dice_loss
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

        concat_preds = torch.cat(all_preds, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)
        metrics = self.compute_metrics(concat_preds, concat_targets)

        return total_loss / len(dataloader), metrics

    def run_colab_training(
        self,
        epochs: int = 15,
        batch_size: int = 16,
        lr: float = 1e-3,
        num_workers: int = 0,
        fine_tune_epochs: int = 5,
    ) -> Path:
        print(f"Starting Colab GPU Training on dataset='{self.dataset_dir}' (Batch Size={batch_size}, Epochs={epochs}+{fine_tune_epochs})...", flush=True)

        train_ds = OpticalSarPairedDataset(self.dataset_dir, split="train", num_classes=8)
        val_ds = OpticalSarPairedDataset(self.dataset_dir, split="val", num_classes=8)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        trainable_params = list(self.fusion_neck.parameters()) + list(self.task_head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

        best_val_mIoU = -1.0
        best_ckpt_path = self.checkpoint_dir / "cmaf_landcover_best.pth"
        history = []

        print("\n================ GOOGLE COLAB TRAINING START ================", flush=True)
        print("Dataset Version: Official WHU-OPT-SAR 52-Pair Development Subset (17,160 tiles)", flush=True)
        print(f"Train Tiles: {len(train_ds):,d} ({len(train_loader)} batches) | Val Tiles: {len(val_ds):,d} ({len(val_loader)} batches)", flush=True)
        print("=============================================================\n", flush=True)

        total_all_epochs = epochs + fine_tune_epochs

        for epoch in range(1, epochs + 1):
            t0 = datetime.datetime.now()
            train_loss = self.train_epoch(train_loader, optimizer, epoch=epoch, total_epochs=total_all_epochs)
            val_loss, metrics = self.validate(val_loader)

            val_mIoU = metrics["mIoU"]
            dur = (datetime.datetime.now() - t0).total_seconds()

            msg = (
                f"\n>>> [Stage 1] Epoch {epoch:02d}/{epochs} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f} | Pixel Acc: {metrics.get('pixel_acc', 0.0)*100:.2f}% | "
                f"City IoU: {metrics.get('iou_city', 0.0):.4f} | Water IoU: {metrics.get('iou_water', 0.0):.4f}\n"
            )
            print(msg, flush=True)
            logger.info(msg.strip())

            history.append({
                "epoch": epoch,
                "stage": "stage1_frozen_encoders",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics": metrics,
            })

            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                self.save_colab_checkpoint(best_ckpt_path, epoch, val_loss, metrics, history)
                print(f"  --> Saved BEST Colab Checkpoint to '{best_ckpt_path}' (Val mIoU: {val_mIoU:.4f})\n", flush=True)

        if fine_tune_epochs > 0:
            print("\n--- Starting Stage 2: Fine-Tuning Top Layer3 Encoders ---\n", flush=True)
            self.unfreeze_encoders_top_layers()

            ft_params = [
                {"params": trainable_params, "lr": lr * 0.5},
                {"params": self.optical_encoder.layer3.parameters(), "lr": 1e-5},
                {"params": self.sar_encoder.layer3.parameters(), "lr": 1e-5},
            ]
            ft_optimizer = torch.optim.AdamW(ft_params, weight_decay=1e-4)

            for epoch in range(epochs + 1, total_all_epochs + 1):
                t0 = datetime.datetime.now()
                train_loss = self.train_epoch(train_loader, ft_optimizer, epoch=epoch, total_epochs=total_all_epochs)
                val_loss, metrics = self.validate(val_loader)

                val_mIoU = metrics["mIoU"]
                dur = (datetime.datetime.now() - t0).total_seconds()

                msg = (
                    f"\n>>> [Stage 2] Epoch {epoch:02d}/{total_all_epochs} ({dur:.1f}s) | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f} | Pixel Acc: {metrics.get('pixel_acc', 0.0)*100:.2f}% | "
                    f"City IoU: {metrics.get('iou_city', 0.0):.4f} | Water IoU: {metrics.get('iou_water', 0.0):.4f}\n"
                )
                print(msg, flush=True)
                logger.info(msg.strip())

                history.append({
                    "epoch": epoch,
                    "stage": "stage2_finetune",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "metrics": metrics,
                })

                if val_mIoU > best_val_mIoU:
                    best_val_mIoU = val_mIoU
                    self.save_colab_checkpoint(best_ckpt_path, epoch, val_loss, metrics, history)
                    print(f"  --> Saved BEST Colab Checkpoint to '{best_ckpt_path}' (Val mIoU: {val_mIoU:.4f})\n", flush=True)

        with open(self.checkpoint_dir / "colab_training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return best_ckpt_path

    def save_colab_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        history: list,
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics,
            "num_classes": 8,
            "dataset_version": "Official WHU-OPT-SAR 52-Pair Development Subset (17,160 tiles)",
            "benchmark_note": "Development training run on 52 official pairs; not the final 100-pair full benchmark",
            "optical_encoder_state_dict": self.optical_encoder.state_dict(),
            "sar_encoder_state_dict": self.sar_encoder.state_dict(),
            "fusion_neck_state_dict": self.fusion_neck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "config": json.loads(json.dumps(self.config.to_dict(), default=str)),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "history": history,
        }
        torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Colab GPU Training for Division 4")
    parser.add_argument("--data_dir", type=str, default="data/official_whu_opt_sar", help="Path to tiled dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Stage 1 training epochs")
    parser.add_argument("--ft_epochs", type=int, default=5, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU iteration")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    args = parser.parse_args()

    trainer = ColabTrainer(dataset_dir=args.data_dir)
    trainer.run_colab_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        fine_tune_epochs=args.ft_epochs,
    )
