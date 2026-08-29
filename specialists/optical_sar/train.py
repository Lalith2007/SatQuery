"""Training Pipeline for Division 4 Optical-SAR Specialist.

Trains Cross-Modal Attention Fusion (CMAF), FiLM query modulator,
and 8-Class Land-Cover task decoder on official paired WHU-OPT-SAR annotations.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
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
logger = logging.getLogger("train")


class DiceLoss(nn.Module):
    """Multi-class Dice Loss for semantic segmentation with ignore_index=255 support."""

    def __init__(self, num_classes: int = 8, ignore_index: int = 255, smooth: float = 1e-5) -> None:
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

        # Compute softmax in float32 for numerical stability
        probs = torch.softmax(logits.float(), dim=1)
        targets_onehot = F.one_hot(targets_clean, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Zero out ignore mask positions in both probs and onehot
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


class OpticalSarTrainer:
    """Trainer class managing dataset loading, 8-class loss calculation, evaluation, and checkpoint saving."""

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        checkpoint_dir: Union[str, Path] = "specialists/optical_sar/checkpoints",
        config: Optional[SpecialistConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or SpecialistConfig()
        self.device = torch.device(device)

        # 1. Initialize Encoders, Fusion Neck, and 8-Class Task Decoder
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

        # 2. Freeze Encoders initially (Stage 1)
        self.freeze_encoders()

        # 3. Loss Functions with ignore_index=255 border handling and class weights
        self.class_weights = torch.tensor([5.0, 1.0, 3.5, 2.8, 2.0, 1.0, 7.0, 5.0], dtype=torch.float32, device=self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=255)
        self.dice_loss_fn = DiceLoss(num_classes=8, ignore_index=255, smooth=1.0)

    def freeze_encoders(self) -> None:
        """Freeze optical and SAR encoder backbones."""
        for param in self.optical_encoder.parameters():
            param.requires_grad = False
        for param in self.sar_encoder.parameters():
            param.requires_grad = False
        logger.info("Optical & SAR ResNet-50 Encoders: FROZEN.")

    def unfreeze_encoder_top_layers(self) -> None:
        """Unfreeze layer3 of optical and SAR encoders for fine-tuning."""
        for param in self.optical_encoder.layer3.parameters():
            param.requires_grad = True
        for param in self.sar_encoder.layer3.parameters():
            param.requires_grad = True
        logger.info("Unfroze top layers (layer3) of Optical and SAR Encoders for fine-tuning.")

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute pixel accuracy, per-class IoU, and mean IoU ignoring 255 border pixels."""
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        valid_mask = (targets_np != 255)
        preds_valid = preds_np[valid_mask]
        targets_valid = targets_np[valid_mask]

        correct = (preds_valid == targets_valid).sum()
        total = targets_valid.size
        pixel_acc = correct / total if total > 0 else 1.0

        ious = []
        class_names = ["background", "farmland", "city", "village", "water", "forest", "road", "others"]
        metrics = {"pixel_acc": float(pixel_acc)}

        for k in range(8):
            pred_k = preds_valid == k
            target_k = targets_valid == k
            intersection = (pred_k & target_k).sum()
            union = (pred_k | target_k).sum()
            iou = intersection / union if union > 0 else 1.0
            ious.append(iou)
            metrics[f"iou_{class_names[k]}"] = float(iou)

        metrics["mIoU"] = float(np.mean(ious))
        return metrics

    def train_epoch(
        self, dataloader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Execute one training epoch over dataloader."""
        self.fusion_neck.train()
        self.task_head.train()

        total_loss = 0.0

        for batch in dataloader:
            opt = batch["optical"].to(self.device)         # (B, 3, H, W)
            sar = batch["sar"].to(self.device)             # (B, 2, H, W)
            targets = batch["label"].to(self.device)       # (B, H, W) 0..7, 255 ignore
            intent = batch["intent_vector"].to(self.device)# (B, 4)

            optimizer.zero_grad()

            with torch.no_grad():
                opt_feats = self.optical_encoder(opt)
                sar_feats = self.sar_encoder(sar)

            fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
            logits, probs = self.task_head(fused, intent, optical_raw=opt, sar_raw=sar)

            if logits.shape[2:] != targets.shape[1:]:
                logits = F.interpolate(
                    logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                )

            ce_loss = self.ce_loss_fn(logits, targets)
            dice_loss = self.dice_loss_fn(logits, targets)
            loss = ce_loss + 0.5 * dice_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.fusion_neck.parameters()) + list(self.task_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model performance over validation dataloader."""
        self.fusion_neck.eval()
        self.task_head.eval()
        self.optical_encoder.eval()
        self.sar_encoder.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                opt = batch["optical"].to(self.device)
                sar = batch["sar"].to(self.device)
                targets = batch["label"].to(self.device)
                intent = batch["intent_vector"].to(self.device)

                opt_feats = self.optical_encoder(opt)
                sar_feats = self.sar_encoder(sar)

                fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                logits, probs = self.task_head(fused, intent, optical_raw=opt, sar_raw=sar)

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

    def run_training(
        self,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-3,
        fine_tune_epochs: int = 3,
    ) -> Path:
        """Run complete 2-stage training pipeline and save best checkpoint."""
        logger.info(f"Starting Division 4 Training on device='{self.device}' for {epochs} epochs...")

        train_ds = OpticalSarPairedDataset(self.dataset_dir, split="train", num_classes=8)
        val_ds = OpticalSarPairedDataset(self.dataset_dir, split="val", num_classes=8)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        trainable_params = (
            list(self.fusion_neck.parameters()) + list(self.task_head.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

        best_val_mIoU = -1.0
        best_checkpoint_path = self.checkpoint_dir / "cmaf_landcover_best.pth"
        history = []

        for epoch in range(1, epochs + 1):
            t0 = datetime.datetime.now()
            train_loss = self.train_epoch(train_loader, optimizer)
            val_loss, metrics = self.validate(val_loader)

            val_mIoU = metrics["mIoU"]
            dur = (datetime.datetime.now() - t0).total_seconds()

            logger.info(
                f"Epoch {epoch:02d}/{epochs} [{dur:.1f}s] | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f} | City IoU: {metrics.get('iou_city', 0.0):.4f} | Water IoU: {metrics.get('iou_water', 0.0):.4f}"
            )

            history.append({
                "epoch": epoch,
                "stage": "stage1_frozen",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "metrics": metrics,
            })

            if val_mIoU > best_val_mIoU:
                best_val_mIoU = val_mIoU
                self.save_checkpoint(best_checkpoint_path, epoch, val_loss, metrics, history)
                logger.info(f"--> Saved new BEST checkpoint at '{best_checkpoint_path}' (Val mIoU: {val_mIoU:.4f})")

        if fine_tune_epochs > 0:
            logger.info("\n--- Starting Stage 2: Fine-Tuning Top Encoder Layers ---")
            self.unfreeze_encoder_top_layers()

            ft_params = [
                {"params": trainable_params, "lr": lr * 0.5},
                {"params": self.optical_encoder.layer3.parameters(), "lr": 1e-5},
                {"params": self.sar_encoder.layer3.parameters(), "lr": 1e-5},
            ]
            ft_optimizer = torch.optim.AdamW(ft_params, weight_decay=1e-4)

            for epoch in range(epochs + 1, epochs + fine_tune_epochs + 1):
                t0 = datetime.datetime.now()
                train_loss = self.train_epoch(train_loader, ft_optimizer)
                val_loss, metrics = self.validate(val_loader)

                val_mIoU = metrics["mIoU"]
                dur = (datetime.datetime.now() - t0).total_seconds()

                logger.info(
                    f"FT Epoch {epoch:02d} [{dur:.1f}s] | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val mIoU: {val_mIoU:.4f}"
                )

                history.append({
                    "epoch": epoch,
                    "stage": "stage2_fine_tune",
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "metrics": metrics,
                })

                if val_mIoU > best_val_mIoU:
                    best_val_mIoU = val_mIoU
                    self.save_checkpoint(best_checkpoint_path, epoch, val_loss, metrics, history)
                    logger.info(f"--> Saved new BEST fine-tuned checkpoint at '{best_checkpoint_path}' (Val mIoU: {val_mIoU:.4f})")

        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return best_checkpoint_path

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        history: list,
    ) -> None:
        """Save state_dict, parameters, metrics, and metadata to PyTorch checkpoint file."""
        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics,
            "num_classes": 8,
            "optical_encoder_state_dict": self.optical_encoder.state_dict(),
            "sar_encoder_state_dict": self.sar_encoder.state_dict(),
            "fusion_neck_state_dict": self.fusion_neck.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "config": self.config.to_dict(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "history": history,
        }
        torch.save(checkpoint, path)
