"""Evaluation and Ablation Pipeline for Division 4 Optical-SAR Specialist.

Evaluates trained checkpoint on held-out test split, computes per-class metrics
(Pixel Accuracy, Precision, Recall, F1/Dice, mIoU across 8 official classes),
performs modality ablation studies, and generates qualitative visual artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import OpticalEncoder, SarEncoder
from specialists.optical_sar.fusion import CrossModalAttentionFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


class OpticalSarEvaluator:
    """Evaluates trained Optical-SAR specialist on test data and performs ablation studies."""

    def __init__(
        self,
        checkpoint_path: str = "specialists/optical_sar/checkpoints/cmaf_landcover_best.pth",
        dataset_dir: str = "data/official_whu_opt_sar",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_dir = Path(dataset_dir)
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{self.checkpoint_path}'")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.config_dict = ckpt.get("config", {})

        self.optical_encoder = OpticalEncoder(pretrained_weights=None).to(self.device)
        self.sar_encoder = SarEncoder(pretrained_weights=None).to(self.device)

        opt_out_ch = self.optical_encoder.out_channels["stride_8"]
        sar_out_ch = self.sar_encoder.out_channels["stride_8"]

        self.fusion_neck = CrossModalAttentionFusion(
            optical_channels=opt_out_ch,
            sar_channels=sar_out_ch,
            out_channels=256,
        ).to(self.device)

        self.task_head = LandCoverTaskHead(feature_channels=256, num_classes=8).to(self.device)

        self.optical_encoder.load_state_dict(ckpt["optical_encoder_state_dict"])
        self.sar_encoder.load_state_dict(ckpt["sar_encoder_state_dict"])
        self.fusion_neck.load_state_dict(ckpt["fusion_neck_state_dict"])
        self.task_head.load_state_dict(ckpt["task_head_state_dict"])

        self.optical_encoder.eval()
        self.sar_encoder.eval()
        self.fusion_neck.eval()
        self.task_head.eval()

        logger.info(f"Successfully loaded 8-class trained checkpoint from '{self.checkpoint_path}' for evaluation.")

    def compute_detailed_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute pixel accuracy, precision, recall, F1/Dice, and per-class IoU ignoring 255 border pixels."""
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        valid_mask = (targets_np != 255)
        preds_valid = preds_np[valid_mask]
        targets_valid = targets_np[valid_mask]

        correct = (preds_valid == targets_valid).sum()
        total = targets_valid.size
        pixel_acc = float(correct / total) if total > 0 else 1.0

        class_names = ["background", "farmland", "city", "village", "water", "forest", "road", "others"]
        metrics = {"pixel_accuracy": pixel_acc}
        ious, precisions, recalls, f1s = [], [], [], []

        for k in range(8):
            pred_k = preds_valid == k
            target_k = targets_valid == k

            tp = (pred_k & target_k).sum()
            fp = (pred_k & ~target_k).sum()
            fn = (~pred_k & target_k).sum()

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            c_name = class_names[k]
            metrics[f"iou_{c_name}"] = float(iou)
            metrics[f"precision_{c_name}"] = float(precision)
            metrics[f"recall_{c_name}"] = float(recall)
            metrics[f"f1_{c_name}"] = float(f1)

        metrics["mIoU"] = float(np.mean(ious))
        metrics["mean_precision"] = float(np.mean(precisions))
        metrics["mean_recall"] = float(np.mean(recalls))
        metrics["mean_f1"] = float(np.mean(f1s))

        return metrics

    def evaluate_split(
        self, dataloader: DataLoader, mode: str = "full"
    ) -> Dict[str, Any]:
        """Evaluate dataset split under specific modality mode: 'full', 'optical_only', or 'sar_only'."""
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                opt = batch["optical"].to(self.device)
                sar = batch["sar"].to(self.device)
                targets = batch["label"].to(self.device)
                intent = batch["intent_vector"].to(self.device)

                if mode == "optical_only":
                    sar = torch.zeros_like(sar)
                elif mode == "sar_only":
                    opt = torch.zeros_like(opt)

                opt_feats = self.optical_encoder(opt)
                sar_feats = self.sar_encoder(sar)

                fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
                logits, probs = self.task_head(fused, intent, optical_raw=opt, sar_raw=sar)

                if logits.shape[2:] != targets.shape[1:]:
                    logits = F.interpolate(
                        logits, size=targets.shape[1:], mode="bilinear", align_corners=False
                    )

                preds = logits.argmax(dim=1)
                all_preds.append(preds)
                all_targets.append(targets)

        concat_preds = torch.cat(all_preds, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)
        return self.compute_detailed_metrics(concat_preds, concat_targets)

    def run_ablation_study(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Run full ablation study comparing Optical-only, SAR-only, and Fused Optical+SAR."""
        logger.info("\n=== Running Modality Ablation Study on Test Split ===")
        results = {
            "optical_only": self.evaluate_split(test_loader, mode="optical_only"),
            "sar_only": self.evaluate_split(test_loader, mode="sar_only"),
            "optical_sar_fused": self.evaluate_split(test_loader, mode="full"),
        }

        for mode, res in results.items():
            logger.info(
                f"[{mode.upper()}] mIoU: {res['mIoU']:.4f} | City IoU: {res.get('iou_city', 0.0):.4f} | Water IoU: {res.get('iou_water', 0.0):.4f} | F1: {res['mean_f1']:.4f}"
            )

        return results

    def save_qualitative_examples(
        self, test_loader: DataLoader, output_dir: Path, num_examples: int = 3
    ) -> None:
        """Generate and save qualitative evaluation visual grids."""
        output_dir.mkdir(parents=True, exist_ok=True)
        batch = next(iter(test_loader))

        opt_b = batch["optical"].to(self.device)
        sar_b = batch["sar"].to(self.device)
        targets_b = batch["label"].to(self.device)
        intent_b = batch["intent_vector"].to(self.device)
        sample_ids = batch["sample_id"]

        with torch.no_grad():
            opt_feats = self.optical_encoder(opt_b)
            sar_feats = self.sar_encoder(sar_b)
            fused = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
            logits, probs = self.task_head(fused, intent_b, optical_raw=opt_b, sar_raw=sar_b)

            if logits.shape[2:] != targets_b.shape[1:]:
                logits = F.interpolate(logits, size=targets_b.shape[1:], mode="bilinear", align_corners=False)

            preds_b = logits.argmax(dim=1)

        for i in range(min(num_examples, len(sample_ids))):
            sample_id = sample_ids[i]
            opt_img = opt_b[i].cpu().numpy().transpose(1, 2, 0)
            sar_img = sar_b[i][0].cpu().numpy()
            target_mask = targets_b[i].cpu().numpy()
            pred_mask = preds_b[i].cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(np.clip(opt_img, 0, 1))
            axes[0].set_title("Optical Input (RGB)")
            axes[0].axis("off")

            axes[1].imshow(sar_img, cmap="gray")
            axes[1].set_title("SAR Input (Intensity)")
            axes[1].axis("off")

            axes[2].imshow(target_mask, vmin=0, vmax=7, cmap="tab10")
            axes[2].set_title("Ground-Truth Mask (8-Class)")
            axes[2].axis("off")

            axes[3].imshow(pred_mask, vmin=0, vmax=7, cmap="tab10")
            axes[3].set_title("Predicted Mask (8-Class Fused)")
            axes[3].axis("off")

            plt.suptitle(f"Qualitative Evaluation — {sample_id}")
            plt.tight_layout()
            out_file = output_dir / f"qualitative_eval_{sample_id}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            logger.info(f"Saved qualitative evaluation grid to '{out_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Division 4 Optical-SAR Specialist")
    parser.add_argument("--ckpt", type=str, default="specialists/optical_sar/checkpoints/cmaf_landcover_best.pth", help="Checkpoint path")
    parser.add_argument("--data_dir", type=str, default="data/official_whu_opt_sar", help="Dataset directory")
    args = parser.parse_args()

    evaluator = OpticalSarEvaluator(checkpoint_path=args.ckpt, dataset_dir=args.data_dir)
    test_ds = OpticalSarPairedDataset(args.data_dir, split="test", num_classes=8)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    test_metrics = evaluator.evaluate_split(test_loader, mode="full")
    ablation_results = evaluator.run_ablation_study(test_loader)

    eval_dir = Path("specialists/optical_sar/eval_results")
    evaluator.save_qualitative_examples(test_loader, output_dir=eval_dir)

    summary = {
        "checkpoint": args.ckpt,
        "test_metrics": test_metrics,
        "ablation_results": ablation_results,
    }
    with open(eval_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Test Split Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"Test Split mIoU (8 Classes): {test_metrics['mIoU']:.4f}")
    print(f"City Class IoU:              {test_metrics.get('iou_city', 0.0):.4f}")
    print(f"Water Class IoU:             {test_metrics.get('iou_water', 0.0):.4f}")
    print(f"Farmland Class IoU:          {test_metrics.get('iou_farmland', 0.0):.4f}")
    print("====================================================")
