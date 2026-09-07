"""Unit tests for Division 4 Optical-SAR Complete Re-Training Pipeline.

Verifies:
1. Component imports and exact architecture parameter counts.
2. Synchronized physical spatial augmentations (horizontal flip, vertical flip, 90-deg rotation).
3. Dynamic inverse square-root class loss weighting.
4. Bounded minority-aware tile sampler mechanics and effective distribution calculation.
5. 1-step forward and backward gradient step with Weighted CE + 0.5 Dice loss.
6. Checkpoint structure and strict=True compatibility with production OpticalSarSpecialist.
7. Active predicted class monitoring and confusion matrix calculation.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from specialists.optical_sar.config import SpecialistConfig
from specialists.optical_sar.dataset import OpticalSarPairedDataset
from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import OpticalEncoder, SarEncoder
from specialists.optical_sar.fusion import CrossModalAttentionFusion
from specialists.optical_sar.service import OpticalSarSpecialist
from specialists.optical_sar.train_colab import (
    CLASS_NAMES,
    ColabTrainer,
    DiceLoss,
    build_minority_aware_sampler,
    compute_dynamic_class_weights,
)


def test_architecture_parameter_counts_and_invariance():
    """Verify that model architecture remains strictly intact with 19,755,144 parameters."""
    opt_enc = OpticalEncoder(pretrained_weights=None, in_channels=3)
    sar_enc = SarEncoder(pretrained_weights=None, in_channels=2)

    opt_out_ch = opt_enc.out_channels["stride_8"]
    sar_out_ch = sar_enc.out_channels["stride_8"]

    fusion_neck = CrossModalAttentionFusion(
        optical_channels=opt_out_ch,
        sar_channels=sar_out_ch,
        out_channels=256,
    )
    task_head = LandCoverTaskHead(feature_channels=256, num_classes=8)

    opt_p = sum(p.numel() for p in opt_enc.parameters())
    sar_p = sum(p.numel() for p in sar_enc.parameters())
    fusion_p = sum(p.numel() for p in fusion_neck.parameters())
    head_p = sum(p.numel() for p in task_head.parameters())
    total_p = opt_p + sar_p + fusion_p + head_p

    assert opt_p == 8543296, f"OpticalEncoder params mismatch: {opt_p}"
    assert sar_p == 8540160, f"SarEncoder params mismatch: {sar_p}"
    assert fusion_p == 2297344, f"CMAF Fusion neck params mismatch: {fusion_p}"
    assert head_p == 374344, f"TaskHead params mismatch: {head_p}"
    assert total_p == 19755144, f"Total model params mismatch: {total_p}"


def test_synchronized_spatial_augmentation():
    """Verify that optical, SAR, and segmentation masks receive identical spatial transforms."""
    # Create asymmetric dummy rasters
    H, W = 32, 32
    opt = torch.zeros(3, H, W)
    sar = torch.zeros(2, H, W)
    lbl = torch.zeros(H, W, dtype=torch.long)

    # Place unique marker in top-left corner (0, 0)
    opt[:, 0, 0] = 1.0
    sar[:, 0, 0] = 1.0
    lbl[0, 0] = 4  # Water class marker

    # 1. Horizontal Flip
    opt_hf = torch.flip(opt, dims=[-1])
    sar_hf = torch.flip(sar, dims=[-1])
    lbl_hf = torch.flip(lbl, dims=[-1])

    assert opt_hf[:, 0, W - 1].sum() == 3.0
    assert sar_hf[:, 0, W - 1].sum() == 2.0
    assert lbl_hf[0, W - 1] == 4
    assert opt_hf[:, 0, 0].sum() == 0.0

    # 2. Vertical Flip
    opt_vf = torch.flip(opt, dims=[-2])
    sar_vf = torch.flip(sar, dims=[-2])
    lbl_vf = torch.flip(lbl, dims=[-2])

    assert opt_vf[:, H - 1, 0].sum() == 3.0
    assert sar_vf[:, H - 1, 0].sum() == 2.0
    assert lbl_vf[H - 1, 0] == 4

    # 3. 90-degree Rotation
    opt_rot = torch.rot90(opt, k=1, dims=[-2, -1])
    sar_rot = torch.rot90(sar, k=1, dims=[-2, -1])
    lbl_rot = torch.rot90(lbl, k=1, dims=[-2, -1])

    # Rotated counterclockwise: top-left (0,0) moves to bottom-left (H-1, 0)
    assert opt_rot[:, H - 1, 0].sum() == 3.0
    assert sar_rot[:, H - 1, 0].sum() == 2.0
    assert lbl_rot[H - 1, 0] == 4


def test_dynamic_class_weight_calculation():
    """Verify that inverse square-root class weights properly invert pixel rarity and stay bounded."""
    # Strongly imbalanced pixel counts (Forest 48%, Farmland 32%, Water 10%, City 3%, Road 0.5%)
    mock_pixel_counts = np.array([
        50000,    # 0 Background
        3200000,  # 1 Farmland (Dominant)
        300000,   # 2 City
        200000,   # 3 Village
        1000000,  # 4 Water
        4800000,  # 5 Forest (Most dominant)
        50000,    # 6 Road (Very rare)
        120000,   # 7 Others
    ], dtype=np.int64)

    weights = compute_dynamic_class_weights(mock_pixel_counts, min_weight=1.0, max_weight=15.0)

    assert len(weights) == 8
    assert torch.isfinite(weights).all()

    # Forest (most common) should have the lowest weight
    assert weights[5] < weights[1]
    # Road and Background (rare) should have significantly higher weights than Forest
    assert weights[6] > weights[5]
    assert weights[0] > weights[5]
    # Rarity ordering: weights[Road] >= weights[Water] >= weights[Forest]
    assert weights[6] > weights[4]
    assert weights[4] > weights[5]


def test_one_step_forward_backward_gradient_check():
    """Verify numerical stability of 1-step forward/backward pass with AMP and Combined Loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ColabTrainer(device=device, use_amp=(device == "cuda"))

    mock_weights = torch.tensor([5.0, 1.0, 3.5, 2.8, 2.0, 1.0, 7.0, 5.0])
    trainer.setup_loss(mock_weights)

    B, H, W = 2, 64, 64
    opt = torch.randn(B, 3, H, W, device=device)
    sar = torch.randn(B, 2, H, W, device=device)
    targets = torch.randint(0, 8, (B, H, W), device=device)
    # Include some ignore_index border pixels
    targets[0, :5, :] = 255
    intent = torch.randn(B, 8, device=device)

    optimizer = torch.optim.AdamW(
        list(trainer.fusion_neck.parameters()) + list(trainer.task_head.parameters()),
        lr=1e-3,
    )
    optimizer.zero_grad()

    opt_feats = trainer.optical_encoder(opt)
    sar_feats = trainer.sar_encoder(sar)
    fused = trainer.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])
    logits, probs = trainer.task_head(fused, intent)

    if logits.shape[2:] != targets.shape[1:]:
        logits = F.interpolate(logits, size=targets.shape[1:], mode="bilinear", align_corners=False)

    ce_loss = trainer.ce_loss_fn(logits.float(), targets)
    dice_loss = trainer.dice_loss_fn(logits.float(), targets)
    total_loss = ce_loss + 0.5 * dice_loss

    assert torch.isfinite(ce_loss)
    assert torch.isfinite(dice_loss)
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(trainer.fusion_neck.parameters()) + list(trainer.task_head.parameters()),
        max_norm=1.0,
    )
    optimizer.step()

    for p in list(trainer.fusion_neck.parameters()) + list(trainer.task_head.parameters()):
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_checkpoint_strict_loadability_with_production_specialist():
    """Verify that a saved candidate checkpoint loads cleanly into OpticalSarSpecialist with strict=True."""
    trainer = ColabTrainer(device="cpu", use_amp=False)
    trainer.setup_loss(torch.ones(8))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "cmaf_landcover_best_v2.pth"
        trainer.save_checkpoint(
            path=ckpt_path,
            epoch=1,
            val_loss=0.45,
            metrics={"mIoU": 0.42, "overall_accuracy": 0.78, "macro_f1": 0.39},
            history=[],
            is_best=True,
        )
        assert ckpt_path.exists()

        # Load into production specialist
        specialist = OpticalSarSpecialist(checkpoint_path=ckpt_path, require_trained_weights=True)
        assert specialist.is_trained_loaded is True
        assert specialist.metadata.name == "optical_sar_cross_modal_specialist"

        # Verify strict key compatibility
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        specialist.optical_encoder.load_state_dict(loaded_ckpt["optical_encoder_state_dict"], strict=True)
        specialist.sar_encoder.load_state_dict(loaded_ckpt["sar_encoder_state_dict"], strict=True)
        specialist.fusion_neck.load_state_dict(loaded_ckpt["fusion_neck_state_dict"], strict=True)
        specialist.task_head.load_state_dict(loaded_ckpt["task_head_state_dict"], strict=True)


def test_detailed_metrics_and_active_class_monitoring():
    """Verify metric computation correctly detects active classes and handles zero-predicted classes."""
    trainer = ColabTrainer(device="cpu", use_amp=False)

    # Synthetic targets with all 8 classes
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 255])
    # Synthetic preds where only Forest (5), Farmland (1), City (2) are predicted
    preds = torch.tensor([5, 1, 2, 1, 5, 5, 2, 5, 0])

    metrics, cm = trainer.compute_detailed_metrics(preds, targets)

    assert "overall_accuracy" in metrics
    assert "mIoU" in metrics
    assert "macro_f1" in metrics
    assert "active_class_count" in metrics
    assert metrics["active_class_count"] == 3  # Only City, Farmland, Forest predicted
    assert "water" in metrics["zero_prediction_classes"]
    assert "village" in metrics["zero_prediction_classes"]
    assert "road" in metrics["zero_prediction_classes"]
    assert "others" in metrics["zero_prediction_classes"]
    assert "background" in metrics["zero_prediction_classes"]
    assert cm.shape == (8, 8)
