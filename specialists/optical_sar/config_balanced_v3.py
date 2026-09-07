"""Balanced V3 Training Configuration for Division 4 Optical-SAR Specialist.

Defines explicit hyperparameters, paths, loss balancing, sampling thresholds,
and checkpoint isolation to prevent majority-class collapse while preserving
the existing baseline checkpoints and production architecture contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingConfigV3(BaseModel):
    """Explicit, self-contained configuration for Balanced V3 Retraining."""

    # 1. Dataset & Split Paths
    data_dir: Path = Field(
        default=Path("D:/official_whu_opt_sar_dataset/official_whu_opt_sar_100scenes"),
        description="Path to official WHU-OPT-SAR 100-scene tiled dataset directory",
    )
    num_classes: int = Field(default=8, description="Number of land-cover target classes (0..7)")
    ignore_index: int = Field(default=255, description="Categorical ignore index for border padding")

    # 2. Checkpoint Isolation
    experiment_name: str = Field(
        default="experiment_balanced_v3_100scenes",
        description="Unique experiment name to prevent overwriting previous runs",
    )
    checkpoint_dir: Path = Field(
        default=Path("specialists/optical_sar/checkpoints/experiment_balanced_v3_100scenes"),
        description="Isolated directory for V3 100-scene checkpoints and logs",
    )
    best_checkpoint_name: str = Field(
        default="cmaf_landcover_balanced_v3_100scenes.pth",
        description="Filename for winning 100-scene model checkpoint",
    )

    # 3. Model Parameters & Contract
    optical_encoder: str = Field(default="resnet50", description="Optical encoder backbone")
    sar_encoder: str = Field(default="resnet50", description="SAR encoder backbone")
    in_channels_optical: int = Field(default=3, description="Optical input channels (RGB)")
    in_channels_sar: int = Field(default=2, description="SAR input channels (VV/VH)")
    fusion_channels: int = Field(default=256, description="CMAF bottleneck channels")
    expected_total_parameters: int = Field(
        default=19755144,
        description="Strict contract: must match production specialist parameter count exactly",
    )

    # 4. Training Schedule
    epochs: int = Field(default=50, description="Total epochs (Stage 1 + Stage 2)")
    warmup_epochs: int = Field(default=5, description="Stage 1 warmup epochs with frozen backbones")
    batch_size: int = Field(default=16, description="Batch size (tested optimal on RTX 4060 8GB)")
    num_workers: int = Field(default=0, description="DataLoader workers (0 is safest on Windows)")
    seed: int = Field(default=42, description="Global random seed for deterministic reproducibility")
    use_amp: bool = Field(default=True, description="Enable CUDA Automatic Mixed Precision (FP16)")

    # 5. Differential Learning Rates & Optimization
    lr_head_warmup: float = Field(default=1e-3, description="Stage 1 AdamW lr for head & neck")
    ft_lr_head: float = Field(default=5e-4, description="Stage 2 AdamW lr for head & neck")
    ft_lr_backbone: float = Field(default=2e-5, description="Stage 2 AdamW lr for unfrozen layer3 backbones")
    weight_decay: float = Field(default=1e-4, description="AdamW weight decay")
    max_grad_norm: float = Field(default=1.0, description="Gradient norm clipping ceiling")
    clip_all_trainable: bool = Field(
        default=True,
        description="Ensure all active parameters (including backbones in Stage 2) are clipped",
    )
    patience: int = Field(default=10, description="Early stopping patience epochs on Val mIoU")

    # 6. Loss & Class Balancing Strategy
    loss_strategy: str = Field(
        default="median_frequency_ce_plus_dice",
        description="Loss formulation: Median-frequency weighted CE + Balanced MultiClass Dice",
    )
    dice_loss_weight: float = Field(
        default=1.0,
        description="Weight coefficient for Dice Loss (1.0 balances spatial overlap with CE)",
    )
    background_weight_cap: float = Field(
        default=3.54,
        description="Safe cap for Background class weight to prevent numerical gradient explosion",
    )
    majority_weight_floor: float = Field(
        default=0.05,
        description="Lower floor for majority class weights (Forest, Farmland)",
    )

    # 7. Sampling Strategy
    sampler_type: str = Field(
        default="meaningful_minority_sampler",
        description="Meaningful minority presence sampler (downweights pure majority tiles)",
    )
    majority_tile_downweight: float = Field(
        default=0.35,
        description="Base weight for tiles where Forest + Farmland >= 90% of pixels",
    )
    minority_boost_road: float = Field(default=2.5, description="Weight boost for tiles with >= 100 Road pixels")
    minority_boost_city: float = Field(default=2.2, description="Weight boost for tiles with >= 250 City pixels")
    minority_boost_others: float = Field(default=1.8, description="Weight boost for tiles with >= 250 Others pixels")
    minority_boost_water: float = Field(default=1.2, description="Weight boost for tiles with >= 500 Water pixels")
    minority_boost_village: float = Field(default=1.0, description="Weight boost for tiles with >= 500 Village pixels")
    minority_boost_background: float = Field(default=2.0, description="Weight boost for tiles with >= 10 Background pixels")
    sampler_max_weight: float = Field(default=5.0, description="Maximum tile weight ceiling")

    # 8. Query Intent Conditioning
    query_conditioning_mode: str = Field(
        default="balanced_neutral",
        description="Mode: 'balanced_neutral' provides unbiased all-ones intent vector during training",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def get_default_v3_config() -> TrainingConfigV3:
    return TrainingConfigV3()
