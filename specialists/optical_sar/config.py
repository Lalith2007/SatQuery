"""Configuration parameters and settings for Division 4 Optical-SAR Specialist."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for dynamic image preprocessing and domain adaptation."""
    lower_quantile: float = Field(default=0.01, ge=0.0, le=0.5, description="Lower quantile for dynamic contrast stretching")
    upper_quantile: float = Field(default=0.99, ge=0.5, le=1.0, description="Upper quantile for dynamic contrast stretching")
    target_spatial_size: Optional[List[int]] = Field(default=None, description="Optional target (H, W) for spatial resizing")
    optical_channels: List[int] = Field(default_factory=lambda: [0, 1, 2], description="Band indices for Optical (RGB)")
    sar_channels: List[int] = Field(default_factory=lambda: [0, 1], description="Band indices for SAR (VV, VH or intensity)")
    enable_speckle_filter: bool = Field(default=True, description="Apply log-transform speckle reduction on SAR imagery")
    nodata_fill_value: float = Field(default=0.0, description="Fill value for NoData regions")


class ModelConfig(BaseModel):
    """Configuration for replaceable encoders and fusion neck."""
    optical_encoder_type: str = Field(default="torchvision_resnet50", description="Encoder type for Optical modality")
    sar_encoder_type: str = Field(default="torchvision_resnet50", description="Encoder type for SAR modality")
    optical_pretrained_weights: Optional[str] = Field(default="DEFAULT", description="Pretrained weight tag for Optical encoder")
    sar_pretrained_weights: Optional[str] = Field(default="DEFAULT", description="Pretrained weight tag for SAR encoder")
    device: str = Field(default="cpu", description="Execution device ('cpu', 'cuda', 'mps')")
    in_channels_optical: int = Field(default=3, description="Input channels for Optical encoder")
    in_channels_sar: int = Field(default=2, description="Input channels for SAR encoder")
    fusion_channels: int = Field(default=256, description="Bottleneck channel dimension for Cross-Modal Attention Fusion")


class SpecialistConfig(BaseModel):
    """Master configuration for Division 4 Optical-SAR Specialist."""
    plugin_name: str = Field(default="optical_sar_cross_modal_specialist", description="Unique tool name in registry")
    version: str = Field(default="1.0.0", description="Specialist version")
    artifacts_dir: Path = Field(default=Path("artifacts_storage/optical_sar"), description="Output directory for generated artifacts")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum decision threshold")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return self.model_dump()
