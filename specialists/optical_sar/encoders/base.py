"""Abstract base class interface for replaceable optical and SAR modality encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class BaseModalityEncoder(nn.Module, ABC):
    """Abstract base class for replaceable remote-sensing modality encoders."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multiscale feature maps from input raster tensor.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor] containing feature maps at multiple scales:
            - 'stride_4': Feature map at 1/4 resolution (B, C1, H/4, W/4)
            - 'stride_8': Feature map at 1/8 resolution (B, C2, H/8, W/8)
            - 'stride_16': Feature map at 1/16 resolution (B, C3, H/16, W/16)
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> Dict[str, int]:
        """Channel dimensions for feature maps at each stride key."""
        pass
