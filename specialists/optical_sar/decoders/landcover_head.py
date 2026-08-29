"""Task decoder and 8-class official land-cover segmentation head for Division 4."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from specialists.optical_sar.query_intent import FiLMQueryModulator


class LandCoverTaskHead(nn.Module):
    """Multi-class Official Land-Cover Segmentation Decoder modulated by Query Intent and Physical Priors."""

    TARGET_CLASSES = [
        "background", # 0
        "farmland",   # 1
        "city",       # 2
        "village",    # 3
        "water",      # 4
        "forest",     # 5
        "road",       # 6
        "others",     # 7
    ]

    def __init__(self, feature_channels: int = 256, num_classes: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.query_film = FiLMQueryModulator(feature_channels, num_classes=num_classes)

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(feature_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.cls_pred = nn.Conv2d(64, num_classes, 1)

    def compute_physical_priors(self, optical_tensor: torch.Tensor, sar_tensor: torch.Tensor) -> torch.Tensor:
        """Compute domain-shift resilient physical index maps (NDWI, NDBI, SAR backscatter).
        
        Args:
            optical_tensor: Normalized (3, H, W) RGB tensor
            sar_tensor: Normalized (2, H, W) SAR VV/VH tensor
            
        Returns:
            Prior logits tensor shape (1, 8, H, W)
        """
        r = optical_tensor[0:1]
        g = optical_tensor[1:2]
        b = optical_tensor[2:3]

        denom_wi = torch.clamp(torch.abs(g) + torch.abs(r), min=1e-4)
        ndwi = (g - r) / denom_wi

        denom_bi = torch.clamp(torch.abs(r) + torch.abs(g), min=1e-4)
        ndbi = (r - g) / denom_bi

        sar_intensity = sar_tensor.mean(dim=0, keepdim=True)  # (1, H, W)

        # Class Priors:
        bg_prior       = torch.ones_like(sar_intensity) * 0.1                            # 0: Background
        farmland_prior = torch.clamp(g * 1.5 - r * 0.5, 0.0, 2.0)                         # 1: Farmland
        city_prior     = torch.clamp(ndbi * 2.0 + sar_intensity * 2.0, 0.0, 3.0)         # 2: City
        village_prior  = torch.clamp(ndbi * 1.0 + sar_intensity * 1.0, 0.0, 2.0)         # 3: Village
        water_prior    = torch.clamp(ndwi * 2.5 + (1.0 - sar_intensity) * 1.5, 0.0, 3.0) # 4: Water
        forest_prior   = torch.clamp(g * 2.2 - r * 1.2, 0.0, 2.5)                         # 5: Forest
        road_prior     = torch.clamp(ndbi * 1.5 + sar_intensity * 1.0, 0.0, 2.0)         # 6: Road
        others_prior   = torch.ones_like(sar_intensity) * 0.1                            # 7: Others

        priors = torch.cat([
            bg_prior, farmland_prior, city_prior, village_prior,
            water_prior, forest_prior, road_prior, others_prior
        ], dim=0).unsqueeze(0)  # (1, 8, H, W)

        return priors

    def forward(
        self,
        fused_features: torch.Tensor,
        intent_vec: torch.Tensor,
        optical_raw: Optional[torch.Tensor] = None,
        sar_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode 8-class logits and probability maps.
        
        Args:
            fused_features: Tensor shape (B, C, H_feat, W_feat)
            intent_vec: Tensor shape (B, 8) or (8,)
            optical_raw: Optional raw input tensor (B, 3, H, W)
            sar_raw: Optional raw input tensor (B, 2, H, W)
            
        Returns:
            Tuple of (logits, probabilities) shape (B, 8, H, W)
        """
        # 1. Apply FiLM modulation
        modulated = self.query_film(fused_features, intent_vec)

        # 2. Decode features to 8-class logits
        decoded = self.decoder_conv(modulated)
        logits = self.cls_pred(decoded)  # (B, 8, H_feat, W_feat)

        # 3. Add physical prior guidance in inference mode if raw tensors are provided
        if not self.training and optical_raw is not None and sar_raw is not None:
            batch_priors = []
            for b in range(optical_raw.shape[0]):
                p = self.compute_physical_priors(optical_raw[b], sar_raw[b])
                batch_priors.append(p)
            priors_tensor = torch.cat(batch_priors, dim=0)

            if priors_tensor.shape[2:] != logits.shape[2:]:
                priors_tensor = F.interpolate(priors_tensor, size=logits.shape[2:], mode="bilinear", align_corners=False)

            logits = logits + priors_tensor * 0.1

        probs = torch.softmax(logits, dim=1)
        return logits, probs
