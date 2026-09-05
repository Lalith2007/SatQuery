"""TinyCD Architecture Implementation for Bi-Temporal Remote Sensing Change Detection.

Siamese U-Net with Mixing Mask Attention Blocks (MAMB) for space-time change analysis.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MixingMaskAttentionBlock(nn.Module):
    """Mixing Mask Attention Block (MAMB) for bi-temporal cross-modality reasoning."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, in_features, 1),
            nn.Sigmoid(),
        )
        self.mix_conv = nn.Sequential(
            nn.Conv2d(in_features * 2, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f0 - f1)
        avg_out = torch.mean(diff, dim=1, keepdim=True)
        max_out, _ = torch.max(diff, dim=1, keepdim=True)
        sp_mask = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        ch_mask = self.channel_attn(diff)
        f0_mod = f0 * sp_mask * ch_mask
        f1_mod = f1 * sp_mask * ch_mask
        f_cat = torch.cat([f0_mod, f1_mod], dim=1)
        return self.mix_conv(f_cat)


class TinyCD(nn.Module):
    """Siamese U-Net with MAMB for Bi-Temporal Change Detection."""

    def __init__(self, in_channels: int = 3, base_features: int = 32) -> None:
        super().__init__()
        b = base_features
        self.stem = ConvBlock(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(b * 4, b * 8))
        self.mamb1 = MixingMaskAttentionBlock(b * 2)
        self.mamb2 = MixingMaskAttentionBlock(b * 4)
        self.mamb3 = MixingMaskAttentionBlock(b * 8)
        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 8, b * 4)
        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 4, b * 2)
        self.up0 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(b * 2, b)
        self.classifier = nn.Sequential(
            nn.Conv2d(b, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(t0)
        s1 = self.stem(t1)
        e0_1 = self.down1(s0)
        e1_1 = self.down1(s1)
        e0_2 = self.down2(e0_1)
        e1_2 = self.down2(e1_1)
        e0_3 = self.down3(e0_2)
        e1_3 = self.down3(e1_2)

        m3 = self.mamb3(e0_3, e1_3)
        m2 = self.mamb2(e0_2, e1_2)
        m1 = self.mamb1(e0_1, e1_1)

        d2 = self.dec2(torch.cat([self.up2(m3), m2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), m1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), s0], dim=1))
        return self.classifier(d0)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
