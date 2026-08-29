"""Intermediate Cross-Modal Attention Fusion (CMAF) module for Division 4."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCrossAttention(nn.Module):
    """Bidirectional spatial cross-attention between Optical and SAR feature maps with QK normalization and FP32 softmax."""

    def __init__(self, in_channels: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query_opt = nn.Conv2d(in_channels, embed_dim, 1)
        self.key_sar = nn.Conv2d(in_channels, embed_dim, 1)
        self.value_sar = nn.Conv2d(in_channels, in_channels, 1)

        self.query_sar = nn.Conv2d(in_channels, embed_dim, 1)
        self.key_opt = nn.Conv2d(in_channels, embed_dim, 1)
        self.value_opt = nn.Conv2d(in_channels, in_channels, 1)

        self.norm_q_opt = nn.LayerNorm(embed_dim)
        self.norm_k_sar = nn.LayerNorm(embed_dim)
        self.norm_q_sar = nn.LayerNorm(embed_dim)
        self.norm_k_opt = nn.LayerNorm(embed_dim)

        self.scale = embed_dim ** -0.5

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = f_opt.shape

        # 1. Optical queries SAR
        q_opt = self.query_opt(f_opt).view(b, self.embed_dim, h * w).transpose(1, 2)  # (B, HW, D)
        k_sar = self.key_sar(f_sar).view(b, self.embed_dim, h * w).transpose(1, 2)    # (B, HW, D)
        v_sar = self.value_sar(f_sar).view(b, -1, h * w).transpose(1, 2)               # (B, HW, C)

        q_opt = self.norm_q_opt(q_opt)
        k_sar = self.norm_k_sar(k_sar)

        # Compute cross-attention softmax in float32 for AMP numerical stability
        scores_opt_sar = (torch.bmm(q_opt, k_sar.transpose(1, 2)) * self.scale).float()
        attn_opt_sar = torch.softmax(scores_opt_sar, dim=-1).to(q_opt.dtype)
        f_sar_enhanced = torch.bmm(attn_opt_sar, v_sar).transpose(1, 2).view(b, c, h, w)

        # 2. SAR queries Optical
        q_sar = self.query_sar(f_sar).view(b, self.embed_dim, h * w).transpose(1, 2)  # (B, HW, D)
        k_opt = self.key_opt(f_opt).view(b, self.embed_dim, h * w).transpose(1, 2)    # (B, HW, D)
        v_opt = self.value_opt(f_opt).view(b, -1, h * w).transpose(1, 2)               # (B, HW, C)

        q_sar = self.norm_q_sar(q_sar)
        k_opt = self.norm_k_opt(k_opt)

        scores_sar_opt = (torch.bmm(q_sar, k_opt.transpose(1, 2)) * self.scale).float()
        attn_sar_opt = torch.softmax(scores_sar_opt, dim=-1).to(q_sar.dtype)
        f_opt_enhanced = torch.bmm(attn_sar_opt, v_opt).transpose(1, 2).view(b, c, h, w)

        return f_opt + f_opt_enhanced, f_sar + f_sar_enhanced


class CrossModalAttentionFusion(nn.Module):
    """Intermediate Cross-Modal Attention Fusion (CMAF) module for multiscale feature maps."""

    def __init__(self, optical_channels: int, sar_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.proj_opt = nn.Conv2d(optical_channels, out_channels, 1)
        self.proj_sar = nn.Conv2d(sar_channels, out_channels, 1)

        self.cross_attn = SpatialCrossAttention(out_channels, embed_dim=64)

        self.gate_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        f_opt_proj = self.proj_opt(f_opt)
        f_sar_proj = self.proj_sar(f_sar)

        f_opt_enh, f_sar_enh = self.cross_attn(f_opt_proj, f_sar_proj)

        concat_feat = torch.cat([f_opt_enh, f_sar_enh], dim=1)
        gate = self.gate_fusion(concat_feat)

        fused = gate * f_opt_enh + (1.0 - gate) * f_sar_enh
        return self.out_conv(fused)
