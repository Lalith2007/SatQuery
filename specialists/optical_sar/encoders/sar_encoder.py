"""SAR modality encoder implementation for Division 4."""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from specialists.optical_sar.encoders.base import BaseModalityEncoder


class SarEncoder(BaseModalityEncoder):
    """Replaceable SAR feature extraction encoder wrapping ResNet backbones."""

    def __init__(
        self,
        backbone_type: str = "resnet50",
        pretrained_weights: Optional[str] = "DEFAULT",
        in_channels: int = 2,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.in_channels_val = in_channels

        if backbone_type == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained_weights else None
            model = resnet18(weights=weights)
            self._out_channels = {"stride_4": 64, "stride_8": 128, "stride_16": 256}
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained_weights else None
            model = resnet50(weights=weights)
            self._out_channels = {"stride_4": 256, "stride_8": 512, "stride_16": 1024}

        # Modify first conv layer to accept SAR in_channels (e.g. 2 channels VV/VH)
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight[:, 0:1] = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                new_conv.weight[:, : min(in_channels, 3)] = old_conv.weight[:, : min(in_channels, 3)]
        model.conv1 = new_conv

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1  # stride 4
        self.layer2 = model.layer2  # stride 8
        self.layer3 = model.layer3  # stride 16

    @property
    def out_channels(self) -> Dict[str, int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_stem = self.stem(x)
        feat_4 = self.layer1(x_stem)
        feat_8 = self.layer2(feat_4)
        feat_16 = self.layer3(feat_8)

        return {
            "stride_4": feat_4,
            "stride_8": feat_8,
            "stride_16": feat_16,
        }
