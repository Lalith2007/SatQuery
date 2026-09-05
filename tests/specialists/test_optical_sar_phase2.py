"""Unit tests for Phase 2 of Division 4 Optical-SAR Specialist.

Verifies replaceable encoders, cross-modal attention fusion (CMAF),
8-class query intent interpreter, FiLM modulation, and multi-class task decoder.
"""

from __future__ import annotations

import pytest
import torch

from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import BaseModalityEncoder, OpticalEncoder, SarEncoder
from specialists.optical_sar.fusion import CrossModalAttentionFusion
from specialists.optical_sar.query_intent import FiLMQueryModulator, QueryIntentInterpreter


def test_optical_encoder_contract() -> None:
    """Test OpticalEncoder inherits from BaseModalityEncoder and extracts multiscale feature maps."""
    encoder = OpticalEncoder(backbone_type="resnet18", pretrained_weights=None, in_channels=3)
    assert isinstance(encoder, BaseModalityEncoder)

    dummy_input = torch.randn(1, 3, 128, 128)
    feats = encoder(dummy_input)

    assert "stride_4" in feats
    assert "stride_8" in feats
    assert "stride_16" in feats
    assert feats["stride_4"].shape == (1, 64, 32, 32)


def test_sar_encoder_contract() -> None:
    """Test SarEncoder adapts to 2-channel input and extracts multiscale feature maps."""
    encoder = SarEncoder(backbone_type="resnet18", pretrained_weights=None, in_channels=2)
    assert isinstance(encoder, BaseModalityEncoder)

    dummy_input = torch.randn(1, 2, 128, 128)
    feats = encoder(dummy_input)

    assert "stride_4" in feats
    assert feats["stride_4"].shape == (1, 64, 32, 32)


def test_cross_modal_attention_fusion() -> None:
    """Test CrossModalAttentionFusion performs bidirectional feature fusion."""
    cmaf = CrossModalAttentionFusion(optical_channels=64, sar_channels=64, out_channels=128)

    f_opt = torch.randn(1, 64, 32, 32)
    f_sar = torch.randn(1, 64, 32, 32)

    fused = cmaf(f_opt, f_sar)
    assert fused.shape == (1, 128, 32, 32)


def test_query_intent_interpreter() -> None:
    """Test QueryIntentInterpreter parses natural query keywords into 8-class activation weights."""
    interpreter = QueryIntentInterpreter()

    w_built = interpreter.parse_query_intent("Find urban built-up areas and building footprints.")
    assert w_built["city"] == 1.0
    assert w_built["water"] == 0.1

    w_water = interpreter.parse_query_intent("Identify water-covered regions and rivers.")
    assert w_water["water"] == 1.0
    assert w_water["city"] == 0.1

    w_both = interpreter.parse_query_intent("Use optical and SAR images together to identify built-up and water-covered regions.")
    assert w_both["city"] == 1.0
    assert w_both["water"] == 1.0

    vec = interpreter.get_intent_vector("Identify water")
    assert vec.shape == (8,)


def test_film_query_modulator() -> None:
    """Test FiLMQueryModulator scales and shifts feature maps using 8-class intent vectors."""
    film = FiLMQueryModulator(feature_channels=128, num_classes=8)
    feat = torch.randn(1, 128, 16, 16)
    intent = torch.tensor([0.1, 0.1, 1.0, 0.8, 1.0, 0.5, 0.5, 0.1])

    modulated = film(feat, intent)
    assert modulated.shape == (1, 128, 16, 16)


def test_query_intent_film_dimension_contract_consistency() -> None:
    """Verify dimensional contract consistency between QueryIntentInterpreter and FiLMQueryModulator across batch sizes."""
    interpreter = QueryIntentInterpreter()
    film = FiLMQueryModulator(feature_channels=256, num_classes=8)
    head = LandCoverTaskHead(feature_channels=256, num_classes=8)

    intent_vec = interpreter.get_intent_vector("Find city buildings and water bodies")
    assert intent_vec.shape == (8,)
    assert film.fc_gamma.in_features == intent_vec.shape[0]

    # Test single item (1, 8)
    single_intent = intent_vec.unsqueeze(0)
    feat_single = torch.randn(1, 256, 32, 32)
    mod_single = film(feat_single, single_intent)
    assert mod_single.shape == (1, 256, 32, 32)

    # Test batch size 16 (16, 8) — guarantees no shape multiplication error (16x4 vs 8x256)
    batch_intent = intent_vec.repeat(16, 1)
    assert batch_intent.shape == (16, 8)
    feat_batch = torch.randn(16, 256, 32, 32)
    mod_batch = film(feat_batch, batch_intent)
    assert mod_batch.shape == (16, 256, 32, 32)

    # Verify LandCoverTaskHead forward pass with batch size 16
    logits_b, probs_b = head(feat_batch, batch_intent)
    assert logits_b.shape == (16, 8, 32, 32)
    assert probs_b.shape == (16, 8, 32, 32)


def test_landcover_task_head() -> None:
    """Test LandCoverTaskHead decodes feature maps and physical priors into 8-class logits and probability maps."""
    head = LandCoverTaskHead(feature_channels=128, num_classes=8)

    fused = torch.randn(1, 128, 32, 32)
    intent = torch.tensor([[0.1, 0.5, 1.0, 0.8, 1.0, 0.5, 0.5, 0.1]])
    opt_raw = torch.randn(1, 3, 128, 128)
    sar_raw = torch.randn(1, 2, 128, 128)

    logits, probs = head(fused, intent, optical_raw=opt_raw, sar_raw=sar_raw)

    assert logits.shape == (1, 8, 32, 32)
    assert probs.shape == (1, 8, 32, 32)
    assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-4)
