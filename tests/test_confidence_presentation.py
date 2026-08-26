"""Unit tests for calibrated confidence presentation layer (Division 5)."""

import pytest
from presentation.confidence import ConfidenceDisplay, ConfidencePresenter, ConfidenceTier


def test_confidence_none_uncalibrated_policy():
    """Strict non-fabrication rule: None confidence must yield UNAVAILABLE without guessing."""
    disp = ConfidencePresenter.format_confidence(None)
    assert not disp.is_available
    assert disp.raw_score is None
    assert disp.formatted_percentage == "N/A"
    assert disp.tier == ConfidenceTier.UNAVAILABLE
    assert "was not provided" in disp.tier_description


def test_confidence_high_tier():
    """Verify high confidence classification threshold (>= 0.85)."""
    disp = ConfidencePresenter.format_confidence(0.92)
    assert disp.is_available
    assert disp.raw_score == 0.92
    assert disp.formatted_percentage == "92.0%"
    assert disp.formatted_decimal == "0.92"
    assert disp.tier == ConfidenceTier.HIGH
    assert disp.color_hex == "#10b981"


def test_confidence_moderate_tier():
    """Verify moderate confidence classification threshold (0.65 <= c < 0.85)."""
    disp = ConfidencePresenter.format_confidence(0.74)
    assert disp.is_available
    assert disp.formatted_percentage == "74.0%"
    assert disp.tier == ConfidenceTier.MODERATE
    assert disp.color_hex == "#f59e0b"


def test_confidence_low_tier():
    """Verify low confidence classification threshold (< 0.65)."""
    disp = ConfidencePresenter.format_confidence(0.45)
    assert disp.is_available
    assert disp.formatted_percentage == "45.0%"
    assert disp.tier == ConfidenceTier.LOW
    assert disp.color_hex == "#ef4444"


def test_confidence_clamping_out_of_range():
    """Verify out-of-range confidence scores are safely clamped between 0.0 and 1.0."""
    disp_high = ConfidencePresenter.format_confidence(1.5)
    assert disp_high.raw_score == 1.0
    assert disp_high.formatted_percentage == "100.0%"

    disp_low = ConfidencePresenter.format_confidence(-0.2)
    assert disp_low.raw_score == 0.0
    assert disp_low.formatted_percentage == "0.0%"


def test_get_tier_label_helper():
    """Verify get_tier_label convenience helper."""
    assert ConfidencePresenter.get_tier_label(0.95) == "HIGH"
    assert ConfidencePresenter.get_tier_label(0.70) == "MODERATE"
    assert ConfidencePresenter.get_tier_label(0.50) == "LOW"
    assert ConfidencePresenter.get_tier_label(None) == "UNAVAILABLE"
