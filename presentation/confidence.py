"""Calibrated confidence presentation layer for SatQuery AI Division 5.

Provides strictly non-fabricated confidence handling, formatted representations,
qualitative tier classification, and metadata descriptors for UI rendering and reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ConfidenceTier(str, Enum):
    """Qualitative confidence classification tiers based on explicit thresholds."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNAVAILABLE = "UNAVAILABLE"


class ConfidenceDisplay(BaseModel):
    """Structured confidence presentation model."""
    raw_score: Optional[float] = Field(default=None, description="Original raw score [0.0 - 1.0] or None if uncalibrated")
    is_available: bool = Field(description="True if calibrated score was provided by specialist")
    formatted_percentage: str = Field(description="Formatted percentage string or 'N/A'")
    formatted_decimal: str = Field(description="Formatted 2-decimal float string or 'N/A'")
    tier: ConfidenceTier = Field(description="Qualitative confidence tier")
    tier_description: str = Field(description="Operational explanation of the confidence level")
    color_hex: str = Field(description="Visual accent color hex code for UI display")
    badge_style: Dict[str, str] = Field(default_factory=dict, description="CSS styles for badge rendering")


class ConfidencePresenter:
    """Formatter and presenter for model prediction confidence scores."""

    HIGH_THRESHOLD: float = 0.85
    MODERATE_THRESHOLD: float = 0.65

    @classmethod
    def format_confidence(cls, confidence: Optional[float]) -> ConfidenceDisplay:
        """Format raw specialist confidence into a structured, presentation-ready object.
        
        Strict Non-Fabrication Rule:
        If confidence is None, this method strictly outputs UNAVAILABLE without guessing.
        """
        if confidence is None:
            return ConfidenceDisplay(
                raw_score=None,
                is_available=False,
                formatted_percentage="N/A",
                formatted_decimal="N/A",
                tier=ConfidenceTier.UNAVAILABLE,
                tier_description="Confidence score was not provided or calibrated for this specialist output.",
                color_hex="#9ca3af",
                badge_style={
                    "background": "rgba(156, 163, 175, 0.15)",
                    "color": "#9ca3af",
                    "border": "1px solid rgba(156, 163, 175, 0.3)",
                },
            )

        # Clamp raw confidence to valid range [0.0, 1.0]
        clamped = max(0.0, min(1.0, float(confidence)))
        pct_str = f"{clamped * 100:.1f}%"
        dec_str = f"{clamped:.2f}"

        if clamped >= cls.HIGH_THRESHOLD:
            tier = ConfidenceTier.HIGH
            desc = "High confidence: Strong model certainty and spatial feature agreement."
            color = "#10b981"  # Emerald green
            badge_style = {
                "background": "rgba(16, 185, 129, 0.15)",
                "color": "#10b981",
                "border": "1px solid rgba(16, 185, 129, 0.35)",
            }
        elif clamped >= cls.MODERATE_THRESHOLD:
            tier = ConfidenceTier.MODERATE
            desc = "Moderate confidence: Reasonable certainty; human verification recommended for critical applications."
            color = "#f59e0b"  # Amber
            badge_style = {
                "background": "rgba(245, 158, 11, 0.15)",
                "color": "#f59e0b",
                "border": "1px solid rgba(245, 158, 11, 0.35)",
            }
        else:
            tier = ConfidenceTier.LOW
            desc = "Low confidence: Substantial uncertainty or potential ambiguity in the input imagery."
            color = "#ef4444"  # Red
            badge_style = {
                "background": "rgba(239, 68, 68, 0.15)",
                "color": "#ef4444",
                "border": "1px solid rgba(239, 68, 68, 0.35)",
            }

        return ConfidenceDisplay(
            raw_score=round(clamped, 4),
            is_available=True,
            formatted_percentage=pct_str,
            formatted_decimal=dec_str,
            tier=tier,
            tier_description=desc,
            color_hex=color,
            badge_style=badge_style,
        )

    @classmethod
    def get_tier_label(cls, confidence: Optional[float]) -> str:
        """Convenience method to return human-readable tier label."""
        return cls.format_confidence(confidence).tier.value
