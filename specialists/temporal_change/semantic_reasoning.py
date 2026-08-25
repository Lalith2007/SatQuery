"""Semantic Reasoning for Division 3: Bi-Temporal Change Intelligence.

Pluggable semantic reasoning backends implementing the SemanticReasoner interface.

MockSemanticReasoner: Deterministic test backend.
SpatialMetricSynthesizer: Production-grade quantitative spatial statistics
  reasoning that honestly reports what binary change detection can and cannot infer.

Scientific honesty: Binary change detection identifies spatial change presence/extent.
It CANNOT determine specific land-cover transitions (e.g. "vegetation to built-up")
unless an explicit semantic classifier or verified RS-VLM component is active.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from core.logging import get_logger
from specialists.temporal_change.interfaces import (
    ChangeDetectionOutput,
    QueryIntent,
    SemanticCapability,
    SemanticReasoner,
    SemanticReasoningOutput,
)

logger = get_logger("temporal_change.semantic_reasoning")


def classify_query_intent(query: str) -> QueryIntent:
    """Classify a user's change-related query into a structured intent.

    This is a lightweight keyword-based classifier. It does NOT claim
    capabilities unsupported by the active semantic backend.
    """
    q = query.lower().strip()

    # Localization queries
    if any(kw in q for kw in ["where", "locate", "location", "region", "area where"]):
        return QueryIntent.CHANGE_LOCALIZATION

    # Built-up / urban queries
    if any(kw in q for kw in ["built-up", "built up", "urban", "construction", "building", "infrastructure"]):
        return QueryIntent.BUILT_UP_CHANGE

    # Vegetation queries
    if any(kw in q for kw in ["vegetation", "forest", "tree", "green", "canopy", "deforestation", "plant"]):
        return QueryIntent.VEGETATION_CHANGE

    # Water queries
    if any(kw in q for kw in ["water", "flood", "reservoir", "river", "lake", "pond"]):
        return QueryIntent.WATER_CHANGE

    # Quantitative queries
    if any(kw in q for kw in ["how much", "percentage", "how many", "ratio", "area"]):
        return QueryIntent.QUANTITATIVE_CHANGE

    return QueryIntent.GENERAL_CHANGE


class MockSemanticReasoner(SemanticReasoner):
    """Deterministic mock semantic reasoner for testing.

    Returns plausible answers clearly labeled as mock output.
    """

    def __init__(self) -> None:
        self._ready = False

    def initialize(self, **kwargs: Any) -> None:
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def reason_about_change(
        self,
        query: str,
        change_output: ChangeDetectionOutput,
        query_intent: QueryIntent,
        t0: Optional[np.ndarray] = None,
        t1: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> SemanticReasoningOutput:
        ratio = change_output.changed_pixel_ratio or 0.0
        n_regions = len(change_output.changed_regions)
        pct = round(ratio * 100, 1)

        intent_answers = {
            QueryIntent.GENERAL_CHANGE: (
                f"[Mock] Bi-temporal analysis detected surface changes across {pct}% of the scene "
                f"in {n_regions} distinct spatial cluster(s)."
            ),
            QueryIntent.CHANGE_LOCALIZATION: (
                f"[Mock] Changes are localized in {n_regions} cluster(s). "
                "See the change map and bounding boxes for spatial details."
            ),
            QueryIntent.BUILT_UP_CHANGE: (
                f"[Mock] Surface changes detected across {pct}% of the scene. "
                "Note: Binary change detection cannot confirm whether changes represent "
                "built-up expansion. A semantic classifier is required for land-cover identification."
            ),
            QueryIntent.VEGETATION_CHANGE: (
                f"[Mock] Surface changes detected across {pct}% of the scene. "
                "Note: Binary change detection cannot confirm vegetation-specific transitions."
            ),
            QueryIntent.WATER_CHANGE: (
                f"[Mock] Surface changes detected across {pct}% of the scene. "
                "Note: Binary change detection cannot confirm water body changes."
            ),
            QueryIntent.QUANTITATIVE_CHANGE: (
                f"[Mock] Approximately {pct}% of the scene area has changed, "
                f"distributed across {n_regions} connected region(s)."
            ),
        }

        answer = intent_answers.get(query_intent, intent_answers[QueryIntent.GENERAL_CHANGE])

        return SemanticReasoningOutput(
            answer=answer,
            semantic_confidence=None,  # Mock does not produce calibrated confidence
            supported_capabilities=[SemanticCapability.QUANTITATIVE_SPATIAL_STATISTICS],
            query_intent=query_intent,
            claims=[{"type": "spatial_statistics", "changed_pct": pct, "regions": n_regions}],
            limitations=[
                "Mock backend: no trained semantic model active.",
                "Cannot determine specific land-cover transitions.",
            ],
            metadata={"is_mock": True},
        )

    @property
    def supported_capabilities(self) -> List[SemanticCapability]:
        return [SemanticCapability.QUANTITATIVE_SPATIAL_STATISTICS]

    def cleanup(self) -> None:
        self._ready = False


class SpatialMetricSynthesizer(SemanticReasoner):
    """Production semantic reasoner based on quantitative spatial statistics.

    This reasoner honestly reports what binary change detection can infer:
      - Total changed area (pixels, percentage)
      - Number and size of changed clusters
      - Spatial distribution description
      - Bounding box locations

    It DOES NOT claim land-cover transitions unless a validated semantic
    component is provided. Unsupported semantic claims are explicitly omitted.
    """

    def __init__(self) -> None:
        self._ready = False

    def initialize(self, **kwargs: Any) -> None:
        self._ready = True
        logger.info("SpatialMetricSynthesizer initialized (quantitative spatial statistics)")

    def is_ready(self) -> bool:
        return self._ready

    def reason_about_change(
        self,
        query: str,
        change_output: ChangeDetectionOutput,
        query_intent: QueryIntent,
        t0: Optional[np.ndarray] = None,
        t1: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> SemanticReasoningOutput:
        ratio = change_output.changed_pixel_ratio or 0.0
        pct = round(ratio * 100, 1)
        regions = change_output.changed_regions
        n_regions = len(regions)
        conf = change_output.change_confidence

        # Build spatial description
        if n_regions == 0 or pct < 0.1:
            answer = "No significant surface changes were detected between the two acquisitions."
            claims = [{"type": "no_change", "changed_pct": pct}]
        else:
            # Describe spatial distribution
            spatial_desc = self._describe_spatial_distribution(regions, change_output)
            answer = (
                f"Bi-temporal analysis detected surface changes across approximately {pct}% "
                f"of the scene, distributed across {n_regions} distinct spatial cluster(s). "
                f"{spatial_desc}"
            )
            claims = [
                {"type": "spatial_statistics", "changed_pct": pct, "regions": n_regions},
            ]
            for i, r in enumerate(regions[:5]):
                claims.append({
                    "type": "region",
                    "region_id": r.region_id,
                    "area_pixels": r.area_pixels,
                    "bbox": list(r.bbox),
                    "mean_confidence": round(r.mean_confidence, 3),
                })

        # Handle semantic intent queries with honest limitations
        limitations: List[str] = []

        if query_intent in (
            QueryIntent.BUILT_UP_CHANGE,
            QueryIntent.VEGETATION_CHANGE,
            QueryIntent.WATER_CHANGE,
        ):
            semantic_note = (
                "Note: The active change detection model provides binary spatial change "
                "detection only. Specific land-cover class transitions (e.g., vegetation "
                "to built-up, water expansion) cannot be confirmed without a validated "
                "semantic classification component."
            )
            answer = f"{answer} {semantic_note}"
            limitations.append(
                "Binary change detector active: land-cover transition claims are not supported."
            )

        if query_intent == QueryIntent.CHANGE_LOCALIZATION and n_regions > 0:
            loc_desc = self._describe_locations(regions[:5])
            answer = f"{answer} {loc_desc}"

        return SemanticReasoningOutput(
            answer=answer,
            semantic_confidence=None,  # Spatial synthesis does not produce semantic confidence
            supported_capabilities=self.supported_capabilities,
            query_intent=query_intent,
            claims=claims,
            limitations=limitations,
            metadata={
                "is_mock": False,
                "reasoner": "SpatialMetricSynthesizer",
                "change_confidence": conf,
            },
        )

    def _describe_spatial_distribution(
        self, regions: List, change_output: ChangeDetectionOutput
    ) -> str:
        """Generate a human-readable spatial distribution description."""
        if not regions:
            return ""

        largest = regions[0]
        total_changed = sum(r.area_pixels for r in regions)
        largest_pct = round(largest.area_pixels / total_changed * 100, 1) if total_changed > 0 else 0

        if len(regions) == 1:
            return f"The change is concentrated in a single cluster of {largest.area_pixels} pixels."

        return (
            f"The largest cluster contains {largest.area_pixels} pixels "
            f"({largest_pct}% of total changed area)."
        )

    def _describe_locations(self, regions: List) -> str:
        """Generate spatial location descriptions from bounding boxes."""
        if not regions:
            return ""

        descriptions = []
        for i, r in enumerate(regions[:3]):
            y_min, x_min, y_max, x_max = r.bbox
            descriptions.append(
                f"Region {i+1}: bounding box [{y_min}, {x_min}, {y_max}, {x_max}] "
                f"({r.area_pixels} pixels)"
            )

        return "Key changed regions: " + "; ".join(descriptions) + "."

    @property
    def supported_capabilities(self) -> List[SemanticCapability]:
        return [SemanticCapability.QUANTITATIVE_SPATIAL_STATISTICS]

    def cleanup(self) -> None:
        self._ready = False
