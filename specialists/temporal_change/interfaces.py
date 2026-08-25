"""Pluggable interfaces for bi-temporal change detection and semantic reasoning.

Division 3 (Dheeraj) — Bi-Temporal Change Intelligence

Defines abstract base classes that allow the specialist tool to work with
any change detection backend (ChangeFormer, BIT, TinyCD, etc.) and any
semantic reasoning backend (CDVQA, RS-VLM, rule-based synthesizer, etc.)
without coupling the specialist to a specific model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class QueryIntent(str, Enum):
    """Classified intent of a user's change-related query."""
    GENERAL_CHANGE = "general_change"
    CHANGE_LOCALIZATION = "change_localization"
    BUILT_UP_CHANGE = "built_up_change"
    VEGETATION_CHANGE = "vegetation_change"
    WATER_CHANGE = "water_change"
    SPECIFIC_OBJECT_CHANGE = "specific_object_change"
    QUANTITATIVE_CHANGE = "quantitative_change"


class SemanticCapability(str, Enum):
    """Capabilities that a semantic reasoner may or may not support."""
    LAND_COVER_CLASSIFICATION = "land_cover_classification"
    CHANGE_VQA = "change_vqa"
    FREE_TEXT_DESCRIPTION = "free_text_description"
    QUANTITATIVE_SPATIAL_STATISTICS = "quantitative_spatial_statistics"


@dataclass
class ChangedRegion:
    """A single connected changed region extracted from postprocessing."""
    region_id: int
    bbox: Tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max) in pixel coords
    area_pixels: int
    centroid: Tuple[float, float]  # (y, x) center
    mean_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeDetectionOutput:
    """Standardized output from a ChangeModel backend.

    Fields are populated only when the model actually produces them.
    """
    change_probability_map: Optional[np.ndarray] = None  # H x W float [0, 1]
    binary_change_map: Optional[np.ndarray] = None  # H x W uint8 {0, 1}
    change_confidence: Optional[float] = None  # Mean probability over changed pixels
    changed_regions: List[ChangedRegion] = field(default_factory=list)
    changed_pixel_ratio: Optional[float] = None  # fraction of changed pixels
    model_name: str = ""
    model_version: str = ""
    device_used: str = "cpu"
    inference_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticReasoningOutput:
    """Standardized output from a SemanticReasoner backend."""
    answer: str = ""
    semantic_confidence: Optional[float] = None
    supported_capabilities: List[SemanticCapability] = field(default_factory=list)
    query_intent: QueryIntent = QueryIntent.GENERAL_CHANGE
    claims: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChangeModel(ABC):
    """Abstract interface for pluggable change detection backends.

    Implementations:
      - Production adapter (wraps ChangeFormer, BIT, TinyCD, etc.)
      - MockChangeModel (deterministic test outputs)
    """

    @abstractmethod
    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        ...

    @abstractmethod
    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Run change detection on a preprocessed image pair.

        Args:
            t0: Preprocessed T0 image as numpy array (H, W, C) or (C, H, W).
            t1: Preprocessed T1 image as numpy array, same shape convention as t0.

        Returns:
            ChangeDetectionOutput with at least binary_change_map populated.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources and GPU memory."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version string."""
        ...


class SemanticReasoner(ABC):
    """Abstract interface for pluggable semantic reasoning backends.

    Implementations:
      - Production reasoner (CDVQA model, RS-VLM, or controlled synthesizer)
      - MockSemanticReasoner (deterministic test outputs)
    """

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """Load any required models or resources."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the reasoner is ready."""
        ...

    @abstractmethod
    def reason_about_change(
        self,
        query: str,
        change_output: ChangeDetectionOutput,
        query_intent: QueryIntent,
        t0: Optional[np.ndarray] = None,
        t1: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> SemanticReasoningOutput:
        """Generate a natural-language answer about detected changes.

        Args:
            query: Original user query string.
            change_output: Output from the ChangeModel backend.
            query_intent: Classified intent of the query.
            t0: Optional original T0 image for visual reasoning.
            t1: Optional original T1 image for visual reasoning.

        Returns:
            SemanticReasoningOutput with answer and capability declarations.
        """
        ...

    @property
    @abstractmethod
    def supported_capabilities(self) -> List[SemanticCapability]:
        """Declare which semantic capabilities this reasoner supports."""
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        ...
