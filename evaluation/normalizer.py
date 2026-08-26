"""Configurable score normalization and aggregation utilities for SatQuery AI Division 5."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.base import MetricResult, MetricType


class NormalizationStrategy(str, Enum):
    """Supported score normalization algorithms."""
    IDENTITY = "identity"
    SCALE_100 = "scale_100"
    MIN_MAX = "min_max"
    INVERT_ERROR = "invert_error"


class ScoreNormalizer:
    """Configurable normalizer and aggregator for heterogeneous benchmark metrics."""

    @classmethod
    def normalize_metric(
        cls,
        metric: MetricResult,
        strategy: NormalizationStrategy = NormalizationStrategy.SCALE_100,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> MetricResult:
        """Apply explicit normalization to a MetricResult, preserving raw score."""
        raw = metric.raw_score

        if strategy == NormalizationStrategy.IDENTITY:
            norm = raw
        elif strategy == NormalizationStrategy.SCALE_100:
            norm = max(0.0, min(100.0, raw * 100.0 if raw <= 1.0 else raw))
        elif strategy == NormalizationStrategy.MIN_MAX:
            denom = max_val - min_val
            norm = ((raw - min_val) / denom * 100.0) if denom > 0 else 0.0
            norm = max(0.0, min(100.0, norm))
        elif strategy == NormalizationStrategy.INVERT_ERROR:
            # For error metrics like RMSE/MAE: 0 error -> 100.0 score
            norm = (1.0 / (1.0 + max(0.0, raw))) * 100.0
        else:
            norm = raw

        metric.normalized_score = round(norm, 2)
        return metric

    @classmethod
    def compute_weighted_aggregate(
        cls,
        metrics: Dict[str, MetricResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted aggregate score (0.0 - 100.0) from normalized metrics."""
        if not metrics:
            return 0.0

        if not weights:
            # Equal weighting by default
            scores = [m.normalized_score if m.normalized_score is not None else m.raw_score for m in metrics.values()]
            return round(float(np.mean(scores)), 2)

        total_weight = 0.0
        weighted_sum = 0.0

        for name, m in metrics.items():
            w = weights.get(name, 1.0)
            score = m.normalized_score if m.normalized_score is not None else m.raw_score
            weighted_sum += score * w
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        return round(weighted_sum / total_weight, 2)
