"""Abstract base classes and schemas for SatQuery AI Benchmark Evaluation (Division 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Controlled vocabulary for evaluation metric types."""
    ACCURACY = "accuracy"
    EXACT_MATCH = "exact_match"
    TOKEN_F1 = "token_f1"
    MIOU = "miou"
    PRECISION_AT_05 = "precision_at_0.5"
    PRECISION_AT_75 = "precision_at_0.75"
    BLEU_1 = "bleu_1"
    BLEU_4 = "bleu_4"
    ROUGE_L = "rouge_l"
    METEOR = "meteor"
    RMSE = "rmse"
    MAE = "mae"
    SCORE = "score"


class MetricResult(BaseModel):
    """Container for an individual metric computation."""
    name: str = Field(description="Metric name e.g. 'vqa_accuracy', 'grounding_miou'")
    metric_type: MetricType = Field(description="Controlled metric classification")
    raw_score: float = Field(description="Raw un-normalized score")
    normalized_score: Optional[float] = Field(default=None, description="Normalized score [0.0 - 1.0 or 0 - 100]")
    sample_count: int = Field(ge=0, description="Number of evaluation samples used")
    interpretation: str = Field(default="", description="Human-readable interpretation of the metric score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional breakdown metadata")


class BenchmarkEvaluationResult(BaseModel):
    """Complete evaluation report container for a benchmark dataset execution."""
    benchmark_name: str = Field(description="Name of evaluated benchmark (e.g. VRSBench, RSVQA, CDVQA, ISRO_SAC)")
    evaluator_version: str = Field(default="1.0.0", description="Evaluator semantic version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC execution timestamp")
    total_samples: int = Field(ge=0, description="Total sample items evaluated")
    metrics: Dict[str, MetricResult] = Field(default_factory=dict, description="Calculated metric dictionary")
    aggregate_raw_score: float = Field(default=0.0, description="Weighted aggregate raw score")
    aggregate_normalized_score: float = Field(default=0.0, description="Weighted aggregate normalized score (0.0 - 100.0)")
    per_category_scores: Dict[str, float] = Field(default_factory=dict, description="Category-level breakdowns")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Benchmark execution metadata")


class BaseBenchmarkEvaluator(ABC):
    """Abstract interface for all benchmark dataset evaluators."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute evaluation against reference ground truth data and return standard results."""
        pass
