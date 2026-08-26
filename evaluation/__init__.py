"""SatQuery AI Division 5: Benchmark Evaluation Package."""

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.benchmarks.cdvqa import CDVQAEvaluator
from evaluation.benchmarks.isro_sac import ISROSACGenericEvaluator
from evaluation.benchmarks.rsvqa import RSVQAEvaluator
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer
from evaluation.runner import BenchmarkRunner

__all__ = [
    "BaseBenchmarkEvaluator",
    "BenchmarkEvaluationResult",
    "BenchmarkRunner",
    "CDVQAEvaluator",
    "ISROSACGenericEvaluator",
    "MetricResult",
    "MetricType",
    "NormalizationStrategy",
    "RSVQAEvaluator",
    "ScoreNormalizer",
    "VRSBenchEvaluator",
]
