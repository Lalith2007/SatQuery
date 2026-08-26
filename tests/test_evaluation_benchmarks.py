"""Unit tests for benchmark evaluation suites and scoring normalizers (Division 5)."""

import json
from pathlib import Path
import pytest

from evaluation.base import MetricResult, MetricType
from evaluation.benchmarks.cdvqa import CDVQAEvaluator
from evaluation.benchmarks.isro_sac import ISROSACGenericEvaluator
from evaluation.benchmarks.rsvqa import RSVQAEvaluator
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer
from evaluation.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# VRSBench Tests
# ---------------------------------------------------------------------------

def test_vrsbench_box_iou_calculation():
    """Verify standard IoU calculations on bounding boxes."""
    box_a = [0.0, 0.0, 1.0, 1.0]
    box_b = [0.0, 0.0, 1.0, 1.0]
    assert VRSBenchEvaluator.calculate_box_iou(box_a, box_b) == 1.0

    # 50% horizontal overlap
    box_c = [0.0, 0.0, 1.0, 0.5]
    box_d = [0.0, 0.0, 1.0, 1.0]
    assert VRSBenchEvaluator.calculate_box_iou(box_c, box_d) == 0.5

    # Non-overlapping boxes
    box_e = [0.0, 0.0, 0.2, 0.2]
    box_f = [0.5, 0.5, 0.8, 0.8]
    assert VRSBenchEvaluator.calculate_box_iou(box_e, box_f) == 0.0


def test_vrsbench_token_f1():
    """Verify token overlap F1 computation."""
    pred = "Airport runway with planes"
    gt = "Airport runway with airplanes"
    f1 = VRSBenchEvaluator.compute_token_f1(pred, gt)
    assert 0.6 <= f1 <= 1.0

    # Exact match
    assert VRSBenchEvaluator.compute_token_f1("dense forest", "dense forest") == 1.0
    # No overlap
    assert VRSBenchEvaluator.compute_token_f1("urban city", "water lake") == 0.0


def test_vrsbench_full_evaluation():
    """Verify VRSBench evaluation pipeline produces standard metrics."""
    evaluator = VRSBenchEvaluator()
    predictions = [
        {"answer": "Airport runway with aircraft.", "bbox": [0.1, 0.1, 0.5, 0.5]},
        {"answer": "Urban residential buildings.", "bbox": [0.2, 0.2, 0.6, 0.6]},
    ]
    ground_truths = [
        {"answer": "Airport runway with aircraft.", "bbox": [0.1, 0.1, 0.5, 0.5], "category": "presence"},
        {"answer": "Urban residential buildings.", "bbox": [0.22, 0.22, 0.61, 0.61], "category": "landcover"},
    ]

    res = evaluator.evaluate(predictions, ground_truths)
    assert res.benchmark_name == "VRSBench"
    assert res.total_samples == 2
    assert "vqa_accuracy" in res.metrics
    assert "grounding_miou" in res.metrics
    assert res.metrics["vqa_accuracy"].raw_score == 1.0
    assert res.metrics["grounding_miou"].raw_score > 0.8
    assert res.aggregate_normalized_score > 80.0


# ---------------------------------------------------------------------------
# RSVQA Tests
# ---------------------------------------------------------------------------

def test_rsvqa_evaluation():
    """Verify RSVQA evaluator handles presence, count, and comparison categories."""
    evaluator = RSVQAEvaluator()
    predictions = [
        {"answer": "yes"},
        {"answer": "4"},
        {"answer": "no"},
    ]
    ground_truths = [
        {"answer": "yes", "type": "presence"},
        {"answer": "4", "type": "count"},
        {"answer": "no", "type": "comparison"},
    ]

    res = evaluator.evaluate(predictions, ground_truths)
    assert res.benchmark_name == "RSVQA"
    assert res.total_samples == 3
    assert res.metrics["overall_accuracy"].raw_score == 1.0
    assert res.metrics["presence_accuracy"].raw_score == 1.0
    assert res.metrics["comparison_accuracy"].raw_score == 1.0
    assert res.metrics["count_rmse"].raw_score == 0.0


# ---------------------------------------------------------------------------
# CDVQA Tests
# ---------------------------------------------------------------------------

def test_cdvqa_evaluation():
    """Verify CDVQA evaluator computes BLEU, ROUGE-L, and binary change accuracy."""
    evaluator = CDVQAEvaluator()
    predictions = [
        {"answer": "Yes, new commercial buildings and paved roads appeared."},
        {"answer": "no"},
    ]
    ground_truths = [
        {"answer": "Yes, new commercial buildings and asphalt roads were constructed."},
        {"answer": "no"},
    ]

    res = evaluator.evaluate(predictions, ground_truths)
    assert res.benchmark_name == "CDVQA"
    assert "change_description_bleu1" in res.metrics
    assert "change_description_rouge_l" in res.metrics
    assert res.metrics["binary_change_accuracy"].raw_score == 1.0
    assert res.metrics["change_description_bleu1"].raw_score > 0.5


# ---------------------------------------------------------------------------
# ISRO/SAC Generic Evaluator Tests
# ---------------------------------------------------------------------------

def test_isro_sac_generic_evaluation_without_hardcoded_answers():
    """Verify ISRO/SAC generic evaluator processes Cartosat-2S and RISAT multi-sensor data."""
    evaluator = ISROSACGenericEvaluator()
    predictions = [
        {"answer": "Metallic port cranes visible through cloud cover.", "bbox": [0.2, 0.2, 0.4, 0.4], "sensor": "risat_sar"},
        {"answer": "High-resolution agricultural crop plots.", "bbox": [0.5, 0.5, 0.9, 0.9], "sensor": "cartosat_2s"},
    ]
    ground_truths = [
        {"answer": "Metallic port cranes visible through cloud cover via RISAT SAR.", "bbox": [0.2, 0.2, 0.4, 0.4], "sensor": "risat_sar"},
        {"answer": "High-resolution agricultural crop plots in Cartosat imagery.", "bbox": [0.5, 0.5, 0.9, 0.9], "sensor": "cartosat_2s"},
    ]

    res = evaluator.evaluate(predictions, ground_truths)
    assert res.benchmark_name == "ISRO_SAC_Evaluation"
    assert res.total_samples == 2
    assert "isro_sac_vqa_accuracy" in res.metrics
    assert "isro_sac_grounding_miou" in res.metrics
    assert res.metrics["isro_sac_grounding_miou"].raw_score == 1.0
    assert "risat_sar" in res.per_category_scores
    assert "cartosat_2s" in res.per_category_scores


# ---------------------------------------------------------------------------
# Normalization & Runner Tests
# ---------------------------------------------------------------------------

def test_score_normalizer_strategies():
    """Verify normalization strategies."""
    m_scale = MetricResult(name="Test Scale", metric_type=MetricType.ACCURACY, raw_score=0.85, sample_count=10)
    ScoreNormalizer.normalize_metric(m_scale, strategy=NormalizationStrategy.SCALE_100)
    assert m_scale.normalized_score == 85.0

    m_err = MetricResult(name="Test RMSE", metric_type=MetricType.RMSE, raw_score=0.0, sample_count=10)
    ScoreNormalizer.normalize_metric(m_err, strategy=NormalizationStrategy.INVERT_ERROR)
    assert m_err.normalized_score == 100.0


def test_benchmark_runner_registry():
    """Verify BenchmarkRunner registry lookup and supported benchmarks."""
    supported = BenchmarkRunner.list_supported_benchmarks()
    assert "vrsbench" in supported
    assert "rsvqa" in supported
    assert "cdvqa" in supported
    assert "isro_sac" in supported

    evaluator = BenchmarkRunner.get_evaluator("vrsbench")
    assert isinstance(evaluator, VRSBenchEvaluator)
