"""RSVQA (Remote Sensing Visual Question Answering) benchmark evaluation suite."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


class RSVQAEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for RSVQA Low-Resolution (LR) and High-Resolution (HR) datasets."""

    def __init__(self):
        super().__init__(name="RSVQA", version="1.0.0")

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardize text for VQA match."""
        if not text:
            return ""
        return re.sub(r"[^\w\s]", "", str(text).lower().strip())

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract first numerical value from count answer text."""
        words_to_num = {
            "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "many": 15, "few": 3,
        }
        cleaned = text.lower().strip()
        for word, val in words_to_num.items():
            if word in cleaned:
                return float(val)
        matches = re.findall(r"\d+", cleaned)
        if matches:
            return float(matches[0])
        return None

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute RSVQA benchmark evaluation across presence, count, and comparison questions."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
            )

        presence_hits: List[int] = []
        comparison_hits: List[int] = []
        overall_hits: List[int] = []
        count_errors_sq: List[float] = []
        count_errors_abs: List[float] = []

        for pred, gt in zip(predictions[:total], ground_truths[:total]):
            pred_ans = self.clean_text(pred.get("answer") or pred.get("prediction") or "")
            gt_ans = self.clean_text(gt.get("answer") or gt.get("ground_truth") or "")
            q_type = str(gt.get("type") or gt.get("category") or "presence").lower()

            is_match = 1 if pred_ans == gt_ans or (pred_ans in gt_ans or gt_ans in pred_ans) else 0
            overall_hits.append(is_match)

            if "presence" in q_type or "exist" in q_type or gt_ans in {"yes", "no"}:
                presence_hits.append(is_match)
            elif "comp" in q_type:
                comparison_hits.append(is_match)
            elif "count" in q_type or "number" in q_type:
                p_num = self.extract_number(pred_ans)
                g_num = self.extract_number(gt_ans)
                if p_num is not None and g_num is not None:
                    diff = abs(p_num - g_num)
                    count_errors_abs.append(diff)
                    count_errors_sq.append(diff ** 2)

        overall_acc = round(float(np.mean(overall_hits)), 4) if overall_hits else 0.0
        presence_acc = round(float(np.mean(presence_hits)), 4) if presence_hits else overall_acc
        comp_acc = round(float(np.mean(comparison_hits)), 4) if comparison_hits else overall_acc
        
        count_mae = round(float(np.mean(count_errors_abs)), 4) if count_errors_abs else 0.0
        count_rmse = round(float(np.sqrt(np.mean(count_errors_sq))), 4) if count_errors_sq else 0.0

        metrics = {
            "overall_accuracy": MetricResult(
                name="Overall VQA Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=overall_acc,
                sample_count=total,
                interpretation="Standard classification accuracy across all RSVQA question types.",
            ),
            "presence_accuracy": MetricResult(
                name="Presence Question Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=presence_acc,
                sample_count=len(presence_hits),
                interpretation="Binary object presence detection accuracy.",
            ),
            "comparison_accuracy": MetricResult(
                name="Comparison Question Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=comp_acc,
                sample_count=len(comparison_hits),
                interpretation="Relative spatial and feature quantity comparison accuracy.",
            ),
            "count_rmse": MetricResult(
                name="Count Root Mean Squared Error (RMSE)",
                metric_type=MetricType.RMSE,
                raw_score=count_rmse,
                sample_count=len(count_errors_sq),
                interpretation="Root mean squared deviation on numerical counting questions (lower is better).",
            ),
        }

        # Normalize metrics
        ScoreNormalizer.normalize_metric(metrics["overall_accuracy"], strategy=NormalizationStrategy.SCALE_100)
        ScoreNormalizer.normalize_metric(metrics["presence_accuracy"], strategy=NormalizationStrategy.SCALE_100)
        ScoreNormalizer.normalize_metric(metrics["comparison_accuracy"], strategy=NormalizationStrategy.SCALE_100)
        ScoreNormalizer.normalize_metric(metrics["count_rmse"], strategy=NormalizationStrategy.INVERT_ERROR)

        agg_norm = ScoreNormalizer.compute_weighted_aggregate(metrics, weights={"overall_accuracy": 0.4, "presence_accuracy": 0.3, "comparison_accuracy": 0.3})

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=overall_acc,
            aggregate_normalized_score=agg_norm,
            per_category_scores={
                "presence_accuracy": presence_acc,
                "comparison_accuracy": comp_acc,
                "count_mae": count_mae,
            },
        )
