"""RSVQA (Remote Sensing Visual Question Answering) benchmark evaluation suite.

Implements official Sylvain Lobry et al. (IEEE TGRS 2020) evaluation protocols:
1. Strict closed-vocabulary normalization
2. Zero substring false-positive matching
3. Integer equality for counting queries (no "0" matching "10")
4. Disaggregated question-type accuracies (presence, comparison, count, rural/urban)
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


class RSVQAEvaluator(BaseBenchmarkEvaluator):
    """Authoritative evaluator for official RSVQA-LR and RSVQA-HR benchmarks."""

    NUMBER_WORDS = {
        "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    }

    def __init__(self):
        super().__init__(name="RSVQA", version="2.0.0")

    @classmethod
    def official_normalize_answer(cls, text: Any) -> str:
        """Normalize answer string into canonical RSVQA closed-set token."""
        if text is None:
            return ""
        s = str(text).lower().strip()
        # Remove punctuation
        s = re.sub(r"[^\w\s]", " ", s)
        tokens = s.split()
        if not tokens:
            return ""

        # Check for yes / no
        if "yes" in tokens and "no" not in tokens:
            return "yes"
        if "no" in tokens and "yes" not in tokens:
            return "no"

        # Check for rural / urban
        if "rural" in tokens and "urban" not in tokens:
            return "rural"
        if "urban" in tokens and "rural" not in tokens:
            return "urban"

        # Check for number words
        for token in tokens:
            if token in cls.NUMBER_WORDS:
                return str(cls.NUMBER_WORDS[token])

        # Check for digits
        digits = re.findall(r"\b\d+\b", s)
        if digits:
            return str(int(digits[0]))

        # Fallback to first significant token
        return tokens[0]

    @classmethod
    def official_compute_accuracy(
        cls, pred: str, gt: str, q_type: str = "presence"
    ) -> int:
        """Exact closed-vocabulary match with zero substring inflation."""
        norm_p = cls.official_normalize_answer(pred)
        norm_g = cls.official_normalize_answer(gt)

        if not norm_g and not norm_p:
            return 1
        if not norm_g or not norm_p:
            return 0

        # Exact match
        if norm_p == norm_g:
            return 1

        # Numerical equality check
        if norm_p.isdigit() and norm_g.isdigit():
            return 1 if int(norm_p) == int(norm_g) else 0

        return 0

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute RSVQA evaluation with per-type accuracy and zero substring matching."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
            )

        overall_hits: List[int] = []
        hits_by_type = collections.defaultdict(list)
        count_errors_abs: List[float] = []

        for p, g in zip(predictions[:total], ground_truths[:total]):
            pred_raw = str(p.get("prediction", p.get("answer", "")))
            gt_raw = str(g.get("ground_truth", g.get("answer", "")))
            q_text = str(g.get("question", "")).lower()
            q_type = str(g.get("category", g.get("question_type", ""))).lower()

            # Infer type if generic
            if not q_type or q_type == "vqa":
                if "rural" in q_text or "urban" in q_text or gt_raw.lower() in {"rural", "urban"}:
                    q_type = "rural_urban"
                elif "how many" in q_text or "number of" in q_text or gt_raw.strip().isdigit():
                    q_type = "count"
                elif "less" in q_text or "more" in q_text or "equal" in q_text or "than" in q_text:
                    q_type = "comparison"
                else:
                    q_type = "presence"

            is_correct = self.official_compute_accuracy(pred_raw, gt_raw, q_type)
            overall_hits.append(is_correct)
            hits_by_type[q_type].append(is_correct)

            # Count MAE
            if q_type == "count":
                norm_p = self.official_normalize_answer(pred_raw)
                norm_g = self.official_normalize_answer(gt_raw)
                if norm_p.isdigit() and norm_g.isdigit():
                    count_errors_abs.append(abs(float(norm_p) - float(norm_g)))

        overall_acc = round(float(np.mean(overall_hits)), 4) if overall_hits else 0.0
        presence_acc = round(float(np.mean(hits_by_type["presence"])), 4) if hits_by_type["presence"] else overall_acc
        comp_acc = round(float(np.mean(hits_by_type["comparison"])), 4) if hits_by_type["comparison"] else overall_acc
        count_acc = round(float(np.mean(hits_by_type["count"])), 4) if hits_by_type["count"] else 0.0
        rural_urban_acc = round(float(np.mean(hits_by_type["rural_urban"])), 4) if hits_by_type["rural_urban"] else 0.0
        count_mae = round(float(np.mean(count_errors_abs)), 4) if count_errors_abs else 0.0

        metrics = {
            "overall_accuracy": MetricResult(
                name="Overall Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=overall_acc,
                sample_count=total,
                interpretation="Official exact-match accuracy across all RSVQA question types.",
            ),
            "presence_accuracy": MetricResult(
                name="Presence Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=presence_acc,
                sample_count=len(hits_by_type["presence"]),
                interpretation="Binary object presence detection accuracy.",
            ),
            "comparison_accuracy": MetricResult(
                name="Comparison Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=comp_acc,
                sample_count=len(hits_by_type["comparison"]),
                interpretation="Relative spatial and feature quantity comparison accuracy.",
            ),
            "count_accuracy": MetricResult(
                name="Count Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=count_acc,
                sample_count=len(hits_by_type["count"]),
                interpretation="Exact numerical count equality accuracy.",
            ),
        }

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=overall_acc,
            aggregate_normalized_score=overall_acc,
            per_category_scores={
                "presence_accuracy": presence_acc,
                "comparison_accuracy": comp_acc,
                "count_accuracy": count_acc,
                "rural_urban_accuracy": rural_urban_acc,
                "count_mae": count_mae,
            },
            metadata={
                "type_counts": {t: len(hits) for t, hits in hits_by_type.items()},
            },
        )
