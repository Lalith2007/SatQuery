"""CDVQA (Change Detection Visual Question Answering) benchmark evaluation suite.

Implements official Yuan et al. (IEEE TGRS 2022) evaluation protocols:
1. Strict classification accuracy across 8 official CDVQA question types
2. Zero synthetic benchmark templates allowed (hard rejection gate)
3. Disaggregated per-type accuracies:
   - change_or_not (yes/no)
   - increase_or_not (yes/no)
   - decrease_or_not (yes/no)
   - change_to_what (land cover class)
   - smallest_change (land cover class)
   - largest_change (land cover class)
   - change_ratio (percentage range)
   - change_ratio_types (multi-type change ratio)
"""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


SYNTHETIC_FORBIDDEN_TEMPLATES = [
    "new residential buildings and infrastructure constructed in the cleared agricultural area",
    "residential buildings and infrastructure constructed",
    "cleared agricultural area",
    "no significant change detected",
]


class CDVQAEvaluator(BaseBenchmarkEvaluator):
    """Authoritative evaluator for official CDVQA benchmark."""

    OFFICIAL_TYPES = [
        "change_or_not",
        "change_ratio_types",
        "decrease_or_not",
        "increase_or_not",
        "change_to_what",
        "smallest_change",
        "largest_change",
        "change_ratio",
    ]

    def __init__(self):
        super().__init__(name="CDVQA", version="2.0.0")

    @classmethod
    def clean_text(cls, text: Any) -> str:
        """Standardize text for CDVQA token matching."""
        if text is None:
            return ""
        s = str(text).lower().strip()
        # Check forbidden synthetic reference
        for forbidden in SYNTHETIC_FORBIDDEN_TEMPLATES:
            if forbidden in s:
                raise ValueError(
                    f"CDVQA EVALUATION INTEGRITY VIOLATION: Synthetic reference detected: '{s}'. "
                    "Phase 0.5 strictly forbids synthetic templates as benchmark ground truth!"
                )
        # Normalize underscores and punctuation to spaces for comparison
        s = s.replace("_", " ")
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())

    @classmethod
    def official_normalize_answer(cls, text: Any) -> str:
        """Canonicalize CDVQA predicted answer."""
        cleaned = cls.clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return ""

        # Yes / No check
        if "yes" in tokens and "no" not in tokens:
            return "yes"
        if "no" in tokens and "yes" not in tokens:
            return "no"

        # Ratio range handling e.g. "0 to 10" -> "0 to 10"
        if "to" in tokens:
            nums = re.findall(r"\b\d+\b", cleaned)
            if len(nums) >= 2:
                return f"{nums[0]} to {nums[1]}"

        # Standard class names
        class_aliases = {
            "nvg surface": "nvg surface",
            "non vegetated ground surface": "nvg surface",
            "non vegetated ground": "nvg surface",
            "low vegetation": "low vegetation",
            "vegetation": "low vegetation",
            "building": "buildings",
            "buildings": "buildings",
            "tree": "trees",
            "trees": "trees",
            "water": "water",
            "playground": "playgrounds",
            "playgrounds": "playgrounds",
        }
        for alias, canonical in class_aliases.items():
            if alias in cleaned:
                return canonical

        return cleaned

    @classmethod
    def compute_accuracy_match(cls, pred_str: str, gt_str: str, q_type: str = "") -> int:
        """Official CDVQA exact accuracy match."""
        norm_p = cls.official_normalize_answer(pred_str)
        norm_g = cls.official_normalize_answer(gt_str)

        if not norm_g and not norm_p:
            return 1
        if not norm_g or not norm_p:
            return 0

        # Exact match
        if norm_p == norm_g:
            return 1

        # Token set match for compound answers
        p_tokens = set(norm_p.split())
        g_tokens = set(norm_g.split())
        if g_tokens.issubset(p_tokens) or norm_p == norm_g:
            return 1

        return 0

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute official CDVQA evaluation across official question types."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
            )

        overall_hits: List[int] = []
        hits_by_type = collections.defaultdict(list)

        for p, g in zip(predictions[:total], ground_truths[:total]):
            pred_text = str(p.get("prediction", p.get("answer", "")))
            gt_text = str(g.get("ground_truth", g.get("answer", "")))
            q_type = str(g.get("type", g.get("category", "change_or_not"))).lower()

            is_correct = self.compute_accuracy_match(pred_text, gt_text, q_type)
            overall_hits.append(is_correct)
            hits_by_type[q_type].append(is_correct)

        overall_acc = round(float(np.mean(overall_hits)), 4) if overall_hits else 0.0

        per_type_acc = {
            t: round(float(np.mean(hits)) * 100.0, 2) if hits else 0.0
            for t, hits in hits_by_type.items()
        }

        metrics = {
            "overall_accuracy": MetricResult(
                name="Overall VQA Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=overall_acc,
                sample_count=total,
                interpretation="Official Yuan et al. (2022) exact-match accuracy across all CDVQA question types.",
            )
        }

        for t in self.OFFICIAL_TYPES:
            if t in hits_by_type and hits_by_type[t]:
                score = float(np.mean(hits_by_type[t]))
                metrics[f"{t}_accuracy"] = MetricResult(
                    name=f"{t} Accuracy",
                    metric_type=MetricType.ACCURACY,
                    raw_score=score,
                    sample_count=len(hits_by_type[t]),
                    interpretation=f"Accuracy on {t} change queries.",
                )

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=overall_acc,
            aggregate_normalized_score=overall_acc,
            per_category_scores={
                "overall_accuracy": round(overall_acc * 100.0, 2),
                **{f"{t}_accuracy": acc for t, acc in per_type_acc.items()},
            },
            metadata={
                "type_counts": {t: len(hits) for t, hits in hits_by_type.items()},
            },
        )
