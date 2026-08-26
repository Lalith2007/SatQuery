"""VRSBench benchmark evaluation suite for SatQuery AI Division 5.

Implements rigorous evaluation of Remote Sensing VQA and Visual Grounding
according to official VRSBench dataset task protocols.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


class VRSBenchEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for VRSBench Visual Question Answering and Visual Grounding benchmarks."""

    def __init__(self):
        super().__init__(name="VRSBench", version="1.1.0")

    @staticmethod
    def calculate_box_iou(box_a: List[float], box_b: List[float]) -> float:
        """Calculate Intersection over Union (IoU) for bounding boxes in [ymin, xmin, ymax, xmax]."""
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0

        y_min_a, x_min_a, y_max_a, x_max_a = box_a
        y_min_b, x_min_b, y_max_b, x_max_b = box_b

        # Intersection coordinates
        inter_ymin = max(y_min_a, y_min_b)
        inter_xmin = max(x_min_a, x_min_b)
        inter_ymax = min(y_max_a, y_max_b)
        inter_xmax = min(x_max_a, x_max_b)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        # Union
        area_a = max(0.0, x_max_a - x_min_a) * max(0.0, y_max_a - y_min_a)
        area_b = max(0.0, x_max_b - x_min_b) * max(0.0, y_min_b - y_min_b if False else y_max_b - y_min_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0

        return round(float(inter_area / union_area), 4)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize answer text for token-level comparison."""
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @classmethod
    def compute_token_f1(cls, pred: str, gt: str) -> float:
        """Compute token-level precision, recall, and F1 score."""
        p_tokens = cls.clean_text(pred).split()
        g_tokens = cls.clean_text(gt).split()

        if not p_tokens and not g_tokens:
            return 1.0
        if not p_tokens or not g_tokens:
            return 0.0

        common = set(p_tokens).intersection(set(g_tokens))
        if not common:
            return 0.0

        prec = len(common) / len(p_tokens)
        rec = len(common) / len(g_tokens)
        f1 = 2 * (prec * rec) / (prec + rec)
        return round(f1, 4)

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute VRSBench VQA & Grounding evaluation."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
            )

        vqa_exact_matches = 0
        vqa_f1_scores = []
        category_hits: Dict[str, List[int]] = {"presence": [], "count": [], "landcover": []}

        ious: List[float] = []
        p50_hits = 0
        p75_hits = 0
        grounding_count = 0

        for pred, gt in zip(predictions[:total], ground_truths[:total]):
            # 1. VQA Evaluation
            pred_ans = str(pred.get("answer") or pred.get("prediction") or "")
            gt_ans = str(gt.get("answer") or gt.get("ground_truth") or "")
            category = str(gt.get("category") or "landcover").lower()

            is_exact = 1 if self.clean_text(pred_ans) == self.clean_text(gt_ans) else 0
            f1 = self.compute_token_f1(pred_ans, gt_ans)

            vqa_exact_matches += is_exact
            vqa_f1_scores.append(f1)

            cat_key = "presence" if "presence" in category or "exist" in category else "count" if "count" in category or "number" in category else "landcover"
            category_hits[cat_key].append(1 if f1 >= 0.5 or is_exact else 0)

            # 2. Visual Grounding Evaluation
            pred_box = pred.get("bbox") or pred.get("predicted_box")
            gt_box = gt.get("bbox") or gt.get("ground_truth_box")

            if pred_box and gt_box and len(pred_box) == 4 and len(gt_box) == 4:
                grounding_count += 1
                iou = self.calculate_box_iou(pred_box, gt_box)
                ious.append(iou)
                if iou >= 0.5:
                    p50_hits += 1
                if iou >= 0.75:
                    p75_hits += 1

        # Calculate VQA Metrics
        vqa_em_acc = round(vqa_exact_matches / total, 4)
        vqa_mean_f1 = round(float(np.mean(vqa_f1_scores)), 4) if vqa_f1_scores else 0.0

        # Calculate Grounding Metrics
        miou = round(float(np.mean(ious)), 4) if ious else 0.0
        p50 = round(p50_hits / max(grounding_count, 1), 4) if grounding_count > 0 else 0.0
        p75 = round(p75_hits / max(grounding_count, 1), 4) if grounding_count > 0 else 0.0

        # Category breakdowns
        cat_scores = {}
        for cat, hits in category_hits.items():
            cat_scores[f"{cat}_accuracy"] = round(float(np.mean(hits)), 4) if hits else 0.0

        # Construct MetricResults
        metrics = {
            "vqa_accuracy": MetricResult(
                name="VQA Exact Match Accuracy",
                metric_type=MetricType.EXACT_MATCH,
                raw_score=vqa_em_acc,
                sample_count=total,
                interpretation="Proportion of predicted answers matching reference answers exactly.",
            ),
            "vqa_token_f1": MetricResult(
                name="VQA Token F1 Score",
                metric_type=MetricType.TOKEN_F1,
                raw_score=vqa_mean_f1,
                sample_count=total,
                interpretation="Mean token-level harmonic mean precision & recall for open-ended VQA.",
            ),
            "grounding_miou": MetricResult(
                name="Visual Grounding mIoU",
                metric_type=MetricType.MIOU,
                raw_score=miou,
                sample_count=grounding_count,
                interpretation="Mean Intersection over Union across localized feature bounding boxes.",
            ),
            "grounding_p_at_05": MetricResult(
                name="Grounding Precision @ 0.5 IoU",
                metric_type=MetricType.PRECISION_AT_05,
                raw_score=p50,
                sample_count=grounding_count,
                interpretation="Percentage of predicted bounding boxes achieving IoU >= 0.50.",
            ),
            "grounding_p_at_75": MetricResult(
                name="Grounding Precision @ 0.75 IoU",
                metric_type=MetricType.PRECISION_AT_75,
                raw_score=p75,
                sample_count=grounding_count,
                interpretation="Percentage of predicted bounding boxes achieving strict IoU >= 0.75.",
            ),
        }

        # Normalize metrics to 0 - 100 scale
        for m in metrics.values():
            ScoreNormalizer.normalize_metric(m, strategy=NormalizationStrategy.SCALE_100)

        # Compute weighted aggregate score (40% VQA F1, 30% mIoU, 30% P@0.5)
        weights = {
            "vqa_token_f1": 0.40,
            "grounding_miou": 0.30,
            "grounding_p_at_05": 0.30,
        }
        agg_norm = ScoreNormalizer.compute_weighted_aggregate(metrics, weights=weights)

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=round(float(np.mean([m.raw_score for m in metrics.values()])), 4),
            aggregate_normalized_score=agg_norm,
            per_category_scores=cat_scores,
            metadata={
                "grounding_samples_evaluated": grounding_count,
                "vqa_samples_evaluated": total,
            },
        )
