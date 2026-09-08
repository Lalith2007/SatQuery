"""Generic evaluation pipeline for hidden ISRO/SAC benchmark datasets (Division 5).

Provides generic evaluation of Cartosat-2S (high-resolution optical) and RISAT (SAR)
test sets across VQA, Grounding, Change, and Cross-Modal tasks WITHOUT hardcoded answers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


class ISROSACGenericEvaluator(BaseBenchmarkEvaluator):
    """Generic evaluator for ISRO/SAC Cartosat-2S & RISAT SAR test benchmarks."""

    def __init__(self):
        super().__init__(name="ISRO_SAC_Evaluation", version="1.0.0")

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Generic evaluation accepting external reference outputs and predictions.
        
        Strict Non-Fabrication Rule:
        Evaluates purely against the supplied ground_truth items without pre-programmed assumptions.
        """
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
                metadata={"status": "No samples provided for evaluation"},
            )

        task_type = config.get("task_type", "auto") if config else "auto"
        sensor_type = config.get("sensor", "cartosat_risat_joint") if config else "cartosat_risat_joint"

        exact_matches = 0
        token_f1_scores = []
        ious = []
        p50_count = 0
        grounding_total = 0

        sensor_breakdowns: Dict[str, List[float]] = {"cartosat_2s": [], "risat_sar": [], "joint_fusion": []}

        for pred, gt in zip(predictions[:total], ground_truths[:total]):
            pred_ans = str(pred.get("answer") or pred.get("prediction") or "")
            gt_ans = str(gt.get("answer") or gt.get("ground_truth") or "")
            sensor = str(gt.get("sensor") or pred.get("sensor") or "joint_fusion").lower()

            # Clean for comparison
            p_clean = re.sub(r"[^\w\s]", "", pred_ans.lower().strip())
            g_clean = re.sub(r"[^\w\s]", "", gt_ans.lower().strip())

            is_exact = 1 if p_clean == g_clean else 0
            exact_matches += is_exact

            f1 = VRSBenchEvaluator.compute_token_f1(pred_ans, gt_ans)
            token_f1_scores.append(f1)

            # Record per-sensor metrics
            sensor_key = "cartosat_2s" if "cartosat" in sensor else "risat_sar" if "risat" in sensor or "sar" in sensor else "joint_fusion"
            sensor_breakdowns[sensor_key].append(f1)

            # Spatial bounding box / localization check
            pred_box = pred.get("bbox") or pred.get("coordinates")
            gt_box = gt.get("bbox") or gt.get("coordinates")
            if pred_box and gt_box and len(pred_box) == 4 and len(gt_box) == 4:
                grounding_total += 1
                iou = VRSBenchEvaluator.calculate_box_iou(pred_box, gt_box)
                ious.append(iou)
                if iou >= 0.5:
                    p50_count += 1

        vqa_acc = round(exact_matches / total, 4)
        vqa_f1 = round(float(np.mean(token_f1_scores)), 4) if token_f1_scores else 0.0
        miou = round(float(np.mean(ious)), 4) if ious else 0.0
        p50 = round(p50_count / max(grounding_total, 1), 4) if grounding_total > 0 else 0.0

        metrics = {
            "isro_sac_vqa_accuracy": MetricResult(
                name="ISRO/SAC VQA Accuracy",
                metric_type=MetricType.EXACT_MATCH,
                raw_score=vqa_acc,
                sample_count=total,
                interpretation="Exact match answer accuracy across ISRO Cartosat/RISAT test items.",
            ),
            "isro_sac_token_f1": MetricResult(
                name="ISRO/SAC Token Overlap F1",
                metric_type=MetricType.TOKEN_F1,
                raw_score=vqa_f1,
                sample_count=total,
                interpretation="Harmonic mean token overlap for open-ended intelligence questions.",
            ),
            "isro_sac_grounding_miou": MetricResult(
                name="ISRO/SAC Feature Grounding mIoU",
                metric_type=MetricType.MIOU,
                raw_score=miou,
                sample_count=grounding_total,
                interpretation="Spatial localization accuracy for Cartosat-2S optical & RISAT radar targets.",
            ),
            "isro_sac_grounding_p50": MetricResult(
                name="ISRO/SAC Grounding P@0.5",
                metric_type=MetricType.PRECISION_AT_05,
                raw_score=p50,
                sample_count=grounding_total,
                interpretation="Precision of target detections exceeding 0.50 IoU benchmark threshold.",
            ),
        }

        # Normalize metrics
        for m in metrics.values():
            ScoreNormalizer.normalize_metric(m, strategy=NormalizationStrategy.SCALE_100)

        weights = {"isro_sac_token_f1": 0.40, "isro_sac_grounding_miou": 0.30, "isro_sac_grounding_p50": 0.30}
        agg_norm = ScoreNormalizer.compute_weighted_aggregate(metrics, weights=weights)

        per_sensor_scores = {}
        for s_name, scores in sensor_breakdowns.items():
            if scores:
                per_sensor_scores[s_name] = round(float(np.mean(scores)) * 100.0, 2)

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=round(float(np.mean([m.raw_score for m in metrics.values()])), 4),
            aggregate_normalized_score=agg_norm,
            per_category_scores=per_sensor_scores,
            metadata={
                "sensor_type": sensor_type,
                "task_type": task_type,
                "grounding_samples_evaluated": grounding_total,
                "sensor_breakdown_counts": {k: len(v) for k, v in sensor_breakdowns.items()},
            },
        )
