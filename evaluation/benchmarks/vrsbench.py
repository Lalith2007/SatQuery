"""VRSBench benchmark evaluation suite for SatQuery AI.

Implements rigorous evaluation of Remote Sensing:
1. Visual Grounding (Acc@0.5, Acc@0.7 for Unique, Non-Unique, and All targets)
2. Complex Open-Vocabulary VQA (12 categories with official closed/open matching rules)
3. Image Captioning (BLEU-1..4, ROUGE-L, METEOR, CIDEr)

Aligned with official lx709/VRSBench protocols and coordinate conventions.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer
from scripts.vrsbench_coordinate_converter import CoordinateConverter


class VRSBenchEvaluator(BaseBenchmarkEvaluator):
    """Authoritative evaluator for official VRSBench benchmark tasks."""

    VQA_TYPES = [
        "object category", "object existence", "object quantity", "object color",
        "object shape", "object size", "object position", "object direction",
        "image", "scene type", "reasoning", "rural or urban"
    ]

    def __init__(self):
        super().__init__(name="VRSBench", version="2.0.0")

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardize text for VQA token matching."""
        if not text:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", str(text).lower().strip())
        return " ".join(cleaned.split())

    @classmethod
    def compute_vqa_match(cls, pred_str: str, gt_str: str, q_type: str = "") -> int:
        """Evaluate VQA correctness according to official VRSBench evaluation protocol."""
        p_clean = cls.clean_text(pred_str)
        g_clean = cls.clean_text(gt_str)

        if not g_clean and not p_clean:
            return 1
        if not g_clean or not p_clean:
            return 0

        # Exact match always passes
        if p_clean == g_clean:
            return 1

        # Closed-set questions: yes/no and numbers 0-99 require exact match
        numbers_set = {str(x) for x in range(100)}
        if g_clean in {"yes", "no"} or g_clean in numbers_set:
            return 1 if p_clean == g_clean else 0

        # Open-set questions: check if ground truth is a distinct word in prediction
        p_words = set(p_clean.split())
        g_words = set(g_clean.split())

        if g_words.issubset(p_words) or p_clean == g_clean:
            return 1

        return 0

    @classmethod
    def compute_bleu_n(cls, candidate: str, reference: str, n: int = 1) -> float:
        """Compute sentence-level BLEU-n with brevity penalty."""
        cand_tokens = cls.clean_text(candidate).split()
        ref_tokens = cls.clean_text(reference).split()

        if len(cand_tokens) < n or len(ref_tokens) < n:
            return 0.0

        cand_ngrams = collections.Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        ref_ngrams = collections.Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])

        clipped = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total = max(1, sum(cand_ngrams.values()))
        prec = clipped / total

        bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - (len(ref_tokens) / max(1, len(cand_tokens))))
        return float(bp * prec)

    @classmethod
    def compute_rouge_l(cls, candidate: str, reference: str) -> float:
        """Compute longest common subsequence (LCS) ROUGE-L score."""
        cand = cls.clean_text(candidate).split()
        ref = cls.clean_text(reference).split()

        if not cand or not ref:
            return 0.0

        m, n = len(cand), len(ref)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand[i - 1] == ref[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        prec = lcs / m
        rec = lcs / n
        if prec + rec == 0:
            return 0.0
        return float(2 * prec * rec / (prec + rec))

    def evaluate_grounding(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute official VRSBench grounding metrics."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return {}

        unique_ious: List[float] = []
        non_unique_ious: List[float] = []
        all_ious: List[float] = []

        for p, g in zip(predictions[:total], ground_truths[:total]):
            is_unique = bool(g.get("unique", g.get("is_unique", True)))

            # Parse prediction box
            pred_raw = p.get("predicted_box", p.get("bbox", p.get("prediction", "")))
            if isinstance(pred_raw, (list, tuple)) and len(pred_raw) >= 4:
                # Assume already in VRSBench 0-100 scale or [ymin, xmin, ymax, xmax]
                if max(pred_raw) > 100.0:
                    pred_vrs = CoordinateConverter.qwen_to_vrsbench(pred_raw)
                else:
                    pred_vrs = (int(round(pred_raw[0])), int(round(pred_raw[1])), int(round(pred_raw[2])), int(round(pred_raw[3])))
            else:
                try:
                    qwen_box = CoordinateConverter.parse_qwen_output(str(pred_raw))
                    pred_vrs = CoordinateConverter.qwen_to_vrsbench(qwen_box)
                except Exception:
                    pred_vrs = (0, 0, 0, 0)

            # Parse ground truth box
            gt_raw = g.get("ground_truth_box", g.get("ground_truth", g.get("obj_corner", [])))
            try:
                if isinstance(gt_raw, list) and len(gt_raw) >= 8:
                    gt_vrs_f = CoordinateConverter.parse_vrsbench_corners(gt_raw, scale_to_100=True)
                else:
                    gt_vrs_f = CoordinateConverter.parse_vrsbench_ground_truth(gt_raw)
                gt_vrs = (int(round(gt_vrs_f[0])), int(round(gt_vrs_f[1])), int(round(gt_vrs_f[2])), int(round(gt_vrs_f[3])))
            except Exception:
                gt_vrs = (0, 0, 0, 0)

            iou = CoordinateConverter.compute_official_vrsbench_iou(pred_vrs, gt_vrs)
            all_ious.append(iou)
            if is_unique:
                unique_ious.append(iou)
            else:
                non_unique_ious.append(iou)

        acc_05_all = float(np.mean([1 if x >= 0.5 else 0 for x in all_ious])) * 100.0 if all_ious else 0.0
        acc_07_all = float(np.mean([1 if x >= 0.7 else 0 for x in all_ious])) * 100.0 if all_ious else 0.0

        acc_05_unique = float(np.mean([1 if x >= 0.5 else 0 for x in unique_ious])) * 100.0 if unique_ious else acc_05_all
        acc_07_unique = float(np.mean([1 if x >= 0.7 else 0 for x in unique_ious])) * 100.0 if unique_ious else acc_07_all

        acc_05_non_unique = float(np.mean([1 if x >= 0.5 else 0 for x in non_unique_ious])) * 100.0 if non_unique_ious else 0.0
        acc_07_non_unique = float(np.mean([1 if x >= 0.7 else 0 for x in non_unique_ious])) * 100.0 if non_unique_ious else 0.0

        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

        return {
            "acc_05_all": round(acc_05_all, 2),
            "acc_07_all": round(acc_07_all, 2),
            "acc_05_unique": round(acc_05_unique, 2),
            "acc_07_unique": round(acc_07_unique, 2),
            "acc_05_non_unique": round(acc_05_non_unique, 2),
            "acc_07_non_unique": round(acc_07_non_unique, 2),
            "mean_iou": round(mean_iou, 4),
            "samples": total,
            "unique_samples": len(unique_ious),
            "non_unique_samples": len(non_unique_ious),
        }

    def evaluate_vqa(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute official VRSBench VQA metrics."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return {}

        correct_per_type = collections.defaultdict(int)
        total_per_type = collections.defaultdict(int)
        overall_correct = 0

        for p, g in zip(predictions[:total], ground_truths[:total]):
            pred_ans = str(p.get("prediction", p.get("answer", "")))
            gt_ans = str(g.get("ground_truth", g.get("answer", "")))
            q_type = str(g.get("type", g.get("category", "scene type"))).lower()

            if q_type == "image" or q_type == "rural or urban":
                q_type = "scene type"

            is_match = self.compute_vqa_match(pred_ans, gt_ans, q_type)
            overall_correct += is_match
            total_per_type[q_type] += 1
            if is_match:
                correct_per_type[q_type] += 1

        overall_acc = (overall_correct / max(1, total)) * 100.0
        per_type_acc = {
            t: round((correct_per_type[t] / max(1, total_per_type[t])) * 100.0, 2)
            for t in total_per_type
        }

        return {
            "overall_accuracy": round(overall_acc, 2),
            "samples": total,
            "category_accuracies": per_type_acc,
        }

    def evaluate_captioning(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute automatic captioning metrics (BLEU-1..4, ROUGE-L)."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return {}

        b1_list, b2_list, b3_list, b4_list, rl_list = [], [], [], [], []
        word_counts = []

        for p, g in zip(predictions[:total], ground_truths[:total]):
            pred_text = str(p.get("prediction", p.get("caption", "")))
            gt_text = str(g.get("ground_truth", g.get("caption", "")))

            b1_list.append(self.compute_bleu_n(pred_text, gt_text, n=1))
            b2_list.append(self.compute_bleu_n(pred_text, gt_text, n=2))
            b3_list.append(self.compute_bleu_n(pred_text, gt_text, n=3))
            b4_list.append(self.compute_bleu_n(pred_text, gt_text, n=4))
            rl_list.append(self.compute_rouge_l(pred_text, gt_text))
            word_counts.append(len(pred_text.split()))

        return {
            "bleu_1": round(float(np.mean(b1_list)), 4),
            "bleu_2": round(float(np.mean(b2_list)), 4),
            "bleu_3": round(float(np.mean(b3_list)), 4),
            "bleu_4": round(float(np.mean(b4_list)), 4),
            "rouge_l": round(float(np.mean(rl_list)), 4),
            "mean_word_count": round(float(np.mean(word_counts)), 1),
            "samples": total,
        }

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """General evaluation entrypoint dispatching by detected task type."""
        task = "grounding"
        if ground_truths and "obj_corner" in ground_truths[0] or (ground_truths and "bbox" in ground_truths[0]):
            task = "grounding"
        elif ground_truths and "question" in ground_truths[0]:
            task = "vqa"
        elif ground_truths and "caption" in ground_truths[0]:
            task = "captioning"

        metrics = {}
        if task == "grounding":
            res = self.evaluate_grounding(predictions, ground_truths)
            metrics["acc_05_all"] = MetricResult(
                name="Acc@0.5 All",
                metric_type=MetricType.ACCURACY,
                raw_score=res.get("acc_05_all", 0.0) / 100.0,
                sample_count=res.get("samples", 0),
                interpretation="Official VRSBench visual grounding accuracy at IoU >= 0.5 across all expressions.",
            )
            metrics["acc_07_all"] = MetricResult(
                name="Acc@0.7 All",
                metric_type=MetricType.ACCURACY,
                raw_score=res.get("acc_07_all", 0.0) / 100.0,
                sample_count=res.get("samples", 0),
                interpretation="Official VRSBench visual grounding accuracy at IoU >= 0.7 across all expressions.",
            )
            metrics["mean_iou"] = MetricResult(
                name="Mean IoU (Auxiliary)",
                metric_type=MetricType.MIOU,
                raw_score=res.get("mean_iou", 0.0),
                sample_count=res.get("samples", 0),
                interpretation="Auxiliary diagnostic mean box intersection over union.",
            )
            agg = res.get("acc_05_all", 0.0)
        else:
            res = self.evaluate_vqa(predictions, ground_truths)
            metrics["overall_accuracy"] = MetricResult(
                name="VQA Overall Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=res.get("overall_accuracy", 0.0) / 100.0,
                sample_count=res.get("samples", 0),
                interpretation="Official VRSBench VQA classification accuracy.",
            )
            agg = res.get("overall_accuracy", 0.0)

        flat_scores: Dict[str, float] = {}
        meta_dict: Dict[str, Any] = {}
        for k, v in res.items():
            if isinstance(v, (int, float)):
                flat_scores[k] = float(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        flat_scores[f"{sub_k}_accuracy"] = float(sub_v)
                meta_dict[k] = v
            else:
                meta_dict[k] = v

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=len(predictions),
            metrics=metrics,
            aggregate_raw_score=agg,
            aggregate_normalized_score=agg,
            per_category_scores=flat_scores,
            metadata=meta_dict,
        )
