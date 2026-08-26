"""CDVQA (Change Detection Visual Question Answering) benchmark evaluation suite."""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Dict, List, Optional
import numpy as np

from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult, MetricResult, MetricType
from evaluation.normalizer import NormalizationStrategy, ScoreNormalizer


class CDVQAEvaluator(BaseBenchmarkEvaluator):
    """Evaluator for Change Detection VQA and bi-temporal change localization benchmarks."""

    def __init__(self):
        super().__init__(name="CDVQA", version="1.0.0")

    @staticmethod
    def clean_tokens(text: str) -> List[str]:
        """Tokenize normalized string."""
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s]", "", str(text).lower().strip())
        return cleaned.split()

    @classmethod
    def compute_bleu_n(cls, candidate: str, reference: str, n: int = 1) -> float:
        """Compute modified n-gram precision with brevity penalty for BLEU-n."""
        cand_tokens = cls.clean_tokens(candidate)
        ref_tokens = cls.clean_tokens(reference)

        if len(cand_tokens) < n or len(ref_tokens) < n:
            return 0.0

        cand_ngrams = collections.Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])
        ref_ngrams = collections.Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])

        clipped_count = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total_cand_ngrams = max(1, sum(cand_ngrams.values()))
        precision = clipped_count / total_cand_ngrams

        # Brevity penalty
        bp = 1.0 if len(cand_tokens) > len(ref_tokens) else math.exp(1 - (len(ref_tokens) / max(len(cand_tokens), 1)))
        return round(bp * precision, 4)

    @classmethod
    def compute_rouge_l(cls, candidate: str, reference: str) -> float:
        """Compute longest common subsequence (LCS) based ROUGE-L score."""
        cand_tokens = cls.clean_tokens(candidate)
        ref_tokens = cls.clean_tokens(reference)

        if not cand_tokens or not ref_tokens:
            return 0.0

        m, n = len(cand_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        prec = lcs / m
        rec = lcs / n
        if prec + rec == 0:
            return 0.0
        return round(2 * prec * rec / (prec + rec), 4)

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """Execute CDVQA benchmark evaluation across change existence, description, and localization."""
        total = min(len(predictions), len(ground_truths))
        if total == 0:
            return BenchmarkEvaluationResult(
                benchmark_name=self.name,
                total_samples=0,
                metrics={},
            )

        binary_hits: List[int] = []
        bleu1_scores: List[float] = []
        bleu4_scores: List[float] = []
        rouge_scores: List[float] = []

        for pred, gt in zip(predictions[:total], ground_truths[:total]):
            pred_text = str(pred.get("answer") or pred.get("description") or "")
            gt_text = str(gt.get("answer") or gt.get("description") or "")

            # Binary change classification accuracy (if question is existence-oriented)
            pred_clean = " ".join(self.clean_tokens(pred_text))
            gt_clean = " ".join(self.clean_tokens(gt_text))

            if gt_clean in {"yes", "no", "changed", "unchanged"}:
                is_correct = 1 if pred_clean == gt_clean or (gt_clean in pred_clean) else 0
                binary_hits.append(is_correct)

            # Lexical description quality
            b1 = self.compute_bleu_n(pred_text, gt_text, n=1)
            b4 = self.compute_bleu_n(pred_text, gt_text, n=min(4, len(self.clean_tokens(gt_text))))
            rl = self.compute_rouge_l(pred_text, gt_text)

            bleu1_scores.append(b1)
            bleu4_scores.append(b4)
            rouge_scores.append(rl)

        mean_b1 = round(float(np.mean(bleu1_scores)), 4) if bleu1_scores else 0.0
        mean_b4 = round(float(np.mean(bleu4_scores)), 4) if bleu4_scores else 0.0
        mean_rouge = round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0
        bin_acc = round(float(np.mean(binary_hits)), 4) if binary_hits else mean_b1

        metrics = {
            "binary_change_accuracy": MetricResult(
                name="Binary Change Existence Accuracy",
                metric_type=MetricType.ACCURACY,
                raw_score=bin_acc,
                sample_count=len(binary_hits) if binary_hits else total,
                interpretation="Accuracy in identifying presence or absence of temporal land-cover change.",
            ),
            "change_description_bleu1": MetricResult(
                name="Change Description BLEU-1",
                metric_type=MetricType.BLEU_1,
                raw_score=mean_b1,
                sample_count=total,
                interpretation="Unigram lexical precision for natural language change descriptions.",
            ),
            "change_description_bleu4": MetricResult(
                name="Change Description BLEU-4",
                metric_type=MetricType.BLEU_4,
                raw_score=mean_b4,
                sample_count=total,
                interpretation="4-gram sentence-level precision for detailed temporal change narratives.",
            ),
            "change_description_rouge_l": MetricResult(
                name="Change Description ROUGE-L",
                metric_type=MetricType.ROUGE_L,
                raw_score=mean_rouge,
                sample_count=total,
                interpretation="Longest common subsequence recall for change description structure.",
            ),
        }

        # Normalize metrics
        for m in metrics.values():
            ScoreNormalizer.normalize_metric(m, strategy=NormalizationStrategy.SCALE_100)

        agg_norm = ScoreNormalizer.compute_weighted_aggregate(
            metrics,
            weights={
                "binary_change_accuracy": 0.35,
                "change_description_bleu1": 0.25,
                "change_description_rouge_l": 0.40,
            },
        )

        return BenchmarkEvaluationResult(
            benchmark_name=self.name,
            total_samples=total,
            metrics=metrics,
            aggregate_raw_score=round(float(np.mean([m.raw_score for m in metrics.values()])), 4),
            aggregate_normalized_score=agg_norm,
            per_category_scores={
                "binary_accuracy": bin_acc,
                "bleu_1": mean_b1,
                "bleu_4": mean_b4,
                "rouge_l": mean_rouge,
            },
        )
