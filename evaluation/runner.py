"""Unified benchmark evaluation runner and CLI for SatQuery AI Division 5.

Enables reproducible execution of VRSBench, RSVQA, CDVQA, and ISRO/SAC evaluation suites.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from core.logging import get_logger, setup_logging
from evaluation.base import BaseBenchmarkEvaluator, BenchmarkEvaluationResult
from evaluation.benchmarks.cdvqa import CDVQAEvaluator
from evaluation.benchmarks.isro_sac import ISROSACGenericEvaluator
from evaluation.benchmarks.rsvqa import RSVQAEvaluator
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator

logger = get_logger("evaluation_runner")


class BenchmarkRunner:
    """Registry and orchestrator for benchmark evaluation executions."""

    EVALUATORS: Dict[str, BaseBenchmarkEvaluator] = {
        "vrsbench": VRSBenchEvaluator(),
        "rsvqa": RSVQAEvaluator(),
        "cdvqa": CDVQAEvaluator(),
        "isro_sac": ISROSACGenericEvaluator(),
    }

    @classmethod
    def list_supported_benchmarks(cls) -> List[str]:
        """Return list of supported benchmark identifiers."""
        return list(cls.EVALUATORS.keys())

    @classmethod
    def get_evaluator(cls, name: str) -> BaseBenchmarkEvaluator:
        """Lookup evaluator by name."""
        key = name.lower().strip()
        if key not in cls.EVALUATORS:
            raise ValueError(f"Unknown benchmark '{name}'. Supported: {list(cls.EVALUATORS.keys())}")
        return cls.EVALUATORS[key]

    @classmethod
    def load_json_data(cls, path: str) -> List[Dict[str, Any]]:
        """Safely load JSON or JSONL benchmark data."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Evaluation data file not found: '{path}'")

        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                return json.load(f)
            else:
                # Handle JSON-lines format
                items = []
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
                return items

    @classmethod
    def run_evaluation(
        cls,
        benchmark_name: str,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> BenchmarkEvaluationResult:
        """Run benchmark evaluation and optionally serialize results to disk."""
        evaluator = cls.get_evaluator(benchmark_name)
        result = evaluator.evaluate(predictions, ground_truths, config=config)

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            json_file = out_p / f"{benchmark_name}_evaluation_report.json"
            with open(json_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(result.model_dump(), indent=2, default=str))

            # Save Markdown scoreboard
            md_file = out_p / f"{benchmark_name}_scoreboard.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(cls.format_scoreboard_markdown(result))

            logger.info(f"Saved evaluation artifacts to '{output_dir}'.")

        return result

    @classmethod
    def format_scoreboard_markdown(cls, result: BenchmarkEvaluationResult) -> str:
        """Format an evaluation result into a clean markdown scoreboard."""
        lines = [
            f"# 📊 SatQuery Benchmark Scoreboard: {result.benchmark_name}",
            f"",
            f"**Evaluator Version**: `{result.evaluator_version}` | **Total Samples Evaluated**: `{result.total_samples}`  ",
            f"**Timestamp (UTC)**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"",
            f"### 🏆 Aggregate Performance",
            f"- **Normalized Combined Score**: **{result.aggregate_normalized_score:.1f} / 100.0**",
            f"- **Raw Mean Metric Score**: **{result.aggregate_raw_score:.4f}**",
            f"",
            f"---",
            f"",
            f"### 📈 Granular Metric Breakdown",
            f"",
            f"| Metric Name | Type | Raw Score | Normalized (0-100) | Samples | Interpretation |",
            f"| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for m in result.metrics.values():
            norm_str = f"**{m.normalized_score:.1f}**" if m.normalized_score is not None else "—"
            raw_str = f"{m.raw_score:.4f}" if isinstance(m.raw_score, float) else str(m.raw_score)
            lines.append(f"| **{m.name}** | `{m.metric_type.value}` | {raw_str} | {norm_str} | {m.sample_count} | {m.interpretation} |")

        if result.per_category_scores:
            lines.extend([
                f"",
                f"---",
                f"",
                f"### 🔍 Category / Sub-Task Breakdown",
                f"",
                f"| Category / Sensor Sub-Group | Score |",
                f"| :--- | :---: |",
            ])
            for cat, score in result.per_category_scores.items():
                lines.append(f"| `{cat}` | **{score}** |")

        return "\n".join(lines)


def main():
    """CLI entrypoint for running benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="SatQuery AI Division 5 — Unified Benchmark Evaluation Runner",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        choices=["vrsbench", "rsvqa", "cdvqa", "isro_sac"],
        help="Target benchmark evaluation suite",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        required=True,
        help="Path to JSON/JSONL model predictions file",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=str,
        required=True,
        help="Path to JSON/JSONL ground truth annotations file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="evaluation_output",
        help="Directory to save evaluation reports and markdown scoreboard",
    )

    args = parser.parse_args()
    setup_logging()

    try:
        preds = BenchmarkRunner.load_json_data(args.predictions)
        gts = BenchmarkRunner.load_json_data(args.ground_truth)
        logger.info(f"Loaded {len(preds)} predictions and {len(gts)} ground truth references.")

        result = BenchmarkRunner.run_evaluation(
            benchmark_name=args.benchmark,
            predictions=preds,
            ground_truths=gts,
            output_dir=args.output_dir,
        )

        md = BenchmarkRunner.format_scoreboard_markdown(result)
        print("\n" + md + "\n")
        logger.info(f"Evaluation complete. Aggregate Score: {result.aggregate_normalized_score:.1f}/100.0")

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
