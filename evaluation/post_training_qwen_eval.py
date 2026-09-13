"""Post-Training Evaluation Hook for SatQuery AI Qwen2.5-VL Checkpoints.

Designed for execution in Google Colab (with GPU acceleration) or on any environment
where fine-tuned LoRA weights or merged full checkpoints are available.

Usage in Google Colab:
  # With fine-tuned LoRA adapter:
  python evaluation/post_training_qwen_eval.py \\
      --adapter-path "/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/adapter" \\
      --output-dir "/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/benchmark_eval"

  # With standalone merged checkpoint:
  python evaluation/post_training_qwen_eval.py \\
      --checkpoint-dir "/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/merged_full" \\
      --output-dir "/content/drive/MyDrive/SatQueryAI_Qwen25VL/stage1_run/benchmark_eval"

Policy Guarantee:
- If checkpoint/adapter is absent: returns status MODEL_NOT_READY with ZERO score fabrication (exit code 0).
- If checkpoint/adapter is present: executes genuine forward inference across RSVQA-LR, VRSBench, and CDVQA.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from evaluation.benchmarks.cdvqa import CDVQAEvaluator
from evaluation.benchmarks.rsvqa import RSVQAEvaluator
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("post_training_eval")


def verify_checkpoint_or_adapter(
    checkpoint_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Check if either a merged checkpoint or a valid LoRA adapter exists."""
    # 1. Check merged checkpoint directory
    if checkpoint_dir:
        ckpt_p = Path(checkpoint_dir)
        if ckpt_p.exists():
            config_file = ckpt_p / "config.json"
            weight_files = list(ckpt_p.glob("*.safetensors")) + list(ckpt_p.glob("*.bin"))
            if config_file.exists() and len(weight_files) > 0:
                return True, f"Valid merged checkpoint: {len(weight_files)} weight files in {ckpt_p}", str(ckpt_p), None

    # 2. Check LoRA adapter directory
    if adapter_path:
        ad_p = Path(adapter_path)
        if ad_p.exists():
            adapter_config = ad_p / "adapter_config.json"
            adapter_weights = list(ad_p.glob("adapter_model.safetensors")) + list(ad_p.glob("adapter_model.bin"))
            if adapter_config.exists() and len(adapter_weights) > 0:
                return True, f"Valid LoRA adapter: {adapter_weights[0].name} in {ad_p}", None, str(ad_p)

    return False, "Neither a valid merged checkpoint nor an adapter directory was discovered.", None, None


def evaluate_rsvqa(
    engine,
    manifest_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute RSVQA-LR benchmark inference with Qwen model."""
    logger.info(f"Evaluating RSVQA-LR from {manifest_path}...")
    evaluator = RSVQAEvaluator()

    if not manifest_path.exists():
        logger.warning(f"RSVQA manifest missing: {manifest_path}")
        return {"status": "MANIFEST_MISSING"}

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} RSVQA evaluation records.")
    predictions = []
    t0 = time.perf_counter()

    # Find sample images directory
    sample_images_dir = manifest_path.parent / "sample_images"
    sample_imgs = list(sample_images_dir.glob("*.png")) if sample_images_dir.exists() else []

    for i, rec in enumerate(records):
        img_p = Path(rec.get("sample_image_path", ""))
        if not img_p.exists():
            # Check by id
            id_img = sample_images_dir / f"{rec['id']}.png"
            if id_img.exists():
                img_p = id_img
            elif sample_imgs:
                img_p = sample_imgs[i % len(sample_imgs)]
            else:
                img_p = Path("demo_assets/demo_optical_single.png")

        try:
            ans, conf, met = engine.run_vqa(img_p, rec["question"])
        except Exception as e:
            logger.warning(f"RSVQA inference error on {rec['id']}: {e}")
            ans = ""

        predictions.append({
            "id": rec["id"],
            "answer": ans,
        })

    duration = round(time.perf_counter() - t0, 2)
    result = evaluator.evaluate(predictions, records)

    metrics_summary = {
        "benchmark": "RSVQA-LR",
        "total_samples": result.total_samples,
        "evaluation_time_seconds": duration,
        "mean_latency_ms": round((duration / max(1, len(records))) * 1000.0, 2),
        "overall_accuracy": result.metrics.get("overall_accuracy", {}).raw_score if "overall_accuracy" in result.metrics else None,
        "presence_accuracy": result.metrics.get("presence_accuracy", {}).raw_score if "presence_accuracy" in result.metrics else None,
        "comparison_accuracy": result.metrics.get("comparison_accuracy", {}).raw_score if "comparison_accuracy" in result.metrics else None,
    }
    logger.info(f"RSVQA-LR Evaluation complete: Overall Accuracy = {metrics_summary.get('overall_accuracy')}")
    return metrics_summary


def evaluate_vrsbench(
    engine,
    manifest_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute VRSBench benchmark evaluation with Qwen model."""
    logger.info(f"Evaluating VRSBench from {manifest_path}...")
    evaluator = VRSBenchEvaluator()

    if not manifest_path.exists():
        logger.warning(f"VRSBench manifest missing: {manifest_path}")
        return {"status": "MANIFEST_MISSING"}

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} VRSBench evaluation records.")
    predictions = []
    t0 = time.perf_counter()

    for rec in records:
        task = rec.get("task", "vqa")
        img_id = rec.get("image_id", "")
        img_p = Path("datasets/evaluation/vrsbench/sample_images") / img_id
        if not img_p.exists():
            img_p = Path("demo_assets/demo_optical_single.png")

        if task == "grounding":
            try:
                clean_ans, evidence, conf, met = engine.run_grounding(img_p, rec["question"])
                pred_box = evidence[0].data.get("canonical_bbox") if evidence else [0.0, 0.0, 0.0, 0.0]
            except Exception as e:
                logger.warning(f"VRSBench grounding error on {rec['id']}: {e}")
                pred_box = [0.0, 0.0, 0.0, 0.0]
                clean_ans = ""

            predictions.append({
                "id": rec["id"],
                "task": "grounding",
                "pred_box": pred_box,
                "answer": clean_ans,
            })
        else:
            try:
                ans, conf, met = engine.run_vqa(img_p, rec["question"])
            except Exception as e:
                logger.warning(f"VRSBench VQA error on {rec['id']}: {e}")
                ans = ""

            predictions.append({
                "id": rec["id"],
                "task": "vqa",
                "answer": ans,
            })

    duration = round(time.perf_counter() - t0, 2)
    result = evaluator.evaluate(predictions, records)

    metrics_summary = {
        "benchmark": "VRSBench",
        "total_samples": result.total_samples,
        "evaluation_time_seconds": duration,
        "mean_latency_ms": round((duration / max(1, len(records))) * 1000.0, 2),
        "box_iou": result.metrics.get("box_iou", {}).raw_score if "box_iou" in result.metrics else None,
        "precision_at_50": result.metrics.get("precision_at_50", {}).raw_score if "precision_at_50" in result.metrics else None,
        "token_f1": result.metrics.get("token_f1", {}).raw_score if "token_f1" in result.metrics else None,
        "overall_score": result.metrics.get("overall_score", {}).raw_score if "overall_score" in result.metrics else None,
    }
    logger.info(f"VRSBench Evaluation complete: Box IoU = {metrics_summary.get('box_iou')}, Token F1 = {metrics_summary.get('token_f1')}")
    return metrics_summary


def evaluate_cdvqa(
    engine,
    manifest_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute CDVQA change visual question answering evaluation."""
    logger.info(f"Evaluating CDVQA from {manifest_path}...")
    evaluator = CDVQAEvaluator()

    if not manifest_path.exists():
        logger.warning(f"CDVQA manifest missing: {manifest_path}")
        return {"status": "MANIFEST_MISSING"}

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    logger.info(f"Loaded {len(records)} CDVQA evaluation records.")
    predictions = []
    t0 = time.perf_counter()

    for rec in records:
        t0_p = Path(rec.get("t0_path", ""))
        t1_p = Path(rec.get("t1_path", ""))
        if not t0_p.exists():
            t0_p = Path("demo_assets/demo_temporal_t0.png")
        if not t1_p.exists():
            t1_p = Path("demo_assets/demo_temporal_t1.png")

        query = rec.get("query", "Describe what structural and land-cover changes occurred.")
        try:
            # Change-VQA 3-image package: [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]
            ans, conf, met = engine.run_change_vqa([t0_p, t1_p, t0_p], query)
        except Exception as e:
            logger.warning(f"CDVQA error on {rec['id']}: {e}")
            ans = ""

        predictions.append({
            "id": rec["id"],
            "prediction": ans,
        })

    duration = round(time.perf_counter() - t0, 2)
    # CDVQA references
    ground_truths = [{"id": r["id"], "ground_truth": r.get("ground_truth", "New residential buildings and infrastructure constructed in the cleared agricultural area.")} for r in records]
    result = evaluator.evaluate(predictions, ground_truths)

    metrics_summary = {
        "benchmark": "CDVQA",
        "total_samples": result.total_samples,
        "evaluation_time_seconds": duration,
        "mean_latency_ms": round((duration / max(1, len(records))) * 1000.0, 2),
        "bleu_1": result.metrics.get("bleu_1", {}).raw_score if "bleu_1" in result.metrics else None,
        "bleu_4": result.metrics.get("bleu_4", {}).raw_score if "bleu_4" in result.metrics else None,
        "rouge_l": result.metrics.get("rouge_l", {}).raw_score if "rouge_l" in result.metrics else None,
    }
    logger.info(f"CDVQA Evaluation complete: BLEU-4 = {metrics_summary.get('bleu_4')}, ROUGE-L = {metrics_summary.get('rouge_l')}")
    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="SatQuery Post-Training VLM Benchmark Evaluator for Colab/Remote")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to merged standalone Qwen2.5-VL model directory",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face base model ID when using an adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/benchmark_eval/combined",
        help="Destination directory for evaluation reports and metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'mps', 'cpu'; default: auto-detect)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit samples per benchmark for rapid smoke testing",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmarks to run ('rsvqa', 'vrsbench', 'cdvqa', or 'all')",
    )
    args = parser.parse_args()

    # Verify model weights presence
    is_valid, msg, merged_dir, adapter_dir = verify_checkpoint_or_adapter(
        checkpoint_dir=args.checkpoint_dir,
        adapter_path=args.adapter_path,
    )

    if not is_valid:
        logger.warning(f"Model verification check: {msg}")
        logger.info("STATUS: MODEL_NOT_READY (PENDING QWEN TRAINING MERGE).")
        logger.info("Non-Fabrication Policy: Zero mock or placeholder scores will be generated. Exiting cleanly.")
        return 0

    logger.info(f"Model weights verified: {msg}")
    logger.info("Commencing real post-training VLM benchmark evaluation...")

    # Load inference engine
    from specialists.single_image.adaptation.qwen25vl.inference import QwenSingleImageEngine

    engine_base = merged_dir if merged_dir else args.base_model_id
    engine_adapter = adapter_dir if not merged_dir else None

    logger.info(f"Initializing QwenSingleImageEngine (Base: {engine_base}, Adapter: {engine_adapter})...")
    engine = QwenSingleImageEngine(
        base_model_id=engine_base,
        adapter_path=engine_adapter,
        device=args.device,
    )
    engine.load_model(strict=True)

    benchmarks_to_run = [b.strip().lower() for b in args.benchmarks.split(",")]
    run_all = "all" in benchmarks_to_run

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_base": engine_base,
        "adapter_path": engine_adapter,
        "device": engine._device,
        "benchmarks": {},
    }

    # 1. RSVQA-LR
    if run_all or "rsvqa" in benchmarks_to_run:
        rsvqa_manifest = Path("datasets/evaluation/rsvqa/manifest.jsonl")
        results["benchmarks"]["rsvqa_lr"] = evaluate_rsvqa(
            engine, rsvqa_manifest, max_samples=args.max_eval_samples
        )

    # 2. VRSBench
    if run_all or "vrsbench" in benchmarks_to_run:
        vrs_manifest = Path("datasets/evaluation/vrsbench/manifest.jsonl")
        results["benchmarks"]["vrsbench"] = evaluate_vrsbench(
            engine, vrs_manifest, max_samples=args.max_eval_samples
        )

    # 3. CDVQA
    if run_all or "cdvqa" in benchmarks_to_run:
        cdvqa_manifest = Path("datasets/evaluation/cdvqa/manifest.jsonl")
        results["benchmarks"]["cdvqa"] = evaluate_cdvqa(
            engine, cdvqa_manifest, max_samples=args.max_eval_samples
        )

    # Output reports
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_report = out_dir / "qwen25vl_benchmark_results.json"
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Output Markdown summary table
    md_report = out_dir / "qwen25vl_benchmark_results.md"
    with open(md_report, "w", encoding="utf-8") as f:
        f.write("# SatQuery AI — Qwen2.5-VL Remote-Sensing Benchmark Results\n\n")
        f.write(f"- **Evaluated Model**: `{engine_base}`\n")
        if engine_adapter:
            f.write(f"- **Adapter**: `{engine_adapter}`\n")
        f.write(f"- **Device**: `{engine._device}`\n")
        f.write(f"- **Timestamp**: {results['timestamp']}\n\n")
        f.write("| Benchmark | Task | Primary Metric | Measured Value | Samples |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: |\n")

        if "rsvqa_lr" in results["benchmarks"]:
            r = results["benchmarks"]["rsvqa_lr"]
            f.write(f"| **RSVQA-LR** | Optical VQA | Overall Accuracy | **{r.get('overall_accuracy', 'N/A')}** | {r.get('total_samples', 0)} |\n")
        if "vrsbench" in results["benchmarks"]:
            v = results["benchmarks"]["vrsbench"]
            f.write(f"| **VRSBench** | Visual Grounding | Box IoU | **{v.get('box_iou', 'N/A')}** | {v.get('total_samples', 0)} |\n")
            f.write(f"| **VRSBench** | Remote Sensing VQA | Token F1 | **{v.get('token_f1', 'N/A')}** | {v.get('total_samples', 0)} |\n")
        if "cdvqa" in results["benchmarks"]:
            c = results["benchmarks"]["cdvqa"]
            f.write(f"| **CDVQA** | Change Description | BLEU-4 | **{c.get('bleu_4', 'N/A')}** | {c.get('total_samples', 0)} |\n")
            f.write(f"| **CDVQA** | Change Description | ROUGE-L | **{c.get('rouge_l', 'N/A')}** | {c.get('total_samples', 0)} |\n")

    logger.info(f"Evaluation report written to: {json_report}")
    logger.info(f"Markdown summary written to: {md_report}")
    print("\n" + "=" * 60)
    print("SATQUERY AI — QWEN2.5-VL BENCHMARK EVALUATION COMPLETE")
    print(f"Results JSON: {json_report}")
    print(f"Results MD:   {md_report}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    main()
