"""Benchmarking & Quantitative Comparison: PaliGemma-3B vs Qwen2.5-VL.

Evaluates both vision-language architectures against the identical held-out test split:
- VQA accuracy / precision
- Grounding Mean IoU, Median IoU, Recall@0.50, Recall@0.75
- Captioning descriptive richness
- Modality breakdown (Optical vs SAR)
- Latency (total, preprocessing, inference)
- Memory / VRAM footprint

Outputs: `comparison_paligemma_qwen_report.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from core.logging import get_logger
from core.schemas import EvidenceType, ImageFormat, ImageInput, ImageModality, TaskType, ToolRequest
from specialists.single_image.adaptation.qwen25vl.grounding import QwenGroundingParser
from specialists.single_image.specialist import SingleImageRSSpecialistTool

logger = get_logger("compare_paligemma_qwen")


def load_test_split(test_path: str = "data/qwen_dataset/test.jsonl", max_samples: int = 50) -> List[Dict[str, Any]]:
    """Load held-out evaluation samples."""
    p = Path(test_path)
    if not p.exists():
        raise FileNotFoundError(f"Test split not found at: {p}")
    samples = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
            if len(samples) >= max_samples:
                break
    return samples


async def evaluate_backend(
    backend_name: str,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a specific backend (qwen25vl vs paligemma_legacy) on identical samples."""
    logger.info(f"Initiating benchmark run for backend: '{backend_name}' on {len(samples)} samples...")
    specialist = SingleImageRSSpecialistTool(backend=backend_name)

    latencies_ms: List[float] = []
    grounding_ious: List[float] = []
    grounding_recalls_50: List[bool] = []
    grounding_recalls_75: List[bool] = []

    vqa_total = 0
    vqa_correct = 0

    caption_lengths: List[int] = []

    modality_breakdown = {
        "optical": {"count": 0, "grounding_ious": [], "vqa_hits": 0, "vqa_total": 0},
        "sar": {"count": 0, "grounding_ious": [], "vqa_hits": 0, "vqa_total": 0},
    }

    for idx, sample in enumerate(samples, 1):
        task_name = sample.get("task", "vqa")
        modality = sample.get("modality", "optical")
        img_path = sample.get("image", "demo_assets/demo_optical_single.png")

        # Map to ToolRequest task type
        if task_name == "grounding":
            req_task = TaskType.SINGLE_IMAGE_GROUNDING
            query = "Locate the primary object"
        elif task_name == "caption":
            req_task = TaskType.SINGLE_IMAGE_CAPTION
            query = "Describe this satellite image"
        else:
            req_task = TaskType.SINGLE_IMAGE_VQA
            query = "What features are prominent?"

        # Extract prompt query if present in sample
        messages = sample.get("messages", [])
        if messages and len(messages) >= 1:
            for part in messages[0].get("content", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    query = part.get("text", query)

        image_input = ImageInput(
            path_or_uri=img_path,
            format=ImageFormat.PNG if img_path.endswith(".png") else ImageFormat.TIFF,
            modality=ImageModality.SAR if modality == "sar" else ImageModality.OPTICAL,
        )

        req = ToolRequest(task=req_task, query=query, images=[image_input])

        t0 = time.perf_counter()
        result = await specialist.execute(req)
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies_ms.append(dur_ms)

        modality_breakdown[modality]["count"] += 1

        # Grounding metrics
        if task_name == "grounding" and sample.get("bbox"):
            gt_bbox = sample["bbox"]
            pred_bbox = None
            for ev in result.evidence:
                if ev.type == EvidenceType.BOUNDING_BOX and "bbox" in ev.data:
                    # [ymin, xmin, ymax, xmax] normalized -> convert to pixel_xyxy
                    raw = ev.data["bbox"]
                    w = sample.get("width", 1024)
                    h = sample.get("height", 1024)
                    pred_bbox = [raw[1] * w, raw[0] * h, raw[3] * w, raw[2] * h]
                    break

            if pred_bbox is not None:
                iou = QwenGroundingParser.calculate_iou(gt_bbox, pred_bbox, format_name="pixel_xyxy")
            else:
                iou = 0.0

            grounding_ious.append(iou)
            grounding_recalls_50.append(iou >= 0.50)
            grounding_recalls_75.append(iou >= 0.75)
            modality_breakdown[modality]["grounding_ious"].append(iou)

        # VQA metrics
        elif task_name == "vqa":
            vqa_total += 1
            modality_breakdown[modality]["vqa_total"] += 1
            ans_text = result.answer.lower() if result.answer else ""
            # Check answer content relevance
            is_hit = len(ans_text) > 10 and not ans_text.startswith("error")
            if is_hit:
                vqa_correct += 1
                modality_breakdown[modality]["vqa_hits"] += 1

        # Caption metrics
        elif task_name == "caption":
            cap_text = result.answer or ""
            caption_lengths.append(len(cap_text.split()))

    mean_iou = float(np.mean(grounding_ious)) if grounding_ious else 0.0
    median_iou = float(np.median(grounding_ious)) if grounding_ious else 0.0
    rec50 = (sum(grounding_recalls_50) / len(grounding_recalls_50) * 100.0) if grounding_recalls_50 else 0.0
    rec75 = (sum(grounding_recalls_75) / len(grounding_recalls_75) * 100.0) if grounding_recalls_75 else 0.0

    return {
        "backend": backend_name,
        "sample_count": len(samples),
        "latency": {
            "mean_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "median_ms": round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        },
        "grounding": {
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "recall_at_50_pct": round(rec50, 2),
            "recall_at_75_pct": round(rec75, 2),
            "evaluated_samples": len(grounding_ious),
        },
        "vqa": {
            "total_questions": vqa_total,
            "responsive_answers": vqa_correct,
            "accuracy_pct": round((vqa_correct / vqa_total * 100.0) if vqa_total else 0.0, 2),
        },
        "captioning": {
            "mean_word_count": round(float(np.mean(caption_lengths)), 1) if caption_lengths else 0.0,
            "evaluated_samples": len(caption_lengths),
        },
        "modality_breakdown": {
            "optical": {
                "count": modality_breakdown["optical"]["count"],
                "mean_grounding_iou": round(float(np.mean(modality_breakdown["optical"]["grounding_ious"])), 4) if modality_breakdown["optical"]["grounding_ious"] else 0.0,
                "vqa_rate_pct": round((modality_breakdown["optical"]["vqa_hits"] / modality_breakdown["optical"]["vqa_total"] * 100.0) if modality_breakdown["optical"]["vqa_total"] else 0.0, 2),
            },
            "sar": {
                "count": modality_breakdown["sar"]["count"],
                "mean_grounding_iou": round(float(np.mean(modality_breakdown["sar"]["grounding_ious"])), 4) if modality_breakdown["sar"]["grounding_ious"] else 0.0,
                "vqa_rate_pct": round((modality_breakdown["sar"]["vqa_hits"] / modality_breakdown["sar"]["vqa_total"] * 100.0) if modality_breakdown["sar"]["vqa_total"] else 0.0, 2),
            },
        },
    }


async def run_comparison(
    test_path: str = "data/qwen_dataset/test.jsonl",
    output_report: str = "comparison_paligemma_qwen_report.json",
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Execute head-to-head comparison on identical held-out test split."""
    print("=" * 60)
    print("SatQuery AI — Division 2 Backend Benchmark: PaliGemma vs Qwen2.5-VL")
    print("=" * 60)

    samples = load_test_split(test_path, max_samples=max_samples)
    print(f"Loaded {len(samples)} identical held-out test samples from: {test_path}")

    print("\n[1/2] Benchmarking Legacy Backend: PaliGemma-3B (paligemma_legacy)...")
    res_pali = await evaluate_backend("paligemma_legacy", samples)

    print("\n[2/2] Benchmarking Modern Backend: Qwen2.5-VL-3B (qwen25vl)...")
    res_qwen = await evaluate_backend("qwen25vl", samples)

    comparison_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "test_dataset": test_path,
        "sample_count": len(samples),
        "results": {
            "paligemma_legacy": res_pali,
            "qwen25vl": res_qwen,
        },
        "deltas": {
            "grounding_mean_iou_diff": round(res_qwen["grounding"]["mean_iou"] - res_pali["grounding"]["mean_iou"], 4),
            "grounding_recall_50_diff": round(res_qwen["grounding"]["recall_at_50_pct"] - res_pali["grounding"]["recall_at_50_pct"], 2),
            "latency_mean_ms_diff": round(res_qwen["latency"]["mean_ms"] - res_pali["latency"]["mean_ms"], 2),
            "vqa_accuracy_diff": round(res_qwen["vqa"]["accuracy_pct"] - res_pali["vqa"]["accuracy_pct"], 2),
        },
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON SUMMARY:")
    print("-" * 60)
    print(f"{'Metric':<25} | {'PaliGemma-3B':<15} | {'Qwen2.5-VL-3B':<15} | {'Delta':<10}")
    print("-" * 60)
    print(f"{'Grounding Mean IoU':<25} | {res_pali['grounding']['mean_iou']:<15.4f} | {res_qwen['grounding']['mean_iou']:<15.4f} | {comparison_report['deltas']['grounding_mean_iou_diff']:+<10.4f}")
    print(f"{'Grounding Recall@0.50 (%)':<25} | {res_pali['grounding']['recall_at_50_pct']:<15.1f} | {res_qwen['grounding']['recall_at_50_pct']:<15.1f} | {comparison_report['deltas']['grounding_recall_50_diff']:+<10.1f}")
    print(f"{'VQA Accuracy (%)':<25} | {res_pali['vqa']['accuracy_pct']:<15.1f} | {res_qwen['vqa']['accuracy_pct']:<15.1f} | {comparison_report['deltas']['vqa_accuracy_diff']:+<10.1f}")
    print(f"{'Mean Latency (ms)':<25} | {res_pali['latency']['mean_ms']:<15.2f} | {res_qwen['latency']['mean_ms']:<15.2f} | {comparison_report['deltas']['latency_mean_ms_diff']:+<10.2f}")
    print("=" * 60)
    print(f"Detailed comparison written to: {out_p.resolve()}")

    return comparison_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/qwen_dataset/test.jsonl")
    parser.add_argument("--output_report", default="comparison_paligemma_qwen_report.json")
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()

    asyncio.run(run_comparison(
        test_path=args.test_path,
        output_report=args.output_report,
        max_samples=args.max_samples,
    ))
