"""Benchmark Evaluation Pipeline for TinyCD Bi-Temporal Change Detector on Large-Scale LEVIR-CD.

Features:
- CLI configurable arguments (dataset root, checkpoint path, split, mode full/dev, threshold)
- Evaluates ALL discovered test samples in FULL mode (or dev subset in DEV mode)
- Explicit label output: 'FULL OFFICIAL TEST EVALUATION' vs 'CONTROLLED TEST SUBSET'
- Precision, Recall, F1 Score, IoU (Jaccard Index), Overall Pixel Accuracy
- Exact Confusion Matrix (TP, FP, FN, TN)
- Benchmark image audit trail (test_manifest.json & raw_predictions.json)
- Side-by-side Engineering Baseline Comparison (MockChangeModel vs TinyCD)
- CUDA-synchronized Latency profiling (cold start, warm mean, median, min, max, std)
- Deterministic Qualitative Comparison Panels (T0 | T1 | Ground Truth | Prediction | Overlay).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from specialists.temporal_change.adaptation.models.tinycd import TinyCD
from specialists.temporal_change.adaptation.dataset_loader import (
    LEVIRCDDatasetLoader,
    TemporalChangeDataset,
)
from specialists.temporal_change.adaptation.train_detector import calculate_batch_metrics
from specialists.temporal_change.adaptation.smoke_test_detector import compute_file_sha256
from specialists.temporal_change.model_adapter import MockChangeModel


def evaluate_test_split_pipeline(
    dataset_root: Path | str,
    checkpoint_path: Path | str = "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth",
    output_dir: Path | str = "specialists/temporal_change/evaluation",
    split: str = "test",
    mode: str = "full",
    threshold: float = 0.5,
    device: str = "auto",
    num_workers: int = 2,
) -> Dict[str, Any]:
    """Evaluate trained TinyCD checkpoint on strictly held-out test split."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "qualitative_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = device

    # Discover test samples dynamically
    test_samples = LEVIRCDDatasetLoader.discover_split_samples(dataset_root, split=split, mode=mode)
    if not test_samples:
        raise FileNotFoundError(f"No {split} samples discovered in '{dataset_root}/{split}'.")

    eval_classification = "FULL OFFICIAL TEST EVALUATION" if mode.lower() == "full" else "CONTROLLED TEST SUBSET"

    print("=" * 60)
    print(f"SATQUERY AI — DIVISION 3: {eval_classification}")
    print(f"Dataset Root:        {dataset_root}")
    print(f"Split:               {split.upper()}")
    print(f"Discovered Samples:  {len(test_samples)}")
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Compute Device:      {dev}")
    print(f"Decision Threshold:  {threshold}")
    print("=" * 60)

    # 1. Load trained TinyCD model
    model = TinyCD(in_channels=3, base_features=32)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Trained checkpoint not found at '{checkpoint_path}'.")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    test_ds = TemporalChangeDataset(test_samples, is_training=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers if dev != "mps" else 0,
        pin_memory=True if dev == "cuda" else False,
    )

    sample_manifest: List[Dict[str, Any]] = []
    raw_predictions: List[Dict[str, Any]] = []
    latencies: List[float] = []

    total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0
    qualitative_candidates: List[Dict[str, Any]] = []

    # Cold start latency measurement
    t_cold_start = time.perf_counter()
    with torch.no_grad():
        dummy_0 = torch.zeros(1, 3, 256, 256, device=dev)
        dummy_1 = torch.zeros(1, 3, 256, 256, device=dev)
        _ = model(dummy_0, dummy_1)
        if dev == "cuda":
            torch.cuda.synchronize()
    cold_latency_ms = (time.perf_counter() - t_cold_start) * 1000.0

    print("\nExecuting Neural Change Detection Inference...")

    # Evaluation loop
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            t0 = batch["t0"].to(dev, non_blocking=True)
            t1 = batch["t1"].to(dev, non_blocking=True)
            mask = batch["mask"].to(dev, non_blocking=True)
            sample_id = batch["sample_id"][0]
            parent_id = batch["parent_id"][0]
            t0_p = batch["t0_path"][0]
            t1_p = batch["t1_path"][0]

            if dev == "cuda":
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()

            pred = model(t0, t1)

            if dev == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            latencies.append(infer_ms)

            m = calculate_batch_metrics(pred, mask, threshold=threshold)
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            total_tn += m["tn"]

            pred_np = pred.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            record = {
                "sample_id": sample_id,
                "parent_id": parent_id,
                "t0_path": t0_p,
                "t1_path": t1_p,
                "f1": round(m["f1"], 4),
                "iou": round(m["iou"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "latency_ms": round(infer_ms, 2),
            }
            sample_manifest.append(record)
            raw_predictions.append(record)

            qualitative_candidates.append({
                "sample_id": sample_id,
                "t0_path": t0_p,
                "t1_path": t1_p,
                "f1": m["f1"],
                "iou": m["iou"],
                "precision": m["precision"],
                "recall": m["recall"],
                "pred_map": (pred_np >= threshold).astype(np.uint8) * 255,
                "gt_map": (mask_np >= 0.5).astype(np.uint8) * 255,
            })

    # Global Metrics Calculation
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
    global_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0
    global_oa = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0

    # Latency Summary
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    latency_summary = {
        "device": dev,
        "cold_start_latency_ms": round(cold_latency_ms, 2),
        "warm_mean_latency_ms": round(float(np.mean(warm_latencies)), 2),
        "warm_median_latency_ms": round(float(np.median(warm_latencies)), 2),
        "min_latency_ms": round(float(np.min(warm_latencies)), 2),
        "max_latency_ms": round(float(np.max(warm_latencies)), 2),
        "std_latency_ms": round(float(np.std(warm_latencies)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(warm_latencies)), 2) if warm_latencies else 0.0,
    }

    # 2. Baseline Comparison: Evaluate MockChangeModel on EXACT same test samples
    mock_model = MockChangeModel()
    mock_model.initialize(device="cpu")
    mock_tp, mock_fp, mock_fn, mock_tn = 0.0, 0.0, 0.0, 0.0

    for s in test_samples:
        t0_arr = np.array(Image.open(s["t0_path"]).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        t1_arr = np.array(Image.open(s["t1_path"]).convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        gt_arr = (np.array(Image.open(s["mask_path"]).convert("L").resize((256, 256)), dtype=np.float32) > 128.0).astype(np.float32) if s.get("mask_path") and os.path.exists(s["mask_path"]) else np.zeros((256, 256), dtype=np.float32)

        mock_out = mock_model.detect_change(t0_arr, t1_arr)
        m_pred = (mock_out.change_probability_map >= threshold).astype(np.float32)

        mock_tp += float((m_pred * gt_arr).sum())
        mock_fp += float((m_pred * (1.0 - gt_arr)).sum())
        mock_fn += float(((1.0 - m_pred) * gt_arr).sum())
        mock_tn += float(((1.0 - m_pred) * (1.0 - gt_arr)).sum())

    mock_prec = mock_tp / (mock_tp + mock_fp) if (mock_tp + mock_fp) > 0 else 0.0
    mock_rec = mock_tp / (mock_tp + mock_fn) if (mock_tp + mock_fn) > 0 else 0.0
    mock_f1 = 2 * (mock_prec * mock_rec) / (mock_prec + mock_rec) if (mock_prec + mock_rec) > 0 else 0.0
    mock_iou = mock_tp / (mock_tp + mock_fp + mock_fn) if (mock_tp + mock_fp + mock_fn) > 0 else 0.0

    baseline_comparison = {
        "mock_change_model": {
            "model_type": "Deterministic Difference",
            "f1_score": round(mock_f1, 4),
            "iou_jaccard": round(mock_iou, 4),
            "precision": round(mock_prec, 4),
            "recall": round(mock_rec, 4),
        },
        "tinycd_trained": {
            "model_type": "Trained Neural Network (Siamese U-Net + MAMB)",
            "f1_score": round(global_f1, 4),
            "iou_jaccard": round(global_iou, 4),
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
        },
        "relative_f1_gain": f"+{round((global_f1 - mock_f1) * 100.0, 2)}%",
    }

    # 3. Generate Deterministic Qualitative Panels: [T0 | T1 | Ground Truth | Prediction | Overlay]
    if qualitative_candidates:
        qualitative_candidates.sort(key=lambda x: x["f1"], reverse=True)
        representative_panels = {
            "strong_detection": qualitative_candidates[0],
            "partial_detection": qualitative_candidates[len(qualitative_candidates) // 2],
            "low_f1_edge_case": qualitative_candidates[-1],
        }

        for category, item in representative_panels.items():
            t0_img = Image.open(item["t0_path"]).convert("RGB").resize((256, 256))
            t1_img = Image.open(item["t1_path"]).convert("RGB").resize((256, 256))
            gt_img = Image.fromarray(item["gt_map"]).convert("RGB")
            pred_img = Image.fromarray(item["pred_map"]).convert("RGB")

            # Create visual overlay: Red mask over T1 image
            overlay_img = t1_img.copy()
            pred_mask_l = Image.fromarray(item["pred_map"]).convert("L")
            red_layer = Image.new("RGB", (256, 256), color=(255, 0, 0))
            overlay_img = Image.composite(red_layer, overlay_img, pred_mask_l)
            overlay_img = Image.blend(t1_img, overlay_img, alpha=0.45)

            # Create 5-column comparison panel: [T0 | T1 | GT | Pred | Overlay]
            panel = Image.new("RGB", (256 * 5, 256))
            panel.paste(t0_img, (0, 0))
            panel.paste(t1_img, (256, 0))
            panel.paste(gt_img, (512, 0))
            panel.paste(pred_img, (768, 0))
            panel.paste(overlay_img, (1024, 0))
            panel.save(panels_dir / f"panel_{category}_{item['sample_id']}.png")

    benchmark_report = {
        "status": "FULL_OFFICIAL_BENCHMARK_EVALUATED" if mode.lower() == "full" else "CONTROLLED_BENCHMARK_SUBSET_EVALUATION",
        "evaluation_classification": eval_classification,
        "model_architecture": "TinyCD (Siamese U-Net + MAMB)",
        "split": split,
        "dataset_mode": mode.upper(),
        "total_test_samples_evaluated": len(test_samples),
        "decision_threshold": threshold,
        "metrics": {
            "f1_score": round(global_f1, 4),
            "iou_jaccard": round(global_iou, 4),
            "precision": round(global_precision, 4),
            "recall": round(global_recall, 4),
            "overall_accuracy": round(global_oa, 4),
        },
        "confusion_matrix": {
            "true_positives": int(total_tp),
            "false_positives": int(total_fp),
            "false_negatives": int(total_fn),
            "true_negatives": int(total_tn),
        },
        "baseline_comparison": baseline_comparison,
        "latency_profile": latency_summary,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": compute_file_sha256(checkpoint_path),
    }

    with open(out_dir / "benchmark_report.json", "w") as f:
        json.dump(benchmark_report, f, indent=2)

    with open(out_dir / "test_manifest.json", "w") as f:
        json.dump(sample_manifest, f, indent=2)

    with open(out_dir / "raw_predictions.json", "w") as f:
        json.dump(raw_predictions, f, indent=2)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print(f"Classification:        {eval_classification}")
    print(f"Evaluated Test Pairs:  {len(test_samples)}")
    print(f"Global F1 Score:       {global_f1:.4f} ({global_f1 * 100:.2f}%)")
    print(f"Intersection over Union: {global_iou:.4f} ({global_iou * 100:.2f}%)")
    print(f"Precision:             {global_precision:.4f}")
    print(f"Recall:                {global_recall:.4f}")
    print(f"Overall Accuracy:      {global_oa:.4f}")
    print(f"Warm Median Latency:   {latency_summary['warm_median_latency_ms']} ms ({latency_summary['throughput_fps']} FPS)")
    print(f"Report Saved:          {out_dir / 'benchmark_report.json'}")
    print("=" * 60)

    return benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TinyCD on LEVIR-CD Bi-Temporal Change Detection.")
    parser.add_argument("--dataset-root", type=str, default=os.getenv("SATQUERY_TC_DATASET_ROOT", "specialists/temporal_change/data/levir_cd"), help="Root path of LEVIR-CD dataset")
    parser.add_argument("--checkpoint", type=str, default="specialists/temporal_change/weights/ChangeDetector-TinyCD.pth", help="Path to trained checkpoint")
    parser.add_argument("--output-dir", type=str, default="specialists/temporal_change/evaluation", help="Directory to save evaluation reports")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--mode", type=str, default=os.getenv("SATQUERY_TC_DATA_MODE", "full"), choices=["full", "dev"], help="Dataset split mode: full or dev")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for change probability")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda, mps, cpu, or auto)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_test_split_pipeline(
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        mode=args.mode,
        threshold=args.threshold,
        device=args.device,
        num_workers=args.num_workers,
    )
