"""Standalone Inference CLI for SatQuery AI Division 2 Qwen2.5-VL.

Usage:
    python infer_qwen25vl.py \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --adapter specialists/single_image/weights/qwen25vl_lora \\
        --image demo_assets/demo_optical_single.png \\
        --question "Where is the primary runway?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from PIL import Image

from specialists.single_image.adaptation.qwen25vl.inference import QwenSingleImageEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery Division 2 Qwen2.5-VL Standalone Inference")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model Hugging Face ID")
    parser.add_argument("--adapter", default=None, help="Path to trained PEFT LoRA adapter directory")
    parser.add_argument("--image", required=True, help="Path to Optical or SAR satellite raster")
    parser.add_argument("--question", required=True, help="Natural language query or grounding prompt")
    parser.add_argument("--task", default="auto", choices=["auto", "vqa", "grounding", "caption"])
    parser.add_argument("--output_json", default=None, help="Optional path to export JSON results")
    args = parser.parse_args()

    img_p = Path(args.image)
    if not img_p.exists():
        print(f"ERROR: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("SatQuery AI — Qwen2.5-VL Remote-Sensing Inference")
    print("=" * 60)
    print(f"Base Model: {args.model}")
    print(f"Adapter:    {args.adapter or 'Base Model / Deterministic Engine'}")
    print(f"Image:      {args.image}")
    print(f"Query:      {args.question}")
    print(f"Task Mode:  {args.task}")
    print("-" * 60)

    engine = QwenSingleImageEngine(
        base_model_id=args.model,
        adapter_path=args.adapter,
    )

    result = engine.query(
        image=img_p,
        question=args.question,
        task=args.task,
        image_id=img_p.stem,
    )

    print(f"\nANSWER:")
    print(result["answer"])

    if result["evidence"]:
        print(f"\nGROUNDING DETECTIONS ({len(result['evidence'])} boxes):")
        for i, ev in enumerate(result["evidence"], 1):
            bbox_norm = ev.data.get("bbox")
            bbox_px = ev.data.get("canonical_bbox")
            print(f"  [{i}] Label: '{ev.label}'")
            print(f"      Normalized [ymin, xmin, ymax, xmax]: {bbox_norm}")
            print(f"      Pixel [x1, y1, x2, y2]:              {bbox_px}")

    print(f"\nSTATUS:")
    status = result["status"]
    print(f"  backend:     {status['backend']}")
    print(f"  model_name:  {status['model_name']}")
    print(f"  is_mock:     {status['is_mock']}")
    print(f"  is_fallback: {status['is_fallback']}")

    print(f"\nMETRICS:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Convert Evidence objects to dicts
        serializable_result = dict(result)
        serializable_result["evidence"] = [e.data for e in result["evidence"]]
        with open(out_p, "w") as f:
            json.dump(serializable_result, f, indent=2)
        print(f"\nResults saved to: {out_p.resolve()}")

    print("=" * 60)


if __name__ == "__main__":
    main()
