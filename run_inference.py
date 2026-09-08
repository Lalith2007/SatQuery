"""SatQuery AI — Division 4 Optical-SAR Cross-Modal Inference CLI.

Executes inference on paired optical and SAR rasters using the trained
CMAF checkpoint (specialists/optical_sar/checkpoints/cmaf_landcover_best.pth).

Usage:
    python run_inference.py
    python run_inference.py --opt <optical_path> --sar <sar_path> --query "Your custom query"
"""

import argparse
import asyncio
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schemas import ImageFormat, ImageInput, ImageModality, TaskType, ToolRequest
from specialists.optical_sar.service import OpticalSarSpecialist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SatQuery Division 4 Optical-SAR Cross-Modal Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="data/official_whu_opt_sar/test/optical/NH49E003024_y0000_x0000.png",
        help="Path to optical RGB raster file (.png, .tif, .jpg)",
    )
    parser.add_argument(
        "--sar",
        type=str,
        default="data/official_whu_opt_sar/test/sar/NH49E003024_y0000_x0000.png",
        help="Path to SAR VV/VH raster file (.png, .tif, .jpg)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Segment urban infrastructure, forest vegetation, and water bodies",
        help="Natural language query for FiLM modulation and land-cover analysis",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="specialists/optical_sar/checkpoints/cmaf_landcover_best.pth",
        help="Path to trained PyTorch checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("SATQUERY AI — DIVISION 4 OPTICAL-SAR INFERENCE RUNNER")
    print("=" * 75)

    opt_path = Path(args.opt)
    sar_path = Path(args.sar)
    ckpt_path = Path(args.checkpoint)

    # 1. Input Verification
    if not opt_path.exists():
        print(f"[ERROR] Optical raster not found at: {opt_path.resolve()}")
        sys.exit(1)
    if not sar_path.exists():
        print(f"[ERROR] SAR raster not found at: {sar_path.resolve()}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Trained checkpoint not found at: {ckpt_path.resolve()}")
        sys.exit(1)

    print(f"Optical Input Raster : {opt_path.name} ({opt_path.stat().st_size:,d} bytes)")
    print(f"SAR Input Raster     : {sar_path.name} ({sar_path.stat().st_size:,d} bytes)")
    print(f"Trained Checkpoint   : {ckpt_path.name} ({ckpt_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"User Query String    : \"{args.query}\"")
    print("-" * 75)

    # 2. Initialize Specialist
    print("Initializing Optical-SAR Specialist and loading weights into GPU...")
    t_init_start = time.time()
    specialist = OpticalSarSpecialist(checkpoint_path=str(ckpt_path))
    t_init = (time.time() - t_init_start) * 1000.0
    print(f"[OK] Model loaded in {t_init:.1f} ms.")

    # 3. Formulate Canonical ToolRequest
    opt_suffix = opt_path.suffix.lower()
    img_format = ImageFormat.PNG if opt_suffix in [".png"] else ImageFormat.TIFF

    request = ToolRequest(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query=args.query,
        images=[
            ImageInput(
                path_or_uri=str(opt_path),
                modality=ImageModality.OPTICAL,
                format=img_format,
            ),
            ImageInput(
                path_or_uri=str(sar_path),
                modality=ImageModality.SAR,
                format=img_format,
            ),
        ],
    )

    # 4. Execute Async Forward Pass
    print("\nExecuting Dual ResNet-50 + CMAF + FiLM inference pipeline...")
    t_infer_start = time.time()
    result = asyncio.run(specialist.execute(request))
    latency_ms = (time.time() - t_infer_start) * 1000.0

    # 5. Display Structured Results
    print("\n" + "=" * 75)
    print("INFERENCE EXECUTION RESULTS")
    print("=" * 75)
    print(f"Execution Status     : {result.status.value.upper()}")
    print(f"Inference Latency    : {latency_ms:.2f} ms")
    print(f"Confidence Score     : {result.confidence:.4f}")
    print("-" * 75)
    print("ANALYSIS SUMMARY:")
    print(result.answer.strip())
    print("-" * 75)
    print(f"EVIDENCE ARTIFACTS GENERATED ({len(result.artifacts)} files):")
    for idx, art in enumerate(result.artifacts, 1):
        art_path = Path(art.uri_or_path)
        size_str = f"({art_path.stat().st_size:,d} bytes)" if art_path.exists() else ""
        print(f"  {idx}. {art.name:<32} | Type: {art.type:<22} | Path: {art.uri_or_path} {size_str}")
    print("=" * 75)


if __name__ == "__main__":
    main()
