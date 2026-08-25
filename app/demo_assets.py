"""Demo dataset generator for judge presentation scenarios.

Generates valid on-disk remote sensing rasters (TIFF, PNG) for Demos A, B, C, D, and E.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import tifffile


def ensure_demo_assets(target_dir: Path | str = "demo_assets") -> dict[str, str]:
    """Ensure sample remote-sensing rasters exist on disk for judging demos."""
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)

    assets = {
        "optical_single": str(path / "demo_optical_single.png"),
        "airport_grounding": str(path / "demo_airport_grounding.png"),
        "change_t0": str(path / "demo_change_t0.png"),
        "change_t1": str(path / "demo_change_t1.png"),
        "optical_cross": str(path / "demo_optical_cross.png"),
        "sar_cross": str(path / "demo_sar_cross.tif"),
    }

    # 1. Single Optical Image (Urban & Agricultural)
    if not Path(assets["optical_single"]).exists():
        img = Image.new("RGB", (256, 256), color=(40, 110, 45))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 100, 100], fill=(130, 135, 140))  # Buildings
        draw.line([0, 128, 256, 128], fill=(50, 50, 50), width=6)  # Main road
        draw.ellipse([140, 140, 240, 240], fill=(30, 80, 180))   # Reservoir
        img.save(assets["optical_single"])

    # 2. Airport / Runway Grounding Image
    if not Path(assets["airport_grounding"]).exists():
        img = Image.new("RGB", (256, 256), color=(55, 95, 50))
        draw = ImageDraw.Draw(img)
        # Runway strip
        draw.rectangle([100, 20, 156, 236], fill=(180, 180, 185))
        # Centerline markings
        for y in range(30, 230, 25):
            draw.rectangle([126, y, 130, y + 12], fill=(255, 255, 255))
        # Apron & aircraft points
        draw.rectangle([20, 60, 80, 180], fill=(140, 140, 145))
        for y in [80, 120, 160]:
            draw.ellipse([40, y, 60, y + 20], fill=(240, 240, 245))
        img.save(assets["airport_grounding"])

    # 3. Bi-Temporal T0 (2021)
    if not Path(assets["change_t0"]).exists():
        img = Image.new("RGB", (256, 256), color=(90, 130, 70))  # Bare agricultural soil
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 120, 120], fill=(110, 150, 80))
        img.save(assets["change_t0"])

    # 4. Bi-Temporal T1 (2023 - Developed)
    if not Path(assets["change_t1"]).exists():
        img = Image.new("RGB", (256, 256), color=(90, 130, 70))
        draw = ImageDraw.Draw(img)
        # New commercial complex & road
        draw.rectangle([40, 40, 120, 120], fill=(170, 175, 180))
        draw.rectangle([55, 55, 105, 105], fill=(70, 90, 120))
        draw.line([0, 80, 256, 80], fill=(40, 40, 40), width=5)
        img.save(assets["change_t1"])

    # 5. Cross-Modal Optical
    if not Path(assets["optical_cross"]).exists():
        img = Image.new("RGB", (256, 256), color=(60, 120, 60))
        draw = ImageDraw.Draw(img)
        # Thin cloud covering top right
        draw.ellipse([140, 20, 250, 130], fill=(220, 225, 230))
        img.save(assets["optical_cross"])

    # 6. Cross-Modal SAR Raster (GeoTIFF)
    if not Path(assets["sar_cross"]).exists():
        sar_arr = (np.random.rand(256, 256) * 0.2).astype(np.float32)
        # Metallic / dihedral scatterers (strong radar return through clouds)
        sar_arr[50:90, 170:210] = 0.95
        tifffile.imwrite(assets["sar_cross"], sar_arr)

    return assets
