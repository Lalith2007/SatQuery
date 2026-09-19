"""Deterministic Changed-Region Extraction & Evidence Packaging for Division 3.

Transforms raw TinyCD inference predictions (binary change mask, probability heatmap,
connected components) into standardized ChangedRegionEvidence artifacts for downstream
VLM consumption, visual grounding, and verifiable operational auditing.

Non-Fabrication & Zero-Leakage Guarantee:
- Extracts regions strictly from TinyCD predicted activations.
- Ground truth labels are never accessed or referenced.
- Deterministic ranking: regions sorted by area descending, then region_id ascending.
- Applies documented spatial padding around crops with edge clamping.
- Explicitly documents fallback behavior when no change is detected.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from core.logging import get_logger
from core.schemas import ChangedRegionEvidence
from specialists.temporal_change.interfaces import ChangeDetectionOutput
from specialists.temporal_change.postprocessing import PostprocessingResult

logger = get_logger("temporal_change.region_extraction")


def extract_changed_region_evidence(
    t0_path: str,
    t1_path: str,
    change_output: ChangeDetectionOutput,
    postproc_result: PostprocessingResult,
    output_dir: Path,
    request_id: str,
    threshold: float = 0.5,
    padding_ratio: float = 0.15,
    min_padding_pixels: int = 16,
) -> ChangedRegionEvidence:
    """Extract deterministic changed-region crops and package ChangedRegionEvidence.

    Args:
        t0_path: Path to source pre-change acquisition image (T0).
        t1_path: Path to source post-change acquisition image (T1).
        change_output: Standardized ChangeDetectionOutput from TinyCD.
        postproc_result: Standardized PostprocessingResult containing detected regions.
        output_dir: Destination directory for visual crop artifacts.
        request_id: Correlated system request tracing identifier.
        threshold: Decision threshold applied to TinyCD probability map.
        padding_ratio: Relative margin added around bounding box (default: 15%).
        min_padding_pixels: Minimum pixel padding added around crop (default: 16px).

    Returns:
        Structured ChangedRegionEvidence artifact.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source images to inspect actual spatial dimensions
    with Image.open(t1_path) as img_t1_raw:
        w_orig, h_orig = img_t1_raw.size
        img_t1 = img_t1_raw.convert("RGB")

    with Image.open(t0_path) as img_t0_raw:
        img_t0 = img_t0_raw.convert("RGB")

    # Mask spatial dimensions from postprocessing
    mask_h, mask_w = postproc_result.binary_map.shape[:2]

    # Save predicted binary mask and heatmap artifacts if not already present
    mask_filename = f"{request_id}_change_mask.png"
    mask_path = output_dir / mask_filename
    if not mask_path.exists():
        mask_vis = Image.fromarray((postproc_result.binary_map * 255).astype(np.uint8))
        mask_vis.save(mask_path, format="PNG")

    heatmap_path_str: Optional[str] = None
    if change_output.change_probability_map is not None:
        heatmap_filename = f"{request_id}_change_heatmap.png"
        heatmap_file = output_dir / heatmap_filename
        if not heatmap_file.exists():
            prob = np.clip(change_output.change_probability_map, 0.0, 1.0)
            # Create RGB false-color heatmap (blue to red)
            h_rgb = np.zeros((prob.shape[0], prob.shape[1], 3), dtype=np.uint8)
            h_rgb[..., 0] = (prob * 255).astype(np.uint8)  # Red = high probability
            h_rgb[..., 1] = ((1.0 - np.abs(prob - 0.5) * 2.0) * 255).astype(np.uint8)  # Green = mid
            h_rgb[..., 2] = ((1.0 - prob) * 255).astype(np.uint8)  # Blue = low
            Image.fromarray(h_rgb).save(heatmap_file, format="PNG")
        heatmap_path_str = str(heatmap_file)

    # 2. Deterministic sorting of candidate regions
    # Rule: Sort by area_pixels descending, then region_id ascending
    raw_regions = list(postproc_result.regions)
    sorted_regions = sorted(raw_regions, key=lambda r: (-r.area_pixels, r.region_id))

    total_pixels = h_orig * w_orig
    candidate_records: List[Dict[str, Any]] = []

    for r in sorted_regions:
        y_min_m, x_min_m, y_max_m, x_max_m = r.bbox

        # Normalized coordinates [0.0, 1.0] relative to model mask grid
        norm_ymin = max(0.0, min(1.0, y_min_m / mask_h))
        norm_xmin = max(0.0, min(1.0, x_min_m / mask_w))
        norm_ymax = max(0.0, min(1.0, y_max_m / mask_h))
        norm_xmax = max(0.0, min(1.0, x_max_m / mask_w))

        # Map to original image pixel coordinates
        orig_ymin = int(round(norm_ymin * h_orig))
        orig_xmin = int(round(norm_xmin * w_orig))
        orig_ymax = int(round(norm_ymax * h_orig))
        orig_xmax = int(round(norm_xmax * w_orig))

        # Scale region area proportionally to original dimensions
        scale_factor = (h_orig * w_orig) / (mask_h * mask_w)
        scaled_area = int(round(r.area_pixels * scale_factor))
        area_frac = scaled_area / total_pixels if total_pixels > 0 else 0.0

        candidate_records.append({
            "region_id": r.region_id,
            "pixel_bbox": [orig_ymin, orig_xmin, orig_ymax, orig_xmax],
            "normalized_bbox": [round(norm_ymin, 4), round(norm_xmin, 4), round(norm_ymax, 4), round(norm_xmax, 4)],
            "area_pixels": scaled_area,
            "area_fraction": round(area_frac, 6),
            "crop_dimensions": [max(1, orig_ymax - orig_ymin), max(1, orig_xmax - orig_xmin)],
            "source_image_id": Path(t1_path).stem,
            "mean_confidence": round(r.mean_confidence, 4) if r.mean_confidence else None,
            "tinycd_threshold": threshold,
            "temporal_ordering": "T0->T1",
        })

    # 3. Handle Empty Change Mask Case
    if not candidate_records or postproc_result.changed_pixel_ratio < 0.0005:
        logger.info(f"No significant change regions detected by TinyCD for request [{request_id}].")

        # Full image fallback overlay
        overlay_filename = f"{request_id}_overlay_no_change.png"
        overlay_path = output_dir / overlay_filename
        if not overlay_path.exists():
            img_t1.save(overlay_path, format="PNG")

        return ChangedRegionEvidence(
            evidence_id=f"cre_{request_id[:8]}_none",
            source_t0_path=t0_path,
            source_t1_path=t1_path,
            predicted_mask_path=str(mask_path),
            predicted_heatmap_path=heatmap_path_str,
            selected_region_id=0,
            selected_region_pixel_bbox=[0, 0, h_orig, w_orig],
            selected_region_normalized_bbox=[0.0, 0.0, 1.0, 1.0],
            area_pixels=0,
            area_fraction=0.0,
            crop_dimensions=[h_orig, w_orig],
            cropped_t1_path=t1_path,  # Explicit fallback to full T1
            cropped_t0_path=t0_path,
            cropped_mask_path=str(mask_path),
            visualization_overlay_path=str(overlay_path),
            candidate_regions=[],
            tinycd_threshold=threshold,
            temporal_ordering="T0->T1",
            has_change=False,
            fallback_reason="NO_CHANGE_DETECTED_IN_TINYCD_MASK: Change ratio below threshold; full post-change image supplied as documented fallback.",
            metadata={
                "tinycd_changed_pixel_ratio": round(postproc_result.changed_pixel_ratio, 6),
                "total_candidate_regions": 0,
                "spatial_coordinates": [0, 0, h_orig, w_orig],
                "crop_dimensions": [h_orig, w_orig],
            },
        )

    # 4. Primary Region Selection (Top Region)
    primary = candidate_records[0]
    p_ymin, p_xmin, p_ymax, p_xmax = primary["pixel_bbox"]

    # 5. Apply Documented Margin / Padding with Boundary Clamping
    box_h = max(1, p_ymax - p_ymin)
    box_w = max(1, p_xmax - p_xmin)
    pad_h = max(min_padding_pixels, int(box_h * padding_ratio))
    pad_w = max(min_padding_pixels, int(box_w * padding_ratio))

    crop_ymin = max(0, p_ymin - pad_h)
    crop_xmin = max(0, p_xmin - pad_w)
    crop_ymax = min(h_orig, p_ymax + pad_h)
    crop_xmax = min(w_orig, p_xmax + pad_w)

    crop_h = crop_ymax - crop_ymin
    crop_w = crop_xmax - crop_xmin

    # 6. Extract and Save Crops (BEFORE: T0, AFTER: T1)
    crop_t1_filename = f"{request_id}_crop_t1_region_{primary['region_id']}.png"
    crop_t1_path = output_dir / crop_t1_filename
    cropped_t1 = img_t1.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    cropped_t1.save(crop_t1_path, format="PNG")

    crop_t0_filename = f"{request_id}_crop_t0_region_{primary['region_id']}.png"
    crop_t0_path = output_dir / crop_t0_filename
    cropped_t0 = img_t0.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    cropped_t0.save(crop_t0_path, format="PNG")

    # 7. Extract Cropped Change Mask for Identical Coordinates (WHERE CHANGE OCCURRED)
    mask_full = Image.fromarray((postproc_result.binary_map * 255).astype(np.uint8))
    if mask_full.size != (w_orig, h_orig):
        mask_full = mask_full.resize((w_orig, h_orig), resample=Image.Resampling.NEAREST)
    cropped_mask = mask_full.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
    crop_mask_filename = f"{request_id}_crop_mask_region_{primary['region_id']}.png"
    crop_mask_path = output_dir / crop_mask_filename
    cropped_mask.save(crop_mask_path, format="PNG")

    # 8. Generate Visual Overlay for the Cropped Region (WHERE CHANGE OCCURRED)
    # Overlay is rendered onto cropped T1 at identical spatial coordinates [crop_ymin, crop_xmin, crop_ymax, crop_xmax]
    overlay_filename = f"{request_id}_change_overlay_region_{primary['region_id']}.png"
    overlay_path = output_dir / overlay_filename

    crop_overlay_rgba = cropped_t1.copy().convert("RGBA")
    mask_arr = np.array(cropped_mask)
    if np.any(mask_arr > 127):
        highlight_layer = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
        highlight_layer[mask_arr > 127] = [255, 40, 40, 110]  # Translucent diagnostic red change highlight
        highlight_img = Image.fromarray(highlight_layer, mode="RGBA")
        crop_overlay_rgba = Image.alpha_composite(crop_overlay_rgba, highlight_img)

    draw_crop = ImageDraw.Draw(crop_overlay_rgba)
    loc_ymin = p_ymin - crop_ymin
    loc_xmin = p_xmin - crop_xmin
    loc_ymax = p_ymax - crop_ymin
    loc_xmax = p_xmax - crop_xmin
    draw_crop.rectangle([loc_xmin, loc_ymin, loc_xmax, loc_ymax], outline=(0, 255, 64, 255), width=3)

    final_crop_overlay = crop_overlay_rgba.convert("RGB")
    final_crop_overlay.save(overlay_path, format="PNG")

    # 9. Also persist full-scene overview overlay for comprehensive macroscopic audits
    full_overlay_filename = f"{request_id}_full_scene_overlay.png"
    full_overlay_path = output_dir / full_overlay_filename
    full_overlay_img = img_t1.copy()
    full_draw = ImageDraw.Draw(full_overlay_img)
    full_draw.rectangle([p_xmin, p_ymin, p_xmax, p_ymax], outline=(0, 255, 64), width=4)
    full_draw.rectangle([crop_xmin, crop_ymin, crop_xmax, crop_ymax], outline=(255, 215, 0), width=2)
    full_overlay_img.save(full_overlay_path, format="PNG")

    return ChangedRegionEvidence(
        evidence_id=f"cre_{request_id[:8]}_r{primary['region_id']}",
        source_t0_path=t0_path,
        source_t1_path=t1_path,
        predicted_mask_path=str(mask_path),
        predicted_heatmap_path=heatmap_path_str,
        selected_region_id=primary["region_id"],
        selected_region_pixel_bbox=[p_ymin, p_xmin, p_ymax, p_xmax],
        selected_region_normalized_bbox=primary["normalized_bbox"],
        area_pixels=primary["area_pixels"],
        area_fraction=primary["area_fraction"],
        crop_dimensions=[crop_h, crop_w],
        cropped_t1_path=str(crop_t1_path),
        cropped_t0_path=str(crop_t0_path),
        cropped_mask_path=str(crop_mask_path),
        visualization_overlay_path=str(overlay_path),
        candidate_regions=candidate_records,
        tinycd_threshold=threshold,
        temporal_ordering="T0->T1",
        has_change=True,
        fallback_reason=None,
        metadata={
            "padded_crop_pixel_bbox": [crop_ymin, crop_xmin, crop_ymax, crop_xmax],
            "padding_pixels_applied": [pad_h, pad_w],
            "total_candidate_regions": len(candidate_records),
            "original_image_size": [h_orig, w_orig],
            "model_mask_size": [mask_h, mask_w],
            "spatial_coordinates": [crop_ymin, crop_xmin, crop_ymax, crop_xmax],
            "crop_dimensions": [crop_h, crop_w],
            "full_scene_overlay_path": str(full_overlay_path),
        },
    )
