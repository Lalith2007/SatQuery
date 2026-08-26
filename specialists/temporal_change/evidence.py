"""Evidence generation for Division 3: Bi-Temporal Change Intelligence.

Creates standardized Evidence and Artifact objects using the existing
core.schemas definitions. Does not create duplicate evidence types.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from core.logging import get_logger
from core.schemas import Artifact, Evidence, EvidenceType

from specialists.temporal_change.interfaces import ChangeDetectionOutput, ChangedRegion
from specialists.temporal_change.postprocessing import PostprocessingResult

logger = get_logger("temporal_change.evidence")


def save_change_map_artifact(
    binary_map: np.ndarray,
    output_dir: Path,
    request_id: str,
    prob_map: Optional[np.ndarray] = None,
) -> List[Artifact]:
    """Save change map visualizations to disk and return Artifact references.

    Generates:
      - Binary change mask (black/white)
      - Color-coded change overlay (if probability map available)
    """
    artifacts: List[Artifact] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Binary mask
    binary_vis = (binary_map * 255).astype(np.uint8)
    binary_path = output_dir / f"{request_id}_change_mask.png"
    Image.fromarray(binary_vis).save(str(binary_path))
    mask_artifact = Artifact(
        name="binary_change_mask.png",
        type="change_map",
        uri_or_path=str(binary_path),
        description="Binary change detection mask (white=changed, black=unchanged).",
        mime_type="image/png",
    )
    artifacts.append(mask_artifact)

    # Color-coded probability overlay
    if prob_map is not None:
        color_map = _probability_to_colormap(prob_map)
        color_path = output_dir / f"{request_id}_change_heatmap.png"
        Image.fromarray(color_map).save(str(color_path))
        hm_artifact = Artifact(
            name="change_probability_heatmap.png",
            type="change_map",
            uri_or_path=str(color_path),
            description="Change probability heatmap (blue=low, red=high probability).",
            mime_type="image/png",
        )
        artifacts.append(hm_artifact)

    try:
        from presentation.evidence_renderer import ArtifactRegistry
        for art in artifacts:
            ArtifactRegistry.register(art.artifact_id, art.uri_or_path, name=art.name)
    except Exception:
        pass

    return artifacts


def _probability_to_colormap(prob_map: np.ndarray) -> np.ndarray:
    """Convert a probability map [0, 1] to a blue-to-red colormap (H, W, 3) uint8."""
    h, w = prob_map.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Blue -> Yellow -> Red gradient
    rgb[:, :, 0] = np.clip(prob_map * 510, 0, 255).astype(np.uint8)  # Red channel
    rgb[:, :, 1] = np.clip((1 - np.abs(prob_map - 0.5) * 2) * 255, 0, 255).astype(np.uint8)  # Green
    rgb[:, :, 2] = np.clip((1 - prob_map) * 510, 0, 255).astype(np.uint8)  # Blue channel

    return rgb


def generate_evidence(
    change_output: ChangeDetectionOutput,
    postproc_result: PostprocessingResult,
    t0_image_id: str,
    t1_image_id: str,
    output_dir: Path,
    request_id: str,
) -> tuple[List[Evidence], List[Artifact]]:
    """Generate standardized evidence and artifacts from change detection results.

    Uses existing EvidenceType definitions from core.schemas.

    Returns:
        Tuple of (evidence_list, artifact_list).
    """
    evidence: List[Evidence] = []
    artifacts: List[Artifact] = []

    # 1. CHANGE_MAP evidence
    change_map_data: Dict[str, Any] = {
        "t0_image_id": t0_image_id,
        "t1_image_id": t1_image_id,
        "changed_pixel_ratio": round(postproc_result.changed_pixel_ratio, 4),
        "changed_pixel_count": postproc_result.changed_pixel_count,
        "total_pixel_count": postproc_result.total_pixel_count,
        "num_regions": len(postproc_result.regions),
    }

    evidence.append(Evidence(
        type=EvidenceType.CHANGE_MAP,
        label="Bi-Temporal Change Detection Map",
        confidence=change_output.change_confidence,
        data=change_map_data,
        metadata={
            "model": change_output.model_name,
            "parameters": postproc_result.parameters,
        },
    ))

    # 2. BOUNDING_BOX evidence for top regions
    for region in postproc_result.regions[:10]:
        y_min, x_min, y_max, x_max = region.bbox
        h = postproc_result.binary_map.shape[0]
        w = postproc_result.binary_map.shape[1]

        # Normalize to [0, 1] image coordinates
        norm_bbox = [
            round(y_min / h, 4),
            round(x_min / w, 4),
            round(y_max / h, 4),
            round(x_max / w, 4),
        ]

        evidence.append(Evidence(
            type=EvidenceType.BOUNDING_BOX,
            label=f"Changed Region {region.region_id}",
            confidence=round(region.mean_confidence, 3) if region.mean_confidence else None,
            data={
                "bbox": norm_bbox,
                "format": "[ymin, xmin, ymax, xmax]",
                "coordinate_system": "normalized_image_coordinates (0.0 - 1.0)",
                "area_pixels": region.area_pixels,
                "region_id": region.region_id,
            },
            image_id=t1_image_id,
        ))

    # 3. TEXT_EVIDENCE with quantitative statistics
    pct = round(postproc_result.changed_pixel_ratio * 100, 2)
    evidence.append(Evidence(
        type=EvidenceType.TEXT_EVIDENCE,
        label="Quantitative Change Statistics",
        confidence=change_output.change_confidence,
        data={
            "changed_percentage": pct,
            "num_regions": len(postproc_result.regions),
            "largest_region_pixels": (
                postproc_result.regions[0].area_pixels if postproc_result.regions else 0
            ),
            "model": change_output.model_name,
            "inference_time_ms": change_output.inference_time_ms,
        },
    ))

    # 4. Save map artifacts to disk
    try:
        saved = save_change_map_artifact(
            binary_map=postproc_result.binary_map,
            output_dir=output_dir,
            request_id=request_id,
            prob_map=change_output.change_probability_map,
        )
        artifacts.extend(saved)
    except Exception as e:
        logger.warning(f"Artifact generation failed: {e}")

    return evidence, artifacts
