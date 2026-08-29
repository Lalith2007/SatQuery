"""Spatial Evidence Generation Engine for Division 4 Optical-SAR Specialist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import label

from core.schemas import Artifact, Evidence, EvidenceType

logger = logging.getLogger(__name__)


class SpatialEvidenceEngine:
    """Generates segmentation masks, bounding boxes, overlay artifacts, and canonical Evidence objects."""

    CLASS_COLOR_MAP = {
        "built_up": (220, 38, 38, 160),    # Crimson #DC2626
        "water": (37, 99, 235, 160),       # Azure #2563EB
        "vegetation": (16, 185, 129, 160), # Emerald #10B981
        "background": (0, 0, 0, 0),        # Transparent
    }

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def derive_bounding_boxes(
        self,
        binary_mask: np.ndarray,
        class_name: str,
        prob_map: np.ndarray,
        min_pixel_area: int = 15,
        max_boxes: int = 10,
    ) -> List[Dict[str, Any]]:
        """Derive bounding boxes from connected components in binary mask.
        
        Returns list of dicts with bbox format [ymin, xmin, ymax, xmax] normalized to [0.0, 1.0].
        """
        h, w = binary_mask.shape
        labeled_mask, num_features = label(binary_mask)
        boxes: List[Dict[str, Any]] = []

        for feat_idx in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_mask == feat_idx)
            area = len(y_indices)
            if area < min_pixel_area:
                continue

            ymin, ymax = int(y_indices.min()), int(y_indices.max())
            xmin, xmax = int(x_indices.min()), int(x_indices.max())

            region_prob = float(prob_map[y_indices, x_indices].mean())

            norm_bbox = [
                round(ymin / h, 4),
                round(xmin / w, 4),
                round((ymax + 1) / h, 4),
                round((xmax + 1) / w, 4),
            ]

            boxes.append({
                "class_name": class_name,
                "bbox": norm_bbox,
                "confidence": round(region_prob, 4),
                "pixel_area": area,
            })

        # Sort by area descending and limit count
        boxes.sort(key=lambda b: b["pixel_area"], reverse=True)
        return boxes[:max_boxes]

    def generate_evidence_artifacts(
        self,
        probs: np.ndarray,            # Shape (4, H, W)
        optical_rgb: np.ndarray,      # Shape (3, H, W) normalized [0, 1]
        sar_intensity: np.ndarray,    # Shape (1 or 2, H, W) normalized [0, 1]
        active_intents: Dict[str, float],
        request_id: str,
    ) -> Tuple[List[Evidence], List[Artifact], Dict[str, Any]]:
        """Generate binary PNG masks, visual overlay PNGs, bounding boxes, and canonical Evidence items."""
        target_h, target_w = optical_rgb.shape[1], optical_rgb.shape[2]

        # Resample probability maps to original optical image dimensions if feature resolution differs
        if probs.shape[1:] != (target_h, target_w):
            probs_tensor = torch.from_numpy(probs).unsqueeze(0)  # (1, 4, H_feat, W_feat)
            probs_resampled = F.interpolate(
                probs_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            probs = probs_resampled

        c, h, w = probs.shape
        class_names = ["built_up", "water", "vegetation", "background"]
        arg_max_map = probs.argmax(axis=0)

        evidences: List[Evidence] = []
        artifacts: List[Artifact] = []
        summary_metrics: Dict[str, Any] = {}

        total_pixels = h * w
        all_bboxes: List[Dict[str, Any]] = []

        # Create base RGB canvas (blend Optical RGB and SAR grayscale)
        sar_gray = (sar_intensity[0] * 255.0).astype(np.uint8)
        sar_pil = Image.fromarray(sar_gray).convert("RGB")
        opt_rgb = (optical_rgb.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        opt_pil = Image.fromarray(opt_rgb).convert("RGB")

        blended_pil = Image.blend(opt_pil, sar_pil, alpha=0.35)
        overlay_pil = blended_pil.convert("RGBA")
        draw_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # Process each target class
        for idx, cls in enumerate(class_names):
            if cls == "background":
                continue

            intent_weight = active_intents.get(cls, 0.1)
            cls_mask = (arg_max_map == idx) & (probs[idx] >= 0.40)
            pixel_count = int(cls_mask.sum())
            area_ratio = round(pixel_count / total_pixels, 4)

            summary_metrics[f"{cls}_pixel_count"] = pixel_count
            summary_metrics[f"{cls}_area_ratio"] = area_ratio

            # Save binary PNG mask
            mask_filename = f"{request_id}_{cls}_mask.png"
            mask_path = self.output_dir / mask_filename
            mask_img = Image.fromarray((cls_mask.astype(np.uint8) * 255))
            mask_img.save(mask_path)

            artifacts.append(
                Artifact(
                    name=mask_filename,
                    type="segmentation_mask",
                    uri_or_path=str(mask_path),
                    description=f"Binary segmentation mask for class '{cls}' (Area: {area_ratio * 100:.1f}%)",
                    mime_type="image/png",
                )
            )

            evidences.append(
                Evidence(
                    type=EvidenceType.MASK,
                    label=f"Optical-SAR {cls.replace('_', ' ').title()} Mask",
                    confidence=round(float(probs[idx][cls_mask].mean()) if pixel_count > 0 else 0.5, 4),
                    data={"mask_path": str(mask_path), "area_ratio": area_ratio, "pixel_count": pixel_count},
                )
            )

            # Draw transparent color overlay
            color = self.CLASS_COLOR_MAP.get(cls, (255, 255, 255, 128))
            color_layer = Image.new("RGBA", (w, h), color)
            draw_canvas.paste(color_layer, (0, 0), mask_img)

            # Derive bounding boxes if target requested
            if intent_weight >= 0.5 and pixel_count > 0:
                bboxes = self.derive_bounding_boxes(cls_mask, cls, probs[idx])
                all_bboxes.extend(bboxes)

                for box in bboxes:
                    evidences.append(
                        Evidence(
                            type=EvidenceType.BOUNDING_BOX,
                            label=f"Detected {cls.replace('_', ' ').title()} Region",
                            confidence=box["confidence"],
                            data={"bbox": box["bbox"], "format": "[ymin, xmin, ymax, xmax]", "class_name": cls},
                        )
                    )

        # Composite color overlay onto base image
        final_overlay = Image.alpha_composite(overlay_pil, draw_canvas)

        # Draw bounding box rectangles onto overlay
        draw = ImageDraw.Draw(final_overlay)
        for box in all_bboxes:
            ymin, xmin, ymax, xmax = box["bbox"]
            y0, x0 = int(ymin * h), int(xmin * w)
            y1, x1 = int(ymax * h), int(xmax * w)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 230, 0, 255), width=2)

        overlay_filename = f"{request_id}_cross_modal_overlay.png"
        overlay_path = self.output_dir / overlay_filename
        final_overlay.convert("RGB").save(overlay_path)

        artifacts.append(
            Artifact(
                name=overlay_filename,
                type="cross_modal_overlay",
                uri_or_path=str(overlay_path),
                description="Visual false-color overlay combining Optical RGB, SAR intensity, and multi-class target masks",
                mime_type="image/png",
            )
        )

        evidences.append(
            Evidence(
                type=EvidenceType.HIGHLIGHTED_IMAGE,
                label="Fused Optical-SAR Spatial Overlay",
                confidence=0.92,
                data={"overlay_path": str(overlay_path), "bounding_boxes_count": len(all_bboxes)},
            )
        )

        return evidences, artifacts, summary_metrics
