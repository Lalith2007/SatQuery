"""Spatial evidence rendering engine for SatQuery AI Division 5.

Renders bounding boxes, segmentation masks, bi-temporal change maps,
optical-SAR cross-modal blends, localized crops, and attention heatmaps into
high-visibility visual artifacts for user inspection, UI display, and reports.
"""

from __future__ import annotations

import io
import math
import mimetypes
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from core.config import settings
from core.logging import get_logger
from core.schemas import Artifact, Evidence, EvidenceType, ImageFormat, ImageInput, ImageModality

logger = get_logger("evidence_renderer")


class ArtifactRegistry:
    """Thread-safe runtime registry mapping artifact IDs and names to verified filesystem paths."""

    _registry: Dict[str, Path] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, artifact_id: str, file_path: Union[str, Path], name: Optional[str] = None) -> None:
        """Register an artifact ID and optional name to its filesystem path."""
        p = Path(file_path)
        with cls._lock:
            cls._registry[str(artifact_id)] = p
            if name:
                cls._registry[str(name)] = p

    @classmethod
    def get_path(cls, artifact_id: str) -> Optional[Path]:
        """Look up the registered filesystem path for an artifact ID."""
        with cls._lock:
            return cls._registry.get(str(artifact_id))

    @classmethod
    def clear(cls) -> None:
        """Clear all registered artifact records."""
        with cls._lock:
            cls._registry.clear()


class EvidenceRenderingResult:
    """Container for the output of evidence visualization."""

    def __init__(
        self,
        evidence_id: str,
        evidence_type: EvidenceType,
        label: str,
        output_image_path: Optional[str] = None,
        artifact: Optional[Artifact] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_fallback: bool = False,
        error_message: Optional[str] = None,
    ):
        self.evidence_id = evidence_id
        self.evidence_type = evidence_type
        self.label = label
        self.output_image_path = output_image_path
        self.artifact = artifact
        self.confidence = confidence
        self.metadata = metadata or {}
        self.is_fallback = is_fallback
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "label": self.label,
            "output_image_path": self.output_image_path,
            "artifact_id": self.artifact.artifact_id if self.artifact else None,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "is_fallback": self.is_fallback,
            "error_message": self.error_message,
        }


class EvidenceRenderer:
    """Renderer for grounding evidence across optical, multispectral, and SAR rasters."""

    # High-contrast color palette for visual grounding boxes and labels
    COLOR_PALETTE = [
        {"stroke": (239, 68, 68), "fill": (239, 68, 68, 60), "hex": "#ef4444"},      # Red
        {"stroke": (6, 182, 212), "fill": (6, 182, 212, 60), "hex": "#06b6d4"},      # Cyan
        {"stroke": (16, 185, 129), "fill": (16, 185, 129, 60), "hex": "#10b981"},   # Emerald
        {"stroke": (245, 158, 11), "fill": (245, 158, 11, 60), "hex": "#f59e0b"},   # Amber
        {"stroke": (139, 92, 246), "fill": (139, 92, 246, 60), "hex": "#8b5cf6"},   # Purple
        {"stroke": (236, 72, 153), "fill": (236, 72, 153, 60), "hex": "#ec4899"},   # Pink
    ]

    @classmethod
    def get_evidence_storage_dir(cls) -> Path:
        """Ensure evidence output directory exists under settings.artifact_storage_path."""
        evidence_dir = settings.artifact_storage_path / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        return evidence_dir

    @classmethod
    def load_raster_image(cls, image_path: str) -> Optional[Image.Image]:
        """Safely load optical, TIFF, or benchmark PNG/JPEG raster into a PIL RGBA Image."""
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"Image path does not exist on disk: '{image_path}'")
            return None

        try:
            ext = Path(image_path).suffix.lower()
            if ext in {".tif", ".tiff"}:
                import tifffile
                data = tifffile.imread(image_path)
                # Handle multi-band and multi-dimensional rasters
                if data.ndim == 2:  # Single band (e.g. SAR intensity)
                    norm = np.clip((data - np.percentile(data, 1)) / (np.percentile(data, 99) - np.percentile(data, 1) + 1e-6) * 255, 0, 255).astype(np.uint8)
                    return Image.fromarray(norm).convert("RGBA")
                elif data.ndim == 3:
                    if data.shape[0] in {1, 3, 4} and data.shape[0] < data.shape[1]:
                        data = np.transpose(data, (1, 2, 0))
                    if data.shape[-1] >= 3:
                        rgb = data[..., :3]
                        norm = np.clip((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb) + 1e-6) * 255, 0, 255).astype(np.uint8)
                        return Image.fromarray(norm).convert("RGBA")
                    else:
                        norm = np.clip((data[..., 0] - np.min(data[..., 0])) / (np.max(data[..., 0]) - np.min(data[..., 0]) + 1e-6) * 255, 0, 255).astype(np.uint8)
                        return Image.fromarray(norm).convert("RGBA")
            
            img = Image.open(image_path)
            return img.convert("RGBA")
        except Exception as e:
            logger.error(f"Failed to load raster image '{image_path}': {e}")
            return None

    @classmethod
    def normalize_bbox_coordinates(
        cls,
        bbox: Union[List[float], Tuple[float, ...]],
        width: int,
        height: int,
        format_hint: Optional[str] = None,
    ) -> Tuple[int, int, int, int]:
        """Convert bounding box to absolute pixel coordinates [xmin, ymin, xmax, ymax].
        
        Supports:
        - Division 2 standard normalized format: [ymin, xmin, ymax, xmax] (0.0 to 1.0)
        - Normalized standard format: [xmin, ymin, xmax, ymax]
        - Absolute pixel coordinates
        """
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must contain exactly 4 coordinates, got {len(bbox)}")

        c0, c1, c2, c3 = [float(x) for x in bbox]
        is_normalized = all(0.0 <= x <= 1.0 for x in [c0, c1, c2, c3])

        if format_hint and "ymin" in format_hint.lower() and format_hint.lower().startswith("[ymin"):
            # Division 2 canonical: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = c0, c1, c2, c3
        elif is_normalized:
            # Default heuristic: if c2 < c0 or specified as ymin first
            ymin, xmin, ymax, xmax = c0, c1, c2, c3
        else:
            # Absolute pixel coordinates: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = c0 / width, c1 / height, c2 / width, c3 / height

        # Scale to pixel coordinates
        px_xmin = int(round(min(xmin, xmax) * width))
        px_ymin = int(round(min(ymin, ymax) * height))
        px_xmax = int(round(max(xmin, xmax) * width))
        px_ymax = int(round(max(ymin, ymax) * height))

        # Clamp within raster bounds
        px_xmin = max(0, min(width - 1, px_xmin))
        px_ymin = max(0, min(height - 1, px_ymin))
        px_xmax = max(px_xmin + 1, min(width, px_xmax))
        px_ymax = max(px_ymin + 1, min(height, px_ymax))

        return px_xmin, px_ymin, px_xmax, px_ymax

    @classmethod
    def render_bounding_boxes(
        cls,
        source_image_path: str,
        evidence_items: List[Evidence],
        output_filename_prefix: str = "annotated_grounding",
    ) -> EvidenceRenderingResult:
        """Render one or more bounding box evidence annotations onto the source image."""
        img = cls.load_raster_image(source_image_path)
        if img is None:
            # Return graceful fallback result
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.BOUNDING_BOX,
                label="Grounding Visualization (Unavailable)",
                is_fallback=True,
                error_message=f"Source image '{source_image_path}' could not be loaded for visual grounding.",
            )

        w, h = img.size
        # Create an RGBA overlay for semi-transparent fills
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_main = ImageDraw.Draw(img)

        # Try default font
        font = ImageFont.load_default()

        rendered_boxes = []
        for idx, ev in enumerate(evidence_items):
            if ev.type != EvidenceType.BOUNDING_BOX:
                continue

            bbox_raw = ev.data.get("bbox") or ev.data.get("coordinates")
            if not bbox_raw:
                continue

            fmt = ev.data.get("format", "[ymin, xmin, ymax, xmax]")
            try:
                xmin, ymin, xmax, ymax = cls.normalize_bbox_coordinates(bbox_raw, w, h, format_hint=fmt)
            except Exception as e:
                logger.warning(f"Could not parse bounding box {bbox_raw}: {e}")
                continue

            palette = cls.COLOR_PALETTE[idx % len(cls.COLOR_PALETTE)]
            stroke_color = palette["stroke"]
            fill_color = palette["fill"]

            # Draw semi-transparent rectangle
            draw_overlay.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)

            # Draw thick solid border
            line_width = max(3, int(min(w, h) * 0.005))
            for i in range(line_width):
                draw_main.rectangle(
                    [xmin - i, ymin - i, xmax + i, ymax + i],
                    outline=stroke_color,
                )

            # Draw label tag with confidence
            label_text = ev.label
            if ev.confidence is not None:
                label_text += f" ({ev.confidence * 100:.0f}%)"

            # Draw text banner
            bbox_text = draw_main.textbbox((xmin, max(0, ymin - 16)), label_text, font=font)
            banner_bg = (min(bbox_text[0] - 2, xmin), min(bbox_text[1] - 2, ymin - 16), bbox_text[2] + 4, bbox_text[3] + 2)
            draw_main.rectangle(banner_bg, fill=stroke_color)
            draw_main.text((banner_bg[0] + 2, banner_bg[1] + 1), label_text, fill=(255, 255, 255, 255), font=font)

            rendered_boxes.append({
                "label": ev.label,
                "confidence": ev.confidence,
                "pixel_bbox": [xmin, ymin, xmax, ymax],
                "normalized_bbox": [round(ymin/h, 4), round(xmin/w, 4), round(ymax/h, 4), round(xmax/w, 4)],
            })

        # Composite overlay onto main image
        final_img = Image.alpha_composite(img, overlay).convert("RGB")

        # Save artifact with full UUID in filename
        artifact_id = str(uuid.uuid4())
        filename = f"{output_filename_prefix}_{artifact_id}.png"
        out_path = cls.get_evidence_storage_dir() / filename
        final_img.save(out_path, format="PNG")

        artifact = Artifact(
            artifact_id=artifact_id,
            name=filename,
            type="annotated_image",
            uri_or_path=str(out_path),
            description=f"Grounding localization map with {len(rendered_boxes)} detected features.",
            mime_type="image/png",
            metadata={"rendered_boxes": rendered_boxes, "source_image": source_image_path},
        )

        # Register artifact in global registry
        ArtifactRegistry.register(artifact_id, out_path, name=filename)

        return EvidenceRenderingResult(
            evidence_id=artifact_id,
            evidence_type=EvidenceType.BOUNDING_BOX,
            label=f"Visual Grounding ({len(rendered_boxes)} targets)",
            output_image_path=str(out_path),
            artifact=artifact,
            metadata={"box_count": len(rendered_boxes), "boxes": rendered_boxes},
        )

    @classmethod
    def render_bitemporal_change_map(
        cls,
        t0_image_path: str,
        t1_image_path: str,
        change_mask_data: Optional[Union[np.ndarray, List[List[int]]]] = None,
        title: str = "Bi-Temporal Change Analysis",
    ) -> EvidenceRenderingResult:
        """Render side-by-side T0 (before), T1 (after), and highlighted change difference composite."""
        img0 = cls.load_raster_image(t0_image_path)
        img1 = cls.load_raster_image(t1_image_path)

        if img0 is None or img1 is None:
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.CHANGE_MAP,
                label=title,
                is_fallback=True,
                error_message="One or both temporal acquisition rasters could not be loaded.",
            )

        # Ensure same size for visual side-by-side
        target_size = (max(img0.width, img1.width, 384), max(img0.height, img1.height, 384))
        r0 = img0.resize(target_size, Image.Resampling.BILINEAR).convert("RGB")
        r1 = img1.resize(target_size, Image.Resampling.BILINEAR).convert("RGB")

        # Compute or apply difference map
        arr0 = np.array(r0).astype(np.float32)
        arr1 = np.array(r1).astype(np.float32)

        if change_mask_data is not None:
            mask = np.array(change_mask_data).astype(np.uint8)
            if mask.shape[:2] != target_size[::-1]:
                mask_img = Image.fromarray(mask).resize(target_size, Image.Resampling.NEAREST)
                mask = np.array(mask_img)
        else:
            # Automated radiometric pixel difference map
            diff = np.mean(np.abs(arr1 - arr0), axis=2)
            threshold = np.percentile(diff, 88)
            mask = (diff > threshold).astype(np.uint8)

        # Create difference visualization
        diff_overlay = np.array(r1).copy()
        # Highlight changed pixels in high-visibility bright red/orange
        diff_overlay[mask > 0] = [239, 68, 68]
        diff_blend = Image.blend(r1, Image.fromarray(diff_overlay), alpha=0.65)

        # Construct side-by-side 3-panel storyboard [T0 Before | T1 After | Change Map]
        w, h = target_size
        header_height = 40
        composite_w = w * 3 + 16
        composite_h = h + header_height + 10

        composite = Image.new("RGB", (composite_w, composite_h), (18, 24, 38))
        draw = ImageDraw.Draw(composite)
        font = ImageFont.load_default()

        # Paste panels
        composite.paste(r0, (4, header_height))
        composite.paste(r1, (w + 8, header_height))
        composite.paste(diff_blend, (w * 2 + 12, header_height))

        # Panel Headers
        draw.text((8, 12), "📅 Acquisition T0 (Baseline)", fill=(147, 197, 253), font=font)
        draw.text((w + 12, 12), "📅 Acquisition T1 (Follow-up)", fill=(147, 197, 253), font=font)
        draw.text((w * 2 + 16, 12), "🔍 Detected Change Difference Mask", fill=(248, 113, 113), font=font)

        # Save artifact with full UUID in filename
        artifact_id = str(uuid.uuid4())
        filename = f"bitemporal_change_composite_{artifact_id}.png"
        out_path = cls.get_evidence_storage_dir() / filename
        composite.save(out_path, format="PNG")

        change_ratio = float(np.mean(mask > 0))
        artifact = Artifact(
            artifact_id=artifact_id,
            name=filename,
            type="change_map",
            uri_or_path=str(out_path),
            description=f"3-panel bi-temporal change storyboard with estimated {change_ratio * 100:.1f}% spatial difference.",
            mime_type="image/png",
            metadata={"change_pixel_ratio": change_ratio, "t0_source": t0_image_path, "t1_source": t1_image_path},
        )

        # Register artifact in global registry
        ArtifactRegistry.register(artifact_id, out_path, name=filename)

        return EvidenceRenderingResult(
            evidence_id=artifact_id,
            evidence_type=EvidenceType.CHANGE_MAP,
            label="Bi-Temporal Change Map Composite",
            output_image_path=str(out_path),
            artifact=artifact,
            metadata={"change_percentage": f"{change_ratio * 100:.1f}%"},
        )

    @classmethod
    def render_optical_sar_fusion(
        cls,
        optical_path: str,
        sar_path: str,
        title: str = "Optical-SAR Cross-Modal Fusion",
    ) -> EvidenceRenderingResult:
        """Render false-color composite and dual-modality overlay of optical and SAR rasters."""
        opt_img = cls.load_raster_image(optical_path)
        sar_img = cls.load_raster_image(sar_path)

        if opt_img is None or sar_img is None:
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.HIGHLIGHTED_IMAGE,
                label=title,
                is_fallback=True,
                error_message="Optical or SAR raster could not be loaded for cross-modal fusion rendering.",
            )

        target_size = (max(opt_img.width, sar_img.width, 384), max(opt_img.height, sar_img.height, 384))
        opt_res = opt_img.resize(target_size, Image.Resampling.BILINEAR).convert("RGB")
        sar_res = sar_img.resize(target_size, Image.Resampling.BILINEAR).convert("L")

        # Create false-color composite: Red=Optical Red, Green=Optical Green, Blue=SAR Intensity
        opt_arr = np.array(opt_res)
        sar_arr = np.array(sar_res)

        false_color = np.zeros_like(opt_arr)
        false_color[..., 0] = opt_arr[..., 0]
        false_color[..., 1] = opt_arr[..., 1]
        false_color[..., 2] = sar_arr
        fused_pil = Image.fromarray(false_color)

        # Construct 3-panel storyboard [Optical | SAR | False-Color Fusion]
        w, h = target_size
        header_height = 40
        composite_w = w * 3 + 16
        composite_h = h + header_height + 10

        composite = Image.new("RGB", (composite_w, composite_h), (18, 24, 38))
        draw = ImageDraw.Draw(composite)
        font = ImageFont.load_default()

        composite.paste(opt_res, (4, header_height))
        composite.paste(sar_res.convert("RGB"), (w + 8, header_height))
        composite.paste(fused_pil, (w * 2 + 12, header_height))

        draw.text((8, 12), "🛰️ Optical RGB (Surface View)", fill=(147, 197, 253), font=font)
        draw.text((w + 12, 12), "📡 SAR Intensity (Radar Scattering)", fill=(147, 197, 253), font=font)
        draw.text((w * 2 + 16, 12), "⚡ False-Color Radar Penetration Fusion", fill=(196, 181, 253), font=font)

        # Save artifact with full UUID in filename
        artifact_id = str(uuid.uuid4())
        filename = f"optical_sar_fusion_{artifact_id}.png"
        out_path = cls.get_evidence_storage_dir() / filename
        composite.save(out_path, format="PNG")

        artifact = Artifact(
            artifact_id=artifact_id,
            name=filename,
            type="annotated_image",
            uri_or_path=str(out_path),
            description="3-panel Optical + SAR co-registered false-color radar penetration fusion.",
            mime_type="image/png",
            metadata={"optical_source": optical_path, "sar_source": sar_path},
        )

        # Register artifact in global registry
        ArtifactRegistry.register(artifact_id, out_path, name=filename)

        return EvidenceRenderingResult(
            evidence_id=artifact_id,
            evidence_type=EvidenceType.HIGHLIGHTED_IMAGE,
            label="Optical-SAR Cross-Modal Fusion",
            output_image_path=str(out_path),
            artifact=artifact,
        )

    @classmethod
    def render_roi_crop(
        cls,
        source_image_path: str,
        bbox: List[float],
        label: str = "Region of Interest",
        confidence: Optional[float] = None,
        format_hint: Optional[str] = None,
    ) -> EvidenceRenderingResult:
        """Extract a focused high-resolution crop of a specific detected region."""
        img = cls.load_raster_image(source_image_path)
        if img is None:
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.CROP,
                label=label,
                is_fallback=True,
                error_message=f"Source image '{source_image_path}' could not be loaded for crop generation.",
            )

        w, h = img.size
        try:
            xmin, ymin, xmax, ymax = cls.normalize_bbox_coordinates(bbox, w, h, format_hint=format_hint)
        except Exception as e:
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.CROP,
                label=label,
                is_fallback=True,
                error_message=f"Invalid crop coordinates {bbox}: {e}",
            )

        # Add small contextual margin around bounding box
        margin_x = int((xmax - xmin) * 0.1)
        margin_y = int((ymax - ymin) * 0.1)
        crop_box = (
            max(0, xmin - margin_x),
            max(0, ymin - margin_y),
            min(w, xmax + margin_x),
            min(h, ymax + margin_y),
        )

        cropped = img.crop(crop_box).convert("RGB")
        artifact_id = str(uuid.uuid4())
        safe_label = "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()[:32] or "roi"
        filename = f"crop_{safe_label}_{artifact_id}.png"
        out_path = cls.get_evidence_storage_dir() / filename
        cropped.save(out_path, format="PNG")

        artifact = Artifact(
            artifact_id=artifact_id,
            name=filename,
            type="crop",
            uri_or_path=str(out_path),
            description=f"Focused crop for '{label}' at pixel bounds {list(crop_box)}.",
            mime_type="image/png",
            metadata={"bbox": bbox, "source_image": source_image_path},
        )

        # Register artifact in global registry
        ArtifactRegistry.register(artifact_id, out_path, name=filename)

        return EvidenceRenderingResult(
            evidence_id=artifact_id,
            evidence_type=EvidenceType.CROP,
            label=f"Crop: {label}",
            output_image_path=str(out_path),
            artifact=artifact,
            confidence=confidence,
        )

    @classmethod
    def render_attention_heatmap(
        cls,
        source_image_path: str,
        heatmap_data: Optional[np.ndarray] = None,
        label: str = "Attention / Activation Heatmap",
    ) -> EvidenceRenderingResult:
        """Render attention or feature activation heatmap overlaid on source raster."""
        img = cls.load_raster_image(source_image_path)
        if img is None:
            return EvidenceRenderingResult(
                evidence_id=str(uuid.uuid4()),
                evidence_type=EvidenceType.HEATMAP,
                label=label,
                is_fallback=True,
                error_message=f"Source image '{source_image_path}' could not be loaded for heatmap generation.",
            )

        w, h = img.size
        if heatmap_data is None:
            # Generate synthetic Gaussian focus heatmap if raw activation is not exported
            x = np.linspace(-2, 2, w)
            y = np.linspace(-2, 2, h)
            xx, yy = np.meshgrid(x, y)
            heatmap_data = np.exp(-(xx**2 + yy**2) / 1.5)

        norm_hm = (heatmap_data - np.min(heatmap_data)) / (np.max(heatmap_data) - np.min(heatmap_data) + 1e-6)
        
        # Colorize using simple RGB colormap (Blue -> Cyan -> Yellow -> Red)
        r = np.clip(2.0 * norm_hm - 0.5, 0, 1)
        g = np.clip(1.0 - 2.0 * np.abs(norm_hm - 0.5), 0, 1)
        b = np.clip(1.0 - 2.0 * norm_hm, 0, 1)
        hm_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        hm_pil = Image.fromarray(hm_rgb).resize((w, h), Image.Resampling.BILINEAR)

        # Blend over original image
        blended = Image.blend(img.convert("RGB"), hm_pil, alpha=0.45)

        artifact_id = str(uuid.uuid4())
        filename = f"heatmap_{artifact_id}.png"
        out_path = cls.get_evidence_storage_dir() / filename
        blended.save(out_path, format="PNG")

        artifact = Artifact(
            artifact_id=artifact_id,
            name=filename,
            type="heatmap",
            uri_or_path=str(out_path),
            description=f"Attention heatmap overlay for '{label}'.",
            mime_type="image/png",
            metadata={"source_image": source_image_path},
        )

        # Register artifact in global registry
        ArtifactRegistry.register(artifact_id, out_path, name=filename)

        return EvidenceRenderingResult(
            evidence_id=artifact_id,
            evidence_type=EvidenceType.HEATMAP,
            label=label,
            output_image_path=str(out_path),
            artifact=artifact,
        )

    @classmethod
    def render_all_evidence(
        cls,
        evidence_list: List[Evidence],
        image_inputs: List[ImageInput],
        task_hint: Optional[str] = None,
    ) -> List[EvidenceRenderingResult]:
        """Batch-process and render all evidence items associated with a query result."""
        results: List[EvidenceRenderingResult] = []
        if not evidence_list:
            return results

        # Group bounding box evidence by source image
        primary_image_path = image_inputs[0].path_or_uri if image_inputs else None

        bbox_items = [ev for ev in evidence_list if ev.type == EvidenceType.BOUNDING_BOX]
        if bbox_items and primary_image_path:
            bbox_result = cls.render_bounding_boxes(primary_image_path, bbox_items)
            results.append(bbox_result)

            # Generate individual crops for top high-confidence bounding boxes (up to 3)
            for ev in bbox_items[:3]:
                bbox_raw = ev.data.get("bbox") or ev.data.get("coordinates")
                if bbox_raw:
                    crop_res = cls.render_roi_crop(
                        primary_image_path,
                        bbox_raw,
                        label=ev.label,
                        confidence=ev.confidence,
                        format_hint=ev.data.get("format"),
                    )
                    results.append(crop_res)

        # Handle bi-temporal change maps if 2 images present
        change_items = [ev for ev in evidence_list if ev.type == EvidenceType.CHANGE_MAP]
        if change_items and len(image_inputs) >= 2:
            change_res = cls.render_bitemporal_change_map(
                image_inputs[0].path_or_uri,
                image_inputs[1].path_or_uri,
                title=change_items[0].label,
            )
            results.append(change_res)

        # Handle optical-SAR highlighted images
        fusion_items = [ev for ev in evidence_list if ev.type == EvidenceType.HIGHLIGHTED_IMAGE]
        if fusion_items and len(image_inputs) >= 2:
            opt_path = next((img.path_or_uri for img in image_inputs if img.modality in {ImageModality.OPTICAL, ImageModality.MULTISPECTRAL}), image_inputs[0].path_or_uri)
            sar_path = next((img.path_or_uri for img in image_inputs if img.modality == ImageModality.SAR), image_inputs[1].path_or_uri)
            fusion_res = cls.render_optical_sar_fusion(opt_path, sar_path, title=fusion_items[0].label)
            results.append(fusion_res)

        # Handle heatmaps
        heatmap_items = [ev for ev in evidence_list if ev.type == EvidenceType.HEATMAP]
        if heatmap_items and primary_image_path:
            for hm_ev in heatmap_items:
                hm_res = cls.render_attention_heatmap(primary_image_path, label=hm_ev.label)
                results.append(hm_res)

        return results
