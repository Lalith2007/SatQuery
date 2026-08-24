"""Geospatial and Remote-Sensing Raster Input Validation Framework.

Performs raster inspection, strict format validation (GeoTIFF/TIFF primary,
PNG/JPEG benchmark), task-modality compatibility checks, and pair compatibility
validation without fabricating missing geospatial metadata or enforcing artificial
pixel-dimension matching across different sensors.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import tifffile

from core.errors import (
    IncompatiblePairError,
    InputValidationError,
    InvalidImageCountError,
    MissingMetadataError,
    UnsupportedFormatError,
    UnsupportedModalityError,
)
from core.logging import get_logger
from core.schemas import (
    GeoSpatialMetadata,
    ImageFormat,
    ImageInput,
    ImageModality,
    TaskType,
)

logger = get_logger("validation")

# Strictly allowed file extensions
ALLOWED_EXTENSIONS = {
    ".tif": ImageFormat.TIFF,
    ".tiff": ImageFormat.TIFF,
    ".geotiff": ImageFormat.GEOTIFF,
    ".png": ImageFormat.PNG,
    ".jpg": ImageFormat.JPEG,
    ".jpeg": ImageFormat.JPEG,
}


class RasterInspector:
    """Safely inspects raster headers and metadata on disk."""

    @staticmethod
    def inspect_file(file_path: Path | str) -> Tuple[int, int, int, str, Optional[GeoSpatialMetadata]]:
        """Inspect a raster file and extract width, height, bands, dtype, and verified geospatial tags."""
        path = Path(file_path)
        if not path.exists():
            raise InputValidationError(f"Raster file does not exist at path: {path}")

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"File format '{ext}' is not supported. Primary supported formats are GeoTIFF/TIFF (.tif, .tiff) "
                f"and benchmark formats (.png, .jpg, .jpeg)."
            )

        # 1. TIFF / GeoTIFF inspection via tifffile
        if ext in {".tif", ".tiff", ".geotiff"}:
            try:
                with tifffile.TiffFile(str(path)) as tif:
                    if not tif.pages:
                        raise InputValidationError(f"TIFF file contains no readable pages: {path}")
                    page = tif.pages[0]
                    height, width = page.shape[0], page.shape[1]
                    
                    # Handle shapes: (H, W), (H, W, C), or (C, H, W)
                    if len(page.shape) == 2:
                        channel_count = 1
                    elif len(page.shape) == 3:
                        # If first dimension is small (e.g. bands), could be band-first
                        if page.shape[0] < min(page.shape[1], page.shape[2]):
                            channel_count = page.shape[0]
                            height, width = page.shape[1], page.shape[2]
                        else:
                            channel_count = page.shape[2]
                            height, width = page.shape[0], page.shape[1]
                    else:
                        channel_count = 1

                    dtype_str = str(page.dtype)

                    # Extract geospatial tags if present without fabricating
                    geospatial = None
                    geokeys = getattr(page, "geotiff_tags", None) or getattr(page, "tags", {})
                    
                    # Check if GeoTIFF tags exist
                    if hasattr(tif, "geotiff_metadata") and tif.geotiff_metadata:
                        geo_meta = tif.geotiff_metadata
                        crs = geo_meta.get("GTCitationGeoKey") or geo_meta.get("GeographicTypeGeoKey")
                        geospatial = GeoSpatialMetadata(
                            crs=str(crs) if crs else None,
                            metadata={"geotiff_keys": list(geo_meta.keys())},
                        )
                    
                    return width, height, channel_count, dtype_str, geospatial
            except Exception as e:
                logger.warning(f"tifffile inspection failed for {path}, falling back to PIL: {e}")
                # Fallback to PIL for non-standard TIFFs
                return RasterInspector._inspect_with_pil(path)

        # 2. Standard benchmark image formats (PNG, JPEG) via PIL
        return RasterInspector._inspect_with_pil(path)

    @staticmethod
    def _inspect_with_pil(path: Path) -> Tuple[int, int, int, str, Optional[GeoSpatialMetadata]]:
        """Inspect PNG/JPEG benchmark raster images using PIL."""
        try:
            with Image.open(str(path)) as img:
                width, height = img.size
                mode = img.mode
                mode_to_channels = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4, "I": 1, "F": 1}
                channel_count = mode_to_channels.get(mode, len(img.getbands()))
                dtype_str = "uint8" if mode in {"RGB", "RGBA", "L", "P"} else "float32"
                return width, height, channel_count, dtype_str, None
        except Exception as err:
            raise InputValidationError(f"Failed to read raster image at '{path}': {err}")


class InputValidator:
    """Core validator for SatQuery requests and image configurations."""

    @classmethod
    def validate_image_input(cls, img: ImageInput, inspect_disk: bool = True) -> ImageInput:
        """Validate a single ImageInput, optionally inspecting on-disk raster."""
        # 1. Format validation
        if not isinstance(img.format, ImageFormat):
            raise UnsupportedFormatError(f"Invalid image format: {img.format}")

        # 2. Modality validation
        if not isinstance(img.modality, ImageModality):
            raise UnsupportedModalityError(f"Invalid image modality: {img.modality}")

        # 3. Disk inspection if local file exists and requested
        if inspect_disk and not img.path_or_uri.startswith(("http://", "https://", "mock://")):
            path = Path(img.path_or_uri)
            if path.exists():
                w, h, channels, dtype, geo = RasterInspector.inspect_file(path)
                # Update image attributes from verified on-disk raster
                img.width = w
                img.height = h
                img.channel_count = channels
                img.dtype = dtype
                if geo and not img.geospatial:
                    img.geospatial = geo
            else:
                # If path does not exist on disk and isn't a URI
                raise InputValidationError(f"Specified image file not found on disk: {img.path_or_uri}")

        return img

    @classmethod
    def validate_task_compatibility(
        cls,
        task: TaskType,
        images: List[ImageInput],
        inspect_disk: bool = False,
    ) -> None:
        """Validate that the image collection satisfies task requirements."""
        if not images:
            raise InvalidImageCountError("At least one image input is required.")

        # Validate each individual image first
        for idx, img in enumerate(images):
            cls.validate_image_input(img, inspect_disk=inspect_disk)

        # Task-specific cardinality & modality validation
        if task in {TaskType.SINGLE_IMAGE_VQA, TaskType.SINGLE_IMAGE_CAPTION, TaskType.SINGLE_IMAGE_GROUNDING}:
            cls._validate_single_image_task(task, images)
        elif task in {TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA}:
            cls._validate_bitemporal_task(task, images)
        elif task == TaskType.OPTICAL_SAR_ANALYSIS:
            cls._validate_optical_sar_task(task, images)
        else:
            raise InputValidationError(f"Unrecognized task type: {task}")

    @classmethod
    def _validate_single_image_task(cls, task: TaskType, images: List[ImageInput]) -> None:
        """Validate requirements for single-image tasks."""
        if len(images) != 1:
            raise InvalidImageCountError(
                f"Task '{task.value}' requires exactly 1 image, but {len(images)} images were provided."
            )

    @classmethod
    def _validate_bitemporal_task(cls, task: TaskType, images: List[ImageInput]) -> None:
        """Validate bi-temporal pair requirements.
        
        Note: Does NOT require identical pixel dimensions.
        """
        if len(images) < 2:
            raise InvalidImageCountError(
                f"Task '{task.value}' requires at least 2 images for bi-temporal analysis, but {len(images)} was provided."
            )
        if len(images) > 2:
            raise InvalidImageCountError(
                f"Task '{task.value}' currently supports pairs (2 images), but {len(images)} were provided."
            )

        img_t0, img_t1 = images[0], images[1]

        # Check temporal timestamps if present (never fabricated)
        t0 = img_t0.geospatial.acquisition_timestamp if img_t0.geospatial else None
        t1 = img_t1.geospatial.acquisition_timestamp if img_t1.geospatial else None

        if t0 and t1:
            # If both have timestamps, verify they are not identical
            if t0 == t1:
                logger.info("Bi-temporal pair has identical timestamps; proceeding under assumed micro-interval.")

        # Check geographic overlap if bounds are present on both (never fabricated)
        b0 = img_t0.geospatial.geo_bounds if img_t0.geospatial else None
        b1 = img_t1.geospatial.geo_bounds if img_t1.geospatial else None

        if b0 and b1:
            # Check bounding box intersection: [minx, miny, maxx, maxy]
            if not cls._bboxes_intersect(b0, b1):
                raise IncompatiblePairError(
                    "Bi-temporal images have non-overlapping geographic bounding boxes: "
                    f"T0 bounds={b0}, T1 bounds={b1}"
                )

    @classmethod
    def _validate_optical_sar_task(cls, task: TaskType, images: List[ImageInput]) -> None:
        """Validate optical-SAR cross-modal pair requirements.
        
        Requires one Optical/Multispectral image and one SAR image.
        Does NOT require identical raster pixel dimensions.
        """
        if len(images) != 2:
            raise InvalidImageCountError(
                f"Task '{task.value}' requires exactly 2 images (1 Optical/Multispectral and 1 SAR), "
                f"but {len(images)} were provided."
            )

        modalities = [img.modality for img in images]
        has_optical = any(m in {ImageModality.OPTICAL, ImageModality.MULTISPECTRAL} for m in modalities)
        has_sar = any(m == ImageModality.SAR for m in modalities)

        if not (has_optical and has_sar):
            # If modalities are UNKNOWN, log warning or raise if strictly required
            if all(m == ImageModality.UNKNOWN for m in modalities):
                logger.warning("Optical-SAR analysis requested with UNKNOWN image modalities; proceeding with caution.")
            else:
                raise IncompatiblePairError(
                    "Optical-SAR cross-modal analysis requires 1 Optical/Multispectral image and 1 SAR image. "
                    f"Provided modalities were: {[m.value for m in modalities]}"
                )

        # Check geographic overlap if bounds exist
        b0 = images[0].geospatial.geo_bounds if images[0].geospatial else None
        b1 = images[1].geospatial.geo_bounds if images[1].geospatial else None

        if b0 and b1:
            if not cls._bboxes_intersect(b0, b1):
                raise IncompatiblePairError(
                    "Optical and SAR images do not geographically overlap: "
                    f"Image 1 bounds={b0}, Image 2 bounds={b1}"
                )

    @staticmethod
    def _bboxes_intersect(b1: List[float], b2: List[float]) -> bool:
        """Check if two [minx, miny, maxx, maxy] bounding boxes intersect."""
        if len(b1) != 4 or len(b2) != 4:
            return True
        minx1, miny1, maxx1, maxy1 = b1
        minx2, miny2, maxx2, maxy2 = b2
        return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)
