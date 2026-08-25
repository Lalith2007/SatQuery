"""Unit tests for remote-sensing raster input validation."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from core.errors import (
    IncompatiblePairError,
    InputValidationError,
    InvalidImageCountError,
    UnsupportedFormatError,
)
from core.schemas import (
    GeoSpatialMetadata,
    ImageFormat,
    ImageInput,
    ImageModality,
    TaskType,
)
from validation.validator import InputValidator, RasterInspector


def test_raster_inspector_png(sample_png_path: Path):
    w, h, channels, dtype, geo = RasterInspector.inspect_file(sample_png_path)
    assert w == 128
    assert h == 128
    assert channels == 3
    assert dtype == "uint8"
    assert geo is None  # Never fabricated for standard PNG


def test_raster_inspector_tiff(sample_tiff_path: Path):
    w, h, channels, dtype, geo = RasterInspector.inspect_file(sample_tiff_path)
    assert w == 256
    assert h == 256
    assert channels == 4
    assert dtype == "uint16"


def test_unsupported_format_rejection(temp_dir: Path):
    fake_webp = temp_dir / "unsupported.webp"
    fake_webp.write_text("dummy")
    with pytest.raises(UnsupportedFormatError) as exc_info:
        RasterInspector.inspect_file(fake_webp)
    assert "not supported" in str(exc_info.value)


def test_single_image_task_validation(optical_image_input: ImageInput):
    # Valid single image
    InputValidator.validate_task_compatibility(
        task=TaskType.SINGLE_IMAGE_VQA,
        images=[optical_image_input],
    )

    # Invalid: 2 images for single image task
    with pytest.raises(InvalidImageCountError):
        InputValidator.validate_task_compatibility(
            task=TaskType.SINGLE_IMAGE_VQA,
            images=[optical_image_input, optical_image_input],
        )


def test_bitemporal_task_validation_differing_dimensions(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
):
    # Verify raster dimensions differ
    assert optical_image_input.width != optical_image_t1_input.width

    # Valid: Bi-temporal pair with differing pixel dimensions is accepted
    InputValidator.validate_task_compatibility(
        task=TaskType.CHANGE_ANALYSIS,
        images=[optical_image_input, optical_image_t1_input],
    )


def test_bitemporal_task_invalid_count(optical_image_input: ImageInput):
    # Invalid: Only 1 image provided
    with pytest.raises(InvalidImageCountError):
        InputValidator.validate_task_compatibility(
            task=TaskType.CHANGE_ANALYSIS,
            images=[optical_image_input],
        )


def test_bitemporal_non_overlapping_bounds(optical_image_input: ImageInput, optical_image_t1_input: ImageInput):
    # Alter bounds to non-overlapping
    optical_image_t1_input.geospatial.geo_bounds = [10.0, 10.0, 12.0, 12.0]
    with pytest.raises(IncompatiblePairError) as exc_info:
        InputValidator.validate_task_compatibility(
            task=TaskType.CHANGE_ANALYSIS,
            images=[optical_image_input, optical_image_t1_input],
        )
    assert "non-overlapping" in str(exc_info.value)


def test_optical_sar_task_validation_differing_dimensions(
    optical_image_input: ImageInput,
    sar_image_input: ImageInput,
):
    # Verify dimensions differ
    assert optical_image_input.width != sar_image_input.width
    assert optical_image_input.modality == ImageModality.OPTICAL
    assert sar_image_input.modality == ImageModality.SAR

    # Valid: Optical + SAR pair accepted despite different dimensions
    InputValidator.validate_task_compatibility(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        images=[optical_image_input, sar_image_input],
    )


def test_optical_sar_incompatible_modalities(optical_image_input: ImageInput, optical_image_t1_input: ImageInput):
    # Invalid: Two optical images for Optical-SAR task
    with pytest.raises(IncompatiblePairError) as exc_info:
        InputValidator.validate_task_compatibility(
            task=TaskType.OPTICAL_SAR_ANALYSIS,
            images=[optical_image_input, optical_image_t1_input],
        )
    assert "Optical-SAR cross-modal analysis requires 1 Optical" in str(exc_info.value)
