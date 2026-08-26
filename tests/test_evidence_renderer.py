"""Unit tests for spatial evidence rendering engine (Division 5)."""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from core.schemas import Evidence, EvidenceType, ImageFormat, ImageInput, ImageModality
from presentation.evidence_renderer import EvidenceRenderer, EvidenceRenderingResult


@pytest.fixture
def sample_raster_png(tmp_path) -> str:
    """Create synthetic optical RGB PNG raster for testing."""
    img_path = tmp_path / "test_optical.png"
    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return str(img_path)


@pytest.fixture
def sample_temporal_pair(tmp_path) -> tuple[str, str]:
    """Create synthetic T0 and T1 optical PNG rasters for testing."""
    t0_path = tmp_path / "test_t0.png"
    t1_path = tmp_path / "test_t1.png"
    arr0 = np.full((256, 256, 3), 100, dtype=np.uint8)
    arr1 = np.full((256, 256, 3), 100, dtype=np.uint8)
    # Add artificial change region to T1
    arr1[50:120, 50:120] = [220, 50, 50]
    Image.fromarray(arr0).save(t0_path)
    Image.fromarray(arr1).save(t1_path)
    return str(t0_path), str(t1_path)


def test_normalize_bbox_coordinates():
    """Verify coordinate normalization from normalized [ymin, xmin, ymax, xmax] to pixel coordinates."""
    w, h = 500, 400
    # Normalized [ymin, xmin, ymax, xmax]
    norm_box = [0.1, 0.2, 0.6, 0.8]
    px_box = EvidenceRenderer.normalize_bbox_coordinates(norm_box, w, h, format_hint="[ymin, xmin, ymax, xmax]")
    xmin, ymin, xmax, ymax = px_box
    assert xmin == 100
    assert ymin == 40
    assert xmax == 400
    assert ymax == 240


def test_render_bounding_boxes(sample_raster_png):
    """Verify bounding box rendering produces valid PNG artifact and metadata."""
    evidence_items = [
        Evidence(
            type=EvidenceType.BOUNDING_BOX,
            label="Aircraft Apron",
            confidence=0.92,
            data={"bbox": [0.1, 0.2, 0.5, 0.7], "format": "[ymin, xmin, ymax, xmax]"},
        ),
        Evidence(
            type=EvidenceType.BOUNDING_BOX,
            label="Runway Target",
            confidence=0.88,
            data={"bbox": [0.6, 0.1, 0.9, 0.9], "format": "[ymin, xmin, ymax, xmax]"},
        ),
    ]

    result = EvidenceRenderer.render_bounding_boxes(sample_raster_png, evidence_items)
    assert not result.is_fallback
    assert result.output_image_path is not None
    assert os.path.exists(result.output_image_path)
    assert result.artifact is not None
    assert result.artifact.type == "annotated_image"
    assert result.metadata["box_count"] == 2


def test_render_bitemporal_change_map(sample_temporal_pair):
    """Verify bi-temporal change map storyboard rendering."""
    t0_path, t1_path = sample_temporal_pair
    result = EvidenceRenderer.render_bitemporal_change_map(t0_path, t1_path)

    assert not result.is_fallback
    assert result.output_image_path is not None
    assert os.path.exists(result.output_image_path)
    assert result.artifact is not None
    assert result.artifact.type == "change_map"


def test_render_optical_sar_fusion(sample_raster_png, sample_temporal_pair):
    """Verify optical-SAR false-color cross-modal fusion rendering."""
    t0_path, _ = sample_temporal_pair
    result = EvidenceRenderer.render_optical_sar_fusion(sample_raster_png, t0_path)

    assert not result.is_fallback
    assert result.output_image_path is not None
    assert os.path.exists(result.output_image_path)
    assert result.artifact is not None


def test_render_roi_crop(sample_raster_png):
    """Verify focused region-of-interest crop extraction."""
    bbox = [0.2, 0.2, 0.6, 0.6]
    result = EvidenceRenderer.render_roi_crop(sample_raster_png, bbox, label="Terminal Building", confidence=0.95)

    assert not result.is_fallback
    assert result.output_image_path is not None
    assert os.path.exists(result.output_image_path)
    assert result.artifact.type == "crop"


def test_render_attention_heatmap(sample_raster_png):
    """Verify attention heatmap overlay generation."""
    result = EvidenceRenderer.render_attention_heatmap(sample_raster_png, label="Attention Peak")

    assert not result.is_fallback
    assert result.output_image_path is not None
    assert os.path.exists(result.output_image_path)
    assert result.artifact.type == "heatmap"


def test_render_missing_image_graceful_fallback():
    """Verify evidence renderer fails gracefully when raster image does not exist."""
    fake_path = "non_existent_raster_12345.png"
    result = EvidenceRenderer.render_bounding_boxes(
        fake_path,
        [Evidence(type=EvidenceType.BOUNDING_BOX, label="Fake", data={"bbox": [0, 0, 1, 1]})],
    )
    assert result.is_fallback
    assert result.output_image_path is None
    assert "could not be loaded" in result.error_message
