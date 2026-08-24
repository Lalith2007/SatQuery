"""Pytest fixtures for SatQuery AI test suite."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Generator

import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient
import tifffile

from app.main import app
from core.schemas import (
    GeoSpatialMetadata,
    ImageFormat,
    ImageInput,
    ImageModality,
)
from registry.registry import ToolRegistry
from specialists.mock import register_default_mocks


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test raster files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_png_path(temp_dir: Path) -> Path:
    """Create a valid PNG image file on disk."""
    file_path = temp_dir / "sample_optical.png"
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(file_path)
    return file_path


@pytest.fixture
def sample_tiff_path(temp_dir: Path) -> Path:
    """Create a valid multi-band TIFF raster on disk."""
    file_path = temp_dir / "sample_multispectral.tif"
    arr = np.random.randint(0, 65535, (4, 256, 256), dtype=np.uint16)
    tifffile.imwrite(str(file_path), arr)
    return file_path


@pytest.fixture
def optical_image_input(sample_png_path: Path) -> ImageInput:
    """Provide a canonical optical ImageInput."""
    return ImageInput(
        image_id="img-optical-01",
        path_or_uri=str(sample_png_path),
        format=ImageFormat.PNG,
        modality=ImageModality.OPTICAL,
        width=128,
        height=128,
        channel_count=3,
        dtype="uint8",
        geospatial=GeoSpatialMetadata(
            crs="EPSG:4326",
            geo_bounds=[-122.5, 37.7, -122.3, 37.9],
            acquisition_timestamp=datetime(2023, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
            sensor="Sentinel-2",
        ),
    )


@pytest.fixture
def optical_image_t1_input(temp_dir: Path) -> ImageInput:
    """Provide a canonical optical ImageInput for time T1 with different raster dimensions."""
    file_path = temp_dir / "sample_optical_t1.png"
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr).save(file_path)
    return ImageInput(
        image_id="img-optical-t1",
        path_or_uri=str(file_path),
        format=ImageFormat.PNG,
        modality=ImageModality.OPTICAL,
        width=200,
        height=200,
        channel_count=3,
        dtype="uint8",
        geospatial=GeoSpatialMetadata(
            crs="EPSG:4326",
            geo_bounds=[-122.5, 37.7, -122.3, 37.9],
            acquisition_timestamp=datetime(2024, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
            sensor="Sentinel-2",
        ),
    )


@pytest.fixture
def sar_image_input(temp_dir: Path) -> ImageInput:
    """Provide a canonical SAR ImageInput with different dimensions."""
    file_path = temp_dir / "sample_sar.tif"
    arr = np.random.rand(150, 150).astype(np.float32)
    tifffile.imwrite(str(file_path), arr)
    return ImageInput(
        image_id="img-sar-01",
        path_or_uri=str(file_path),
        format=ImageFormat.TIFF,
        modality=ImageModality.SAR,
        width=150,
        height=150,
        channel_count=1,
        dtype="float32",
        geospatial=GeoSpatialMetadata(
            crs="EPSG:4326",
            geo_bounds=[-122.5, 37.7, -122.3, 37.9],
            acquisition_timestamp=datetime(2023, 5, 1, 10, 0, 0, tzinfo=timezone.utc),
            sensor="Sentinel-1",
        ),
    )


@pytest.fixture
def test_registry() -> ToolRegistry:
    """Provide a freshly populated isolated ToolRegistry."""
    registry = ToolRegistry()
    register_default_mocks(registry)
    return registry


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """Provide FastAPI test client."""
    with TestClient(app) as client:
        yield client
