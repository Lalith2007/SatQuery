"""Unit tests for deterministic sensor converters."""

import numpy as np
import pytest
from specialists.single_image.adaptation.qwen25vl.sensor_converters import (
    Sentinel1SARConverter,
    Sentinel2MultispectralConverter,
    TransformedImageMetadata,
)


def test_sentinel1_converter():
    vv = np.random.uniform(-25.0, 0.0, (120, 120)).astype(np.float32)
    vh = np.random.uniform(-32.0, -5.0, (120, 120)).astype(np.float32)

    img, meta = Sentinel1SARConverter.convert_s1_to_rgb(
        vv, vh,
        record_id="rec_001",
        scene_id="S1_scene_01",
        patch_id="S1_patch_01",
        geolocation={"country": "Finland", "lat": 60.1, "lon": 24.9},
    )

    assert img.size == (120, 120)
    assert img.mode == "RGB"
    assert isinstance(meta, TransformedImageMetadata)
    assert meta.sensor == "Sentinel-1 SAR"
    assert "log(VV/VH)" in meta.channels_rendered[2]
    assert "Phase information" in meta.spectral_information_loss_doc


def test_sentinel2_converter():
    band_dict = {
        f"B{i:02d}": np.random.uniform(100.0, 2500.0, (120, 120)).astype(np.float32)
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    }
    band_dict["B8A"] = np.random.uniform(100.0, 2500.0, (120, 120)).astype(np.float32)

    img, meta = Sentinel2MultispectralConverter.convert_s2_to_rgb(
        band_dict,
        record_id="rec_002",
        scene_id="S2_scene_01",
        patch_id="S2_patch_01",
        geolocation={"country": "Austria", "lat": 48.2, "lon": 16.3},
    )

    assert img.size == (120, 120)
    assert img.mode == "RGB"
    assert isinstance(meta, TransformedImageMetadata)
    assert meta.sensor == "Sentinel-2 MSI"
    assert "B04 (Red)" in meta.channels_rendered[0]
    assert "SWIR1 B11" in meta.spectral_information_loss_doc
