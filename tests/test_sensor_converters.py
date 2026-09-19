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
        is_db=True,
    )

    assert img.size == (120, 120)
    assert img.mode == "RGB"
    assert isinstance(meta, TransformedImageMetadata)
    assert meta.sensor == "Sentinel-1 SAR"
    assert "ratio" in meta.channels_rendered[2]
    assert "Phase information" in meta.spectral_information_loss_doc

    arr = np.array(img)
    assert arr.shape == (120, 120, 3)
    assert arr.dtype == np.uint8
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_sentinel1_ratio_mathematical_identity():
    """Verify that VV_dB - VH_dB mathematically equals 10 * log10(VV_linear / VH_linear)."""
    # Create known linear intensities
    lin_vv = np.full((10, 10), 0.1, dtype=np.float32)   # 0.1 linear = -10.0 dB
    lin_vh = np.full((10, 10), 0.01, dtype=np.float32)  # 0.01 linear = -20.0 dB
    
    # In dB: VV_dB = -10.0 dB, VH_dB = -20.0 dB
    # Ratio in dB = -10.0 - (-20.0) = +10.0 dB
    # Ratio in linear = 0.1 / 0.01 = 10.0 -> 10 * log10(10.0) = 10.0 dB
    db_vv = 10.0 * np.log10(lin_vv)
    db_vh = 10.0 * np.log10(lin_vh)
    
    img_from_db, _ = Sentinel1SARConverter.convert_s1_to_rgb(
        db_vv, db_vh, "rec_db", "scene_db", "patch_db", is_db=True
    )
    img_from_lin, _ = Sentinel1SARConverter.convert_s1_to_rgb(
        lin_vv, lin_vh, "rec_lin", "scene_lin", "patch_lin", is_db=False
    )
    
    arr_db = np.array(img_from_db)
    arr_lin = np.array(img_from_lin)
    
    # Both paths must yield identical pixel representations
    np.testing.assert_array_equal(arr_db, arr_lin)
    
    # Verify Blue channel (ratio) calculation:
    # ratio_db = 10.0 dB
    # normalized = (10.0 - (-5.0)) / (20.0 - (-5.0)) = 15.0 / 25.0 = 0.60
    # uint8 value = round(0.60 * 255.0) = 153
    expected_blue = int(round(0.60 * 255.0))
    assert np.all(arr_db[:, :, 2] == expected_blue)


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
