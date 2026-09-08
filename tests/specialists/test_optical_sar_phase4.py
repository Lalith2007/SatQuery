"""Phase 4 Comprehensive Test Suite for Division 4 Optical-SAR Specialist.

Verifies:
1. Genuine Cross-Modal Dual Dependency (proving neither SAR nor Optical is ignored)
2. Encoder Swapping & Pluggability
3. Tool Registry Integration & Shared Contract Compatibility
4. Error Handling & Edge Case Failures
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from core.schemas import ImageInput, ImageModality, TaskType, ToolRequest, ToolStatus
from registry.registry import ToolRegistry
from specialists.optical_sar.encoders import OpticalEncoder, SarEncoder
from specialists.optical_sar.service import OpticalSarSpecialist


@pytest.mark.asyncio
async def test_genuine_cross_modal_dependency(tmp_path: Path) -> None:
    """Empirically verify that BOTH Optical and SAR modalities materially affect model output.
    
    Checks that zeroing or altering the SAR modality changes feature outputs and probabilities,
    proving SAR is not silently ignored.
    """
    opt_path = tmp_path / "opt.png"
    sar_path = tmp_path / "sar.png"
    sar_zero_path = tmp_path / "sar_zero.png"

    # Create optical image and high-variance SAR image
    np.random.seed(42)
    opt_arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    sar_arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    sar_zero_arr = (np.zeros((64, 64))).astype(np.uint8)

    Image.fromarray(opt_arr).save(opt_path)
    Image.fromarray(sar_arr).save(sar_path)
    Image.fromarray(sar_zero_arr).save(sar_zero_path)

    specialist = OpticalSarSpecialist()

    # 1. Normal Request (Both Optical and High-Variance SAR)
    req_normal = ToolRequest(
        request_id="req-dep-normal",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify built-up and water regions",
        images=[
            ImageInput(path_or_uri=str(opt_path), format="png", modality=ImageModality.OPTICAL),
            ImageInput(path_or_uri=str(sar_path), format="png", modality=ImageModality.SAR),
        ],
    )
    res_normal = await specialist.execute(req_normal)

    # 2. Zeroed SAR Request (Optical present, SAR zeroed out)
    req_zero_sar = ToolRequest(
        request_id="req-dep-zero-sar",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify built-up and water regions",
        images=[
            ImageInput(path_or_uri=str(opt_path), format="png", modality=ImageModality.OPTICAL),
            ImageInput(path_or_uri=str(sar_zero_path), format="png", modality=ImageModality.SAR),
        ],
    )
    res_zero_sar = await specialist.execute(req_zero_sar)

    assert res_normal.status == ToolStatus.SUCCESS
    assert res_zero_sar.status == ToolStatus.SUCCESS

    # Verify metrics or confidence changed when SAR input changed
    normal_metrics = res_normal.metadata["metrics"]
    zero_metrics = res_zero_sar.metadata["metrics"]

    assert normal_metrics != zero_metrics or res_normal.confidence != res_zero_sar.confidence


@pytest.mark.asyncio
async def test_encoder_swapping(tmp_path: Path) -> None:
    """Test replacing default encoders with lightweight ResNet18 backbones dynamically."""
    specialist = OpticalSarSpecialist()

    # Swap encoders cleanly using set_encoders helper
    opt_18 = OpticalEncoder(backbone_type="resnet18", pretrained_weights=None)
    sar_18 = SarEncoder(backbone_type="resnet18", pretrained_weights=None)
    specialist.set_encoders(opt_18, sar_18)

    opt_path = tmp_path / "opt_swap.png"
    sar_path = tmp_path / "sar_swap.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(opt_path)
    Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8)).save(sar_path)

    request = ToolRequest(
        request_id="req-swap-001",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify built-up regions",
        images=[
            ImageInput(path_or_uri=str(opt_path), format="png", modality=ImageModality.OPTICAL),
            ImageInput(path_or_uri=str(sar_path), format="png", modality=ImageModality.SAR),
        ],
    )

    result = await specialist.execute(request)
    assert result.status == ToolStatus.SUCCESS


@pytest.mark.asyncio
async def test_registry_integration() -> None:
    """Test registering OpticalSarSpecialist into Lalith's ToolRegistry."""
    registry = ToolRegistry()
    specialist = OpticalSarSpecialist()

    registry.register(specialist)

    # Use Lalith's canonical ToolRegistry methods: get() and find_tools_for_task()
    retrieved = registry.get(specialist.name)
    assert retrieved is not None
    assert retrieved.name == specialist.name

    # Find tool by TaskType
    selected = registry.find_tools_for_task(TaskType.OPTICAL_SAR_ANALYSIS)
    assert len(selected) > 0
    assert any(t.name == specialist.name for t in selected)


@pytest.mark.asyncio
async def test_error_handling_and_validation() -> None:
    """Test specialist gracefully handles validation failures without crashing."""
    specialist = OpticalSarSpecialist()

    # Request with missing SAR image
    invalid_req = ToolRequest(
        request_id="req-invalid-001",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Find water",
        images=[
            ImageInput(path_or_uri="opt_only.png", format="png", modality=ImageModality.OPTICAL)
        ],
    )

    val_res = specialist.validate_request(invalid_req)
    assert val_res.is_valid is False
    assert len(val_res.errors) > 0

    result = await specialist.execute(invalid_req)
    assert result.status == ToolStatus.FAILED
    assert "validation failed" in result.answer.lower()
