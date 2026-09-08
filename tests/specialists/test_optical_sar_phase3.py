"""Unit tests for Phase 3 of Division 4 Optical-SAR Specialist.

Verifies evidence generation, bounding box derivation, visual overlays,
confidence/uncertainty maps, health check, and end-to-end specialist service execution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from core.schemas import ImageInput, ImageModality, TaskType, ToolRequest, ToolStatus
from specialists.optical_sar.confidence import ConfidenceEngine
from specialists.optical_sar.evidence import SpatialEvidenceEngine
from specialists.optical_sar.service import OpticalSarSpecialist


def test_spatial_evidence_engine(tmp_path: Path) -> None:
    """Test SpatialEvidenceEngine generates PNG masks, bounding boxes, overlay artifacts, and Evidence items."""
    engine = SpatialEvidenceEngine(tmp_path)

    probs = np.zeros((4, 100, 100), dtype=np.float32)
    probs[0, 20:50, 20:50] = 0.9  # Built-up square
    probs[1, 60:80, 60:80] = 0.85 # Water square
    probs[3] = 0.1

    opt_rgb = np.random.rand(3, 100, 100).astype(np.float32)
    sar_intensity = np.random.rand(2, 100, 100).astype(np.float32)
    active_intents = {"built_up": 1.0, "water": 1.0, "vegetation": 0.1, "background": 0.1}

    evidences, artifacts, metrics = engine.generate_evidence_artifacts(
        probs=probs,
        optical_rgb=opt_rgb,
        sar_intensity=sar_intensity,
        active_intents=active_intents,
        request_id="test_req_001",
    )

    assert len(evidences) >= 3
    assert len(artifacts) >= 3
    assert "built_up_pixel_count" in metrics
    assert "water_pixel_count" in metrics

    # Verify generated artifact files exist on disk
    for art in artifacts:
        assert Path(art.uri_or_path).exists()


def test_confidence_engine(tmp_path: Path) -> None:
    """Test ConfidenceEngine calculates pixel margins, normalized entropy, global confidence scalar, and heatmap."""
    engine = ConfidenceEngine(tmp_path)

    probs = np.zeros((4, 64, 64), dtype=np.float32)
    probs[0] = 0.95
    probs[1] = 0.03
    probs[2] = 0.01
    probs[3] = 0.01

    global_conf, conf_map, artifact = engine.calculate_confidence(probs, request_id="test_req_002")

    assert 0.0 <= global_conf <= 1.0
    assert global_conf > 0.70
    assert conf_map.shape == (64, 64)
    assert Path(artifact.uri_or_path).exists()


@pytest.mark.asyncio
async def test_optical_sar_specialist_execution(tmp_path: Path) -> None:
    """Test end-to-end OpticalSarSpecialist execution on synthetic ToolRequest."""
    opt_path = tmp_path / "opt.png"
    sar_path = tmp_path / "sar.png"

    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8)).save(opt_path)
    Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8)).save(sar_path)

    request = ToolRequest(
        request_id="req-test-e2e-001",
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Use optical and SAR images together to identify built-up and water-covered regions.",
        images=[
            ImageInput(path_or_uri=str(opt_path), format="png", modality=ImageModality.OPTICAL),
            ImageInput(path_or_uri=str(sar_path), format="png", modality=ImageModality.SAR),
        ],
    )

    specialist = OpticalSarSpecialist()
    assert specialist.health_check() is True

    result = await specialist.execute(request)

    assert result.request_id == "req-test-e2e-001"
    assert result.status == ToolStatus.SUCCESS
    assert "built-up" in result.answer.lower()
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.evidence) > 0
    assert len(result.artifacts) > 0
    assert len(result.execution_trace) >= 4
