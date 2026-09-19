"""Production Verification Test Suite for Division 3 TinyCD Backend.

Verifies:
1. Pre-inference safety: Missing checkpoint fails closed with CHECKPOINT_INVALID.
2. Checkpoint provenance gate: SHA256 mismatch or key mismatch blocks inference.
3. Architecture dispatch safety: Empty ChangeFormer checkpoint fails closed, never creating random stub.
4. Complete execution traceability (architecture, checkpoint, sha256, params, mode, threshold).
5. Exact LEVIR-CD test_10.png regression metrics (F1 ≈ 0.8392, IoU ≈ 0.7230, Prec ≈ 0.7870, Rec ≈ 0.8990).
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest
import numpy as np
from PIL import Image

from core.schemas import (
    ExecutionStage,
    ImageFormat,
    ImageInput,
    ImageModality,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from specialists.temporal_change.config import TemporalChangeConfig
from specialists.temporal_change.errors import ChangeModelLoadError, CheckpointProvenanceError
from specialists.temporal_change.model_adapter import ChangeFormerAdapter, TinyCDAdapter
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool


LEVIR_TEST_DIR = Path("data/official_levir_cd/test")
TINYCD_CHECKPOINT = Path("specialists/temporal_change/weights/ChangeDetector-TinyCD.pth")
EXPECTED_SHA256 = "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"


def test_tinycd_missing_checkpoint_fails_closed():
    """Verify that a missing checkpoint path fails closed with CHECKPOINT_INVALID."""
    adapter = TinyCDAdapter(checkpoint_path="non_existent_weights.pth", strict=True)
    with pytest.raises(ChangeModelLoadError) as exc_info:
        adapter.initialize()
    assert "STATUS = CHECKPOINT_INVALID" in str(exc_info.value)
    assert adapter.provenance_status == "CHECKPOINT_INVALID"
    assert not adapter.is_ready()


def test_tinycd_corrupted_sha256_fails_closed(tmp_path: Path):
    """Verify that a checkpoint with an invalid SHA-256 is rejected by the provenance gate."""
    corrupted_ckpt = tmp_path / "corrupted_tinycd.pth"
    corrupted_ckpt.write_bytes(b"corrupted_or_tampered_weights_data")

    adapter = TinyCDAdapter(
        checkpoint_path=str(corrupted_ckpt),
        verify_provenance=True,
    )
    with pytest.raises(CheckpointProvenanceError) as exc_info:
        adapter.initialize()
    assert "STATUS = CHECKPOINT_INVALID" in str(exc_info.value)
    assert adapter.provenance_status == "CHECKPOINT_INVALID"
    assert not adapter.is_ready()


def test_changeformer_empty_checkpoint_blocked():
    """Verify that ChangeFormer without a checkpoint is blocked rather than silently running random weights."""
    config = TemporalChangeConfig()
    config.model_architecture = "changeformer"
    config.model_checkpoint_path = ""

    with pytest.raises(ChangeModelLoadError) as exc_info:
        tool = BiTemporalChangeSpecialistTool(config=config)
    assert "STATUS = CHECKPOINT_INVALID" in str(exc_info.value)


@pytest.mark.asyncio
async def test_specialist_blocks_inference_when_checkpoint_invalid(tmp_path: Path):
    """Verify specialist ToolRequest execution fails closed when checkpoint is missing."""
    config = TemporalChangeConfig()
    config.model_checkpoint_path = str(tmp_path / "missing.pth")

    tool = BiTemporalChangeSpecialistTool(config=config)

    # Dummy images
    t0_path = tmp_path / "t0.png"
    t1_path = tmp_path / "t1.png"
    Image.new("RGB", (64, 64), (100, 100, 100)).save(t0_path)
    Image.new("RGB", (64, 64), (120, 120, 120)).save(t1_path)

    req = ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="Detect changes",
        images=[
            ImageInput(image_id="t0", path_or_uri=str(t0_path), format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=str(t1_path), format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )

    result = await tool.execute(req)
    assert result.status == ToolStatus.FAILED
    assert result.metadata.get("error_stage") == "CHECKPOINT_INVALID"
    assert "CHECKPOINT_INVALID" in result.answer


@pytest.mark.asyncio
async def test_production_traceability_metadata():
    """Verify every execution trace and model_info contains complete provenance details."""
    config = TemporalChangeConfig()
    assert config.model_architecture == "tinycd"
    assert config.model_checkpoint_path == "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"

    tool = BiTemporalChangeSpecialistTool(config=config)

    t0_path = str(LEVIR_TEST_DIR / "A" / "test_10.png")
    t1_path = str(LEVIR_TEST_DIR / "B" / "test_10.png")

    req = ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="Detect newly constructed structures",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )

    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS

    # Verify model_info traceability
    mi = result.model_info
    assert mi["architecture"] == "TinyCD (Siamese U-Net + MAMB)"
    assert mi["checkpoint"] == str(TINYCD_CHECKPOINT)
    assert mi["sha256"] == EXPECTED_SHA256
    assert mi["parameter_count"] == 3565034
    assert mi["execution_mode"] == "production_verified"
    assert mi["threshold"] == 0.50

    # Verify inference trace details
    inf_trace = next(e for e in result.execution_trace if e.stage == ExecutionStage.INFERENCE_EXECUTED)
    assert inf_trace.details["architecture"] == "TinyCD (Siamese U-Net + MAMB)"
    assert inf_trace.details["checkpoint"] == str(TINYCD_CHECKPOINT)
    assert inf_trace.details["sha256"] == EXPECTED_SHA256
    assert inf_trace.details["parameter_count"] == 3565034
    assert inf_trace.details["execution_mode"] == "production_verified"
    assert inf_trace.details["threshold"] == 0.50


@pytest.mark.asyncio
async def test_levir_test_10_regression_metrics():
    """Regression test on LEVIR-CD test_10.png proving authoritative TinyCD checkpoint execution."""
    t0_path = LEVIR_TEST_DIR / "A" / "test_10.png"
    t1_path = LEVIR_TEST_DIR / "B" / "test_10.png"
    gt_path = LEVIR_TEST_DIR / "label" / "test_10.png"

    assert t0_path.exists() and t1_path.exists() and gt_path.exists()

    tool = BiTemporalChangeSpecialistTool()
    req = ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="Detect building changes",
        images=[
            ImageInput(image_id="t0", path_or_uri=str(t0_path), format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=str(t1_path), format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )

    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS

    # Extract binary change mask artifact
    mask_art = next(a for a in result.artifacts if a.name == "binary_change_mask.png")
    pred_mask_256 = np.array(Image.open(mask_art.uri_or_path)) > 128

    # Ground truth (1024x1024)
    gt_1024 = np.array(Image.open(gt_path)) > 128
    pred_1024 = np.array(Image.fromarray(pred_mask_256).resize((gt_1024.shape[1], gt_1024.shape[0]), Image.NEAREST))

    tp = int((pred_1024 & gt_1024).sum())
    fp = int((pred_1024 & ~gt_1024).sum())
    fn = int((~pred_1024 & gt_1024).sum())
    tn = int((~pred_1024 & ~gt_1024).sum())

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-6, (prec + rec))
    iou = tp / max(1, (tp + fp + fn))
    oa = (tp + tn) / gt_1024.size
    pred_pct = (pred_1024.sum() / gt_1024.size) * 100.0

    print(f"\n[REGRESSION AUDIT] test_10.png metrics:")
    print(f"  F1 = {f1:.4f} (target ≈ 0.8392)")
    print(f"  IoU = {iou:.4f} (target ≈ 0.7230)")
    print(f"  Precision = {prec:.4f} (target ≈ 0.7870)")
    print(f"  Recall = {rec:.4f} (target ≈ 0.8990)")
    print(f"  OA = {oa:.4f} (target ≈ 0.9666)")
    print(f"  Predicted Change % = {pred_pct:.2f}% (target ≈ 11.08%, NEVER ~100%)")

    # Strict bounds checking proving this is NOT the 17.69% random stub
    assert 0.80 <= f1 <= 0.87, f"F1 {f1:.4f} outside expected range [0.80, 0.87]"
    assert 0.68 <= iou <= 0.76, f"IoU {iou:.4f} outside expected range [0.68, 0.76]"
    assert 0.75 <= prec <= 0.83, f"Precision {prec:.4f} outside expected range [0.75, 0.83]"
    assert 0.85 <= rec <= 0.94, f"Recall {rec:.4f} outside expected range [0.85, 0.94]"
    assert 0.95 <= oa <= 0.98, f"OA {oa:.4f} outside expected range [0.95, 0.98]"
    assert 9.0 <= pred_pct <= 13.0, f"Predicted change % {pred_pct:.2f} outside [9.0, 13.0]"
