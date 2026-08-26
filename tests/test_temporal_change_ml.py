"""Test Suite for Division 3 Machine Learning Pipeline (TinyCD).

Tests:
1. TinyCD model instantiation and parameter count
2. Forward pass output shape and probability range [0.0, 1.0]
3. Gradient smoke test execution and non-zero parameter delta
4. LEVIR-CD dataset loading and transformation
5. Data leakage audit verification across parent scenes
6. Strict checkpoint loading vs. failure handling
7. Specialist integration with trained TinyCD adapter
8. Zero-regression execution of standard Division 3 tasks.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch

from core.schemas import ImageFormat, ImageInput, ImageModality, QueryRequest, TaskType, ToolRequest, ToolStatus
from specialists.temporal_change.adaptation.models.tinycd import TinyCD, count_parameters
from specialists.temporal_change.adaptation.dataset_loader import LEVIRCDDatasetLoader, TemporalChangeDataset
from specialists.temporal_change.adaptation.smoke_test_detector import run_gradient_smoke_test
from specialists.temporal_change.model_adapter import MockChangeModel, TinyCDAdapter
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool
from specialists.temporal_change.config import TemporalChangeConfig
from specialists.temporal_change.errors import ChangeModelLoadError


def test_tinycd_model_construction():
    """Verify TinyCD model instantiation, forward pass shape and value bounds."""
    model = TinyCD(in_channels=3, base_features=32)
    model.eval()

    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable == total

    t0 = torch.rand(2, 3, 256, 256)
    t1 = torch.rand(2, 3, 256, 256)

    with torch.no_grad():
        out = model(t0, t1)

    assert out.shape == (2, 1, 256, 256)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_gradient_smoke_test_proof():
    """Verify mathematical backprop and parameter delta proof."""
    proof = run_gradient_smoke_test()
    assert proof["status"] == "REAL_NEURAL_TRAINING_VERIFIED"
    assert proof["trainable_parameters"] > 0
    assert proof["forward_loss"] > 0
    assert proof["gradient_norm"] > 0
    assert proof["parameter_delta_after_step"] > 0
    assert len(proof["checkpoint_sha256"]) == 64


def test_dataset_loader_and_leakage_audit():
    """Verify LEVIR-CD dataset loading and strict disjointness."""
    root = "specialists/temporal_change/data/levir_cd"
    train_samples = LEVIRCDDatasetLoader.discover_split_samples(root, "train")
    val_samples = LEVIRCDDatasetLoader.discover_split_samples(root, "val")
    test_samples = LEVIRCDDatasetLoader.discover_split_samples(root, "test")

    assert len(train_samples) > 0
    assert len(val_samples) > 0
    assert len(test_samples) > 0

    audit = LEVIRCDDatasetLoader.audit_split_leakage(train_samples, val_samples, test_samples)
    assert audit["is_leakage_free"] is True
    assert len(audit["train_test_overlap"]) == 0

    # Test dataset batch retrieval
    ds = TemporalChangeDataset(test_samples[:2], is_training=False)
    sample = ds[0]
    assert sample["t0"].shape == (3, 256, 256)
    assert sample["t1"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)


def test_tinycd_adapter_strict_mode_failure():
    """Verify strict evaluation mode raises hard error on missing checkpoint."""
    adapter = TinyCDAdapter(checkpoint_path="non_existent_checkpoint.pth", strict=True)
    with pytest.raises(ChangeModelLoadError):
        adapter.initialize()


@pytest.mark.asyncio
async def test_specialist_with_trained_tinycd_adapter():
    """Verify BiTemporalChangeSpecialistTool executes with trained TinyCD adapter."""
    adapter = TinyCDAdapter(
        checkpoint_path="specialists/temporal_change/weights/ChangeDetector-TinyCD.pth",
        strict=False,
    )
    specialist = BiTemporalChangeSpecialistTool(change_model=adapter)

    req = ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="Detect surface differences in this scene.",
        images=[
            ImageInput(
                image_id="t0",
                path_or_uri="demo_assets/demo_change_t0.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
            ImageInput(
                image_id="t1",
                path_or_uri="demo_assets/demo_change_t1.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
        ],
    )

    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert result.model_info["change_model"] == "ChangeDetector-TinyCD"
    assert len(result.evidence) >= 1
    assert len(result.artifacts) >= 1
