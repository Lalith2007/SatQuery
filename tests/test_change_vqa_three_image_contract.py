"""Comprehensive Regression Suite for Change-VQA Three-Image Application Contract.

Tests A through M:
A. Standard VQA with 1 image succeeds.
B. Change-VQA with exactly 3 images succeeds.
C. Change-VQA with 2 images -> raises ValueError (rejected).
D. Change-VQA with 4 images -> raises ValueError (rejected).
E. Normal single-image task cannot accidentally receive 3 images -> raises ValueError.
F. Image ordering [T0, T1, overlay] is strictly preserved in inference calls.
G. Roles [BEFORE, AFTER, WHERE_CHANGE_OCCURRED] are preserved.
H. Original query is preserved in wrapper and inference calls.
I. Change metadata (bbox, area, change ratio, temporal order) is preserved.
J. T1-only fallback is executed ONLY when explicitly requested (vlm_contract == 'single_image_t1').
K. No ground-truth label access during execution.
L. No random weights or PaliGemma fallback when Qwen backend is selected.
M. MODEL_NOT_READY returned honestly when real Qwen weights are absent.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
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
from specialists.single_image.specialist import SingleImageRSSpecialistTool
from specialists.single_image.adaptation.qwen25vl.inference import QwenSingleImageEngine, QwenInferenceMetrics


@pytest.fixture
def dummy_image_files(tmp_path):
    """Create three test rasters for BEFORE (T0), AFTER (T1), and OVERLAY."""
    t0_path = tmp_path / "crop_t0.png"
    t1_path = tmp_path / "crop_t1.png"
    overlay_path = tmp_path / "crop_overlay.png"

    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(t0_path)
    img.save(t1_path)
    img.save(overlay_path)

    return str(t0_path), str(t1_path), str(overlay_path)


@pytest.mark.asyncio
async def test_a_standard_vqa_single_image(dummy_image_files):
    """Test A: Standard VQA with exactly 1 image succeeds."""
    t0, _, _ = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    request = ToolRequest(
        request_id="test_a",
        task=TaskType.SINGLE_IMAGE_VQA,
        images=[
            ImageInput(
                image_id="img1",
                path_or_uri=t0,
                format=ImageFormat.PNG,
                modality=ImageModality.OPTICAL,
            )
        ],
        query="What is the dominant land cover?",
    )

    result = await tool.execute(request)
    assert result.status == ToolStatus.SUCCESS
    assert result.parameters["image_id"] == "img1"


@pytest.mark.asyncio
async def test_b_change_vqa_three_images(dummy_image_files):
    """Test B: Change-VQA with exactly 3 images succeeds."""
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(
            image_id="t0",
            path_or_uri=t0,
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"},
        ),
        ImageInput(
            image_id="t1",
            path_or_uri=t1,
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"},
        ),
        ImageInput(
            image_id="overlay",
            path_or_uri=overlay,
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            metadata={"is_changed_region_crop": True, "temporal_role": "WHERE_CHANGE_OCCURRED"},
        ),
    ]

    request = ToolRequest(
        request_id="test_b",
        task=TaskType.CHANGE_VQA,
        images=images,
        query="Describe what changed in this region.",
        context={"changed_region_evidence": {"area_pixels": 450, "selected_region_id": 1}},
    )

    result = await tool.execute(request)
    assert result.status == ToolStatus.PARTIAL_SUCCESS
    assert result.metadata.get("change_vqa_status") == "CHANGE_VQA_MODEL_NOT_READY"
    assert result.metadata.get("handoff_mode") == "MULTI_IMAGE_CHANGE_VQA"
    assert len(result.metadata["evidence_package"]) == 3


@pytest.mark.asyncio
async def test_c_change_vqa_two_images_rejected(dummy_image_files):
    """Test C: Change-VQA with 2 images -> rejected with ToolStatus.FAILED & engine ValueError."""
    t0, t1, _ = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(
            image_id="t0",
            path_or_uri=t0,
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"},
        ),
        ImageInput(
            image_id="t1",
            path_or_uri=t1,
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"},
        ),
    ]

    request = ToolRequest(
        request_id="test_c",
        task=TaskType.CHANGE_VQA,
        images=images,
        query="What changed?",
    )

    result = await tool.execute(request)
    assert result.status == ToolStatus.FAILED
    assert "requires exactly 3 images" in result.answer

    with pytest.raises(ValueError, match="requires exactly 3 evidence images"):
        tool.qwen_engine.run_change_vqa(images=[t0, t1], query="What changed?")


@pytest.mark.asyncio
async def test_d_change_vqa_four_images_rejected(dummy_image_files):
    """Test D: Change-VQA with 4 images -> rejected with ToolStatus.FAILED & engine ValueError."""
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(image_id="t0", path_or_uri=t0, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"}),
        ImageInput(image_id="t1", path_or_uri=t1, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"}),
        ImageInput(image_id="overlay", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "WHERE_CHANGE_OCCURRED"}),
        ImageInput(image_id="extra", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True}),
    ]

    request = ToolRequest(
        request_id="test_d",
        task=TaskType.CHANGE_VQA,
        images=images,
        query="What changed?",
    )

    result = await tool.execute(request)
    assert result.status == ToolStatus.FAILED
    assert "requires exactly 3 images" in result.answer

    with pytest.raises(ValueError, match="requires exactly 3 evidence images"):
        tool.qwen_engine.run_change_vqa(images=[t0, t1, overlay, overlay], query="What changed?")


@pytest.mark.asyncio
async def test_e_single_image_task_three_images_rejected(dummy_image_files):
    """Test E: Normal single-image task cannot accidentally receive 3 images."""
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(image_id="img1", path_or_uri=t0, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ImageInput(image_id="img2", path_or_uri=t1, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ImageInput(image_id="img3", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
    ]

    request = ToolRequest(
        request_id="test_e",
        task=TaskType.SINGLE_IMAGE_VQA,
        images=images,
        query="What is here?",
    )

    result = await tool.execute(request)
    assert result.status == ToolStatus.FAILED
    assert "requires exactly 1 image" in result.answer

    with pytest.raises(ValueError, match="requires exactly 1 image"):
        tool.qwen_engine.run_vqa(image=[t0, t1, overlay], query="What is here?")


@pytest.mark.asyncio
async def test_f_g_h_i_multi_image_inference_interface_validation(dummy_image_files):
    """Tests F, G, H, I: MULTI-IMAGE INFERENCE INTERFACE VALIDATION.

    Verifies that the wrapper sends:
    - 3 ImageInputs in preserved order [T0, T1, overlay]
    - Preserved roles [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]
    - Original user query
    - Change metadata
    """
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(image_id="t0", path_or_uri=t0, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"}),
        ImageInput(image_id="t1", path_or_uri=t1, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"}),
        ImageInput(image_id="overlay", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "WHERE_CHANGE_OCCURRED"}),
    ]

    change_meta = {
        "selected_region_id": 42,
        "selected_region_pixel_bbox": [10, 20, 50, 60],
        "area_pixels": 1200,
        "area_fraction": 0.05,
    }

    test_query = "Describe the newly constructed warehouse infrastructure in detail."

    request = ToolRequest(
        request_id="test_fghi",
        task=TaskType.CHANGE_VQA,
        images=images,
        query=test_query,
        context={"changed_region_evidence": change_meta},
    )

    mock_metrics = QwenInferenceMetrics()
    mock_metrics.device_used = "cpu"
    mock_metrics.inference_time_ms = 12.5

    tool.qwen_engine._is_real_weights_loaded = True
    try:
        with patch.object(
            tool.qwen_engine,
            "run_change_vqa",
            return_value=("Verified 3-image semantic interpretation", 0.95, mock_metrics),
        ) as mock_run:
            result = await tool.execute(request)

            assert mock_run.called, "run_change_vqa MUST be called on the Qwen engine!"
            call_kwargs = mock_run.call_args.kwargs

            # Test F: Image order preserved [T0, T1, overlay]
            assert call_kwargs["images"] == [t0, t1, overlay], "Image order must be [T0, T1, overlay]"

            # Test G: Roles preserved
            assert call_kwargs["roles"] == ["BEFORE", "AFTER", "WHERE_CHANGE_OCCURRED"]

            # Test H: Query preserved
            assert call_kwargs["query"] == test_query

            # Test I: Metadata preserved
            assert call_kwargs["metadata"] == change_meta

            # Output check
            assert result.status == ToolStatus.SUCCESS
            assert result.answer == "Verified 3-image semantic interpretation"
            assert result.confidence == 0.95
    finally:
        tool.qwen_engine._is_real_weights_loaded = False


@pytest.mark.asyncio
async def test_j_t1_only_fallback_explicit_request(dummy_image_files):
    """Test J: T1-only fallback occurs ONLY when explicitly requested."""
    _, t1, _ = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    # Explicit single_image_t1 contract with 1 image
    request_fallback = ToolRequest(
        request_id="test_j_t1",
        task=TaskType.CHANGE_VQA,
        images=[
            ImageInput(
                image_id="t1",
                path_or_uri=t1,
                format=ImageFormat.PNG,
                modality=ImageModality.OPTICAL,
                metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"},
            )
        ],
        query="What is in this post-change region?",
        metadata={"vlm_contract": "single_image_t1"},
    )

    result_fallback = await tool.execute(request_fallback)
    assert result_fallback.status == ToolStatus.PARTIAL_SUCCESS
    assert result_fallback.metadata.get("handoff_mode") == "SINGLE_IMAGE_T1_FALLBACK"

    # In default mode (without explicit flag), 1 image to Change-VQA is rejected!
    request_invalid = ToolRequest(
        request_id="test_j_default_invalid",
        task=TaskType.CHANGE_VQA,
        images=[
            ImageInput(
                image_id="t1",
                path_or_uri=t1,
                format=ImageFormat.PNG,
                modality=ImageModality.OPTICAL,
                metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"},
            )
        ],
        query="What is in this post-change region?",
    )

    result_invalid = await tool.execute(request_invalid)
    assert result_invalid.status == ToolStatus.FAILED
    assert "requires exactly 3 images" in result_invalid.answer


@pytest.mark.asyncio
async def test_k_no_ground_truth_label_access(dummy_image_files):
    """Test K: Verify that ground-truth labels are never accessed during Change-VQA."""
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    accessed_paths = []
    original_open = Path.open

    def instrumented_open(self, *args, **kwargs):
        accessed_paths.append(str(self))
        return original_open(self, *args, **kwargs)

    images = [
        ImageInput(image_id="t0", path_or_uri=t0, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"}),
        ImageInput(image_id="t1", path_or_uri=t1, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"}),
        ImageInput(image_id="overlay", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "WHERE_CHANGE_OCCURRED"}),
    ]

    request = ToolRequest(
        request_id="test_k",
        task=TaskType.CHANGE_VQA,
        images=images,
        query="What changed?",
    )

    with patch.object(Path, "open", instrumented_open):
        await tool.execute(request)

    label_accesses = [p for p in accessed_paths if "label" in p.lower()]
    assert len(label_accesses) == 0, f"Detected leaked label access: {label_accesses}"


@pytest.mark.asyncio
async def test_l_m_model_not_ready_and_no_paligemma_fallback(dummy_image_files):
    """Tests L & M: Returns MODEL_NOT_READY when Qwen absent; no random or PaliGemma fallback."""
    t0, t1, overlay = dummy_image_files
    tool = SingleImageRSSpecialistTool(backend="qwen25vl")

    images = [
        ImageInput(image_id="t0", path_or_uri=t0, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "BEFORE"}),
        ImageInput(image_id="t1", path_or_uri=t1, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "AFTER"}),
        ImageInput(image_id="overlay", path_or_uri=overlay, format=ImageFormat.PNG, modality=ImageModality.OPTICAL, metadata={"is_changed_region_crop": True, "temporal_role": "WHERE_CHANGE_OCCURRED"}),
    ]

    request = ToolRequest(
        request_id="test_lm",
        task=TaskType.CHANGE_VQA,
        images=images,
        query="Describe change.",
    )

    with patch.object(tool.paligemma_engine, "run_vqa") as mock_pali:
        result = await tool.execute(request)
        # Test L: PaliGemma is NOT called
        assert not mock_pali.called, "PaliGemma must NEVER be called in Qwen backend Change-VQA!"

    # Test M: Returns MODEL_NOT_READY honestly
    assert result.status == ToolStatus.PARTIAL_SUCCESS
    assert "[CHANGE_VQA_MODEL_NOT_READY]" in result.answer
    assert result.confidence is None
    assert result.metadata["change_vqa_status"] == "CHANGE_VQA_MODEL_NOT_READY"
    assert result.model_info["readiness_status"] == "CHANGE_VQA_MODEL_NOT_READY"
    assert result.model_info["is_mock"] is False
    assert result.model_info["is_fallback"] is False
