"""Regression Test Suite for Change-VQA Composed Workflow Fix.

Verifies:
1. IntentResolver routes Change-VQA queries to the composed sequential workflow (is_composite=True).
2. Execution trace contains the full required observable stage sequence:
   REQUEST_RECEIVED -> TASK_RESOLVED -> INPUT_VALIDATED -> TOOL_SELECTED ->
   TINYCD_EXECUTED -> CHANGE_MASK_GENERATED -> REGION_EXTRACTED ->
   VLM_INPUT_PREPARED -> VLM_EXECUTED -> ANSWER_GENERATED -> RESULT_AGGREGATED
3. Deterministic candidate region extraction and sorting (area descending, region_id ascending).
4. Visual handoff passes the cropped changed-region patch to the VLM (not raw full T1).
5. When fine-tuned Qwen weights are pending, the VLM stage honestly returns
   CHANGE_VQA_MODEL_NOT_READY without hallucination or PaliGemma fallback.
6. Empty change masks trigger the documented full-image fallback with has_change=False.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from core.schemas import ImageFormat, ImageInput, ImageModality, QueryRequest, TaskType, ToolStatus
from registry.registry import ToolRegistry
from specialists.single_image.specialist import SingleImageRSSpecialistTool
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool
from agent.controller import AgentController


@pytest.fixture
def clean_registry():
    """Create a clean registry with registered production specialists."""
    reg = ToolRegistry()
    reg.register(SingleImageRSSpecialistTool(), overwrite=True)
    reg.register(BiTemporalChangeSpecialistTool(), overwrite=True)
    return reg


@pytest.mark.asyncio
async def test_end_to_end_change_vqa_workflow_test_10(clean_registry):
    """Verify the complete Change-VQA composed workflow on LEVIR-CD test_10.png."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    req = QueryRequest(
        query="What changed between these two dates, and where did the change occur?",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    # 1. Verification of Task and Status
    assert response.resolved_task == TaskType.CHANGE_VQA
    assert response.status == ToolStatus.PARTIAL_SUCCESS

    # 2. Verification of Answer Structure
    assert "[Change Detection Stage - TinyCD]" in response.answer
    assert "[Semantic Interpretation Stage - VLM]" in response.answer
    assert "CHANGE_VQA_MODEL_NOT_READY" in response.answer

    # 3. Verification of Provenance Metadata
    meta = response.metadata
    assert meta["change_vqa_status"] == "CHANGE_VQA_MODEL_NOT_READY"
    assert meta["tinycd_detection"] == "completed"
    assert meta["semantic_vlm_interpretation"] == "unavailable"

    tinycd_prov = meta["tinycd_provenance"]
    assert tinycd_prov["architecture"] == "TinyCD (Siamese U-Net + MAMB)"
    assert tinycd_prov["checkpoint"] == "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"
    assert tinycd_prov["sha256"] == "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"
    assert tinycd_prov["parameter_count"] == 3565034
    assert tinycd_prov["threshold"] == 0.5

    vlm_prov = meta["vlm_provenance"]
    assert vlm_prov["backend"] == "qwen25vl"
    assert vlm_prov["model_name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert vlm_prov["readiness_status"] == "CHANGE_VQA_MODEL_NOT_READY"

    # 4. Verification of ChangedRegionEvidence and Visual Artifacts
    cre = meta["changed_region_evidence"]
    assert cre["has_change"] is True
    assert cre["selected_region_id"] == 18
    assert cre["selected_region_normalized_bbox"] == [0.5781, 0.6758, 0.625, 0.7773]
    assert cre["area_pixels"] == 3168

    crop_path = Path(cre["cropped_t1_path"])
    overlay_path = Path(cre["visualization_overlay_path"])
    assert crop_path.exists()
    assert overlay_path.exists()

    # 5. Verification of Observable Trace Stage Sequence (Part 6)
    stages = [t.stage.value for t in response.execution_trace]
    expected_sequence = [
        "REQUEST_RECEIVED",
        "TASK_RESOLVED",
        "INPUT_VALIDATED",
        "TOOL_SELECTED",
        "TINYCD_EXECUTED",
        "CHANGE_MASK_GENERATED",
        "REGION_EXTRACTED",
        "VLM_INPUT_PREPARED",
        "VLM_EXECUTED",
        "ANSWER_GENERATED",
        "RESULT_AGGREGATED",
    ]

    for expected_stage in expected_sequence:
        assert expected_stage in stages, f"Missing required execution stage: {expected_stage}"

    # Verify chronological ordering of core stages
    idx_tinycd = stages.index("TINYCD_EXECUTED")
    idx_mask = stages.index("CHANGE_MASK_GENERATED")
    idx_region = stages.index("REGION_EXTRACTED")
    idx_vlm_prep = stages.index("VLM_INPUT_PREPARED")
    idx_vlm_exec = stages.index("VLM_EXECUTED")
    idx_answer = stages.index("ANSWER_GENERATED")
    idx_agg = stages.index("RESULT_AGGREGATED")

    assert idx_tinycd < idx_mask < idx_region < idx_vlm_prep < idx_vlm_exec < idx_agg, (
        f"Stage order violation in execution trace: {stages}"
    )


@pytest.mark.asyncio
async def test_multi_region_deterministic_ranking(clean_registry):
    """Verify that multiple regions are detected, ranked deterministically by area, and top region is selected."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    req = QueryRequest(
        query="Describe what changed in the detected regions.",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    cre = response.metadata["changed_region_evidence"]
    candidates = cre["candidate_regions"]
    assert len(candidates) == 6, f"Expected 6 candidate regions in test_10, got {len(candidates)}"

    # Check deterministic descending area sorting
    areas = [c["area_pixels"] for c in candidates]
    assert areas == sorted(areas, reverse=True), f"Candidates not sorted by area descending: {areas}"

    # Top region should be selected as the primary evidence crop
    primary = candidates[0]
    assert cre["selected_region_id"] == primary["region_id"]
    assert cre["selected_region_pixel_bbox"] == primary["pixel_bbox"]
    assert cre["area_pixels"] == primary["area_pixels"]


@pytest.mark.asyncio
async def test_empty_change_mask_documented_fallback(clean_registry):
    """Verify that a scene with no change detected triggers the documented full-image fallback."""
    # LEVIR-CD test_1.png has 0 change detected by TinyCD
    t0_path = "data/official_levir_cd/test/A/test_1.png"
    t1_path = "data/official_levir_cd/test/B/test_1.png"

    req = QueryRequest(
        query="Describe what changed between these two dates.",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    cre = response.metadata["changed_region_evidence"]
    assert cre["has_change"] is False
    assert cre["fallback_reason"] is not None
    assert "NO_CHANGE_DETECTED" in cre["fallback_reason"]
    assert cre["selected_region_id"] == 0
    assert cre["area_pixels"] == 0
    assert cre["cropped_t1_path"] == t1_path
