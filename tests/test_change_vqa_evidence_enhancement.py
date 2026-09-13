"""Regression Test Suite for Change-VQA Evidence Package Enhancement.

Verifies:
1. Cropped T0 and T1 regions correspond to identical spatial coordinates and pixel dimensions.
2. The change overlay corresponds to the identical spatial coordinates as T0 and T1 crops.
3. Region localization is strictly determined by TinyCD model predictions with zero external leakage.
4. Ground-truth benchmark labels ('test/label/') are never opened, read, or referenced.
5. VLM receives the complete 5-item evidence package:
   - Cropped T0 region (BEFORE)
   - Cropped T1 region (AFTER)
   - Change-mask / overlay for that same region (WHERE CHANGE OCCURRED)
   - Original user query
   - Deterministic change metadata
6. Trace contains VLM_INPUT_PREPARED with the evidence artifact IDs and temporal roles.
7. Preserves the existing T1-only path as a documented fallback when single-image contract is enforced.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest

from core.schemas import (
    ExecutionStage,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    TaskType,
    ToolStatus,
)
from registry.registry import ToolRegistry
from specialists.single_image.specialist import SingleImageRSSpecialistTool
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool
from specialists.temporal_change.region_extraction import extract_changed_region_evidence
from specialists.temporal_change.interfaces import ChangeDetectionOutput, ChangedRegion
from specialists.temporal_change.postprocessing import PostprocessingResult
from agent.controller import AgentController


@pytest.fixture
def clean_registry():
    """Create a clean registry with registered production specialists."""
    reg = ToolRegistry()
    reg.register(SingleImageRSSpecialistTool(), overwrite=True)
    reg.register(BiTemporalChangeSpecialistTool(), overwrite=True)
    return reg


@pytest.mark.asyncio
async def test_crops_and_overlay_identical_spatial_coordinates(clean_registry):
    """Prove that T0 crop, T1 crop, and change overlay correspond to identical spatial coordinates."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    req = QueryRequest(
        query="What changed in the detected area?",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    cre = response.metadata.get("changed_region_evidence")
    assert cre is not None, "ChangedRegionEvidence must be present"
    assert cre["has_change"] is True

    crop_t0_path = Path(cre["cropped_t0_path"])
    crop_t1_path = Path(cre["cropped_t1_path"])
    overlay_path = Path(cre["visualization_overlay_path"])
    mask_path = Path(cre["cropped_mask_path"])

    assert crop_t0_path.exists()
    assert crop_t1_path.exists()
    assert overlay_path.exists()
    assert mask_path.exists()

    with Image.open(crop_t0_path) as img_t0, \
         Image.open(crop_t1_path) as img_t1, \
         Image.open(overlay_path) as img_overlay, \
         Image.open(mask_path) as img_mask:

        w0, h0 = img_t0.size
        w1, h1 = img_t1.size
        wo, ho = img_overlay.size
        wm, hm = img_mask.size

        # 1. Coordinate & dimension equality across all artifacts
        assert (w0, h0) == (w1, h1), "T0 and T1 crops must have identical pixel dimensions"
        assert (w1, h1) == (wo, ho), "Overlay must have identical dimensions as T1 crop"
        assert (wo, ho) == (wm, hm), "Mask must have identical dimensions as T1 crop"
        assert [h0, w0] == cre["crop_dimensions"], "Image dimensions must match evidence crop_dimensions"

        # 2. Pixel-level slice equality verification against source images
        crop_coords = cre["metadata"]["spatial_coordinates"]
        ymin, xmin, ymax, xmax = crop_coords

        with Image.open(t0_path) as src_t0, Image.open(t1_path) as src_t1:
            expected_t0 = src_t0.crop((xmin, ymin, xmax, ymax)).convert("RGB")
            expected_t1 = src_t1.crop((xmin, ymin, xmax, ymax)).convert("RGB")

            assert np.array_equal(np.array(img_t0.convert("RGB")), np.array(expected_t0)), (
                "T0 crop pixels must match exact source T0 spatial slice"
            )
            assert np.array_equal(np.array(img_t1.convert("RGB")), np.array(expected_t1)), (
                "T1 crop pixels must match exact source T1 spatial slice"
            )


def test_only_tinycd_predictions_determine_region(tmp_path):
    """Prove that only TinyCD predictions determine region localization."""
    # Create two synthetic 256x256 test images
    img_t0 = Image.new("RGB", (256, 256), color=(100, 100, 100))
    img_t1 = Image.new("RGB", (256, 256), color=(120, 120, 120))
    t0_file = tmp_path / "synthetic_t0.png"
    t1_file = tmp_path / "synthetic_t1.png"
    img_t0.save(t0_file)
    img_t1.save(t1_file)

    # Scenario A: Change at top-left [30, 30, 70, 70]
    binary_a = np.zeros((256, 256), dtype=np.uint8)
    binary_a[30:70, 30:70] = 1
    region_a = ChangedRegion(region_id=1, bbox=(30, 30, 70, 70), area_pixels=1600, centroid=(50.0, 50.0), mean_confidence=0.95)
    postproc_a = PostprocessingResult(
        binary_map=binary_a,
        changed_pixel_ratio=1600 / (256 * 256),
        regions=[region_a],
        total_pixel_count=256 * 256,
        changed_pixel_count=1600,
    )
    change_out_a = ChangeDetectionOutput(
        binary_change_map=binary_a,
        change_probability_map=binary_a.astype(np.float32) * 0.95,
    )

    cre_a = extract_changed_region_evidence(
        t0_path=str(t0_file),
        t1_path=str(t1_file),
        change_output=change_out_a,
        postproc_result=postproc_a,
        output_dir=tmp_path / "out_a",
        request_id="req_a",
    )

    # Scenario B: Change at bottom-right [180, 180, 220, 220]
    binary_b = np.zeros((256, 256), dtype=np.uint8)
    binary_b[180:220, 180:220] = 1
    region_b = ChangedRegion(region_id=1, bbox=(180, 180, 220, 220), area_pixels=1600, centroid=(200.0, 200.0), mean_confidence=0.92)
    postproc_b = PostprocessingResult(
        binary_map=binary_b,
        changed_pixel_ratio=1600 / (256 * 256),
        regions=[region_b],
        total_pixel_count=256 * 256,
        changed_pixel_count=1600,
    )
    change_out_b = ChangeDetectionOutput(
        binary_change_map=binary_b,
        change_probability_map=binary_b.astype(np.float32) * 0.92,
    )


    cre_b = extract_changed_region_evidence(
        t0_path=str(t0_file),
        t1_path=str(t1_file),
        change_output=change_out_b,
        postproc_result=postproc_b,
        output_dir=tmp_path / "out_b",
        request_id="req_b",
    )

    # Regions must strictly match the respective prediction inputs
    assert cre_a.selected_region_pixel_bbox == [30, 30, 70, 70]
    assert cre_b.selected_region_pixel_bbox == [180, 180, 220, 220]

    # Crop coordinates must be centered around the prediction with padding
    assert cre_a.metadata["spatial_coordinates"][0] < 30  # padded ymin
    assert cre_b.metadata["spatial_coordinates"][0] < 180 # padded ymin
    assert cre_a.metadata["spatial_coordinates"] != cre_b.metadata["spatial_coordinates"]


@pytest.mark.asyncio
async def test_ground_truth_labels_never_used(clean_registry):
    """Prove that ground truth label directories are never accessed during Change-VQA execution."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    accessed_paths = []
    orig_open = open

    def auditing_open(file, *args, **kwargs):
        accessed_paths.append(str(file))
        return orig_open(file, *args, **kwargs)

    req = QueryRequest(
        query="What changed between these images?",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)

    with patch("builtins.open", side_effect=auditing_open):
        response = await controller.process_query(req)

    assert response.status == ToolStatus.PARTIAL_SUCCESS

    # Verify no file in /label/ or /labels/ was accessed
    label_accesses = [p for p in accessed_paths if "/label/" in p or "/labels/" in p]
    assert len(label_accesses) == 0, f"Ground-truth label leakage detected! Accessed: {label_accesses}"


@pytest.mark.asyncio
async def test_vlm_receives_complete_intended_evidence_package(clean_registry):
    """Verify that VLM receives: (1) cropped T0, (2) cropped T1, (3) change overlay, (4) query, (5) change metadata."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"
    user_query = "Describe the building expansion and identify when it occurred."

    req = QueryRequest(
        query=user_query,
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    meta = response.metadata
    assert meta["change_vqa_status"] == "CHANGE_VQA_MODEL_NOT_READY"

    # 1. Verification of 3 evidence images with distinct roles
    evidence_package = meta.get("evidence_package")
    assert evidence_package is not None, "evidence_package must be present in response metadata"
    assert len(evidence_package) == 3, f"Expected 3 evidence items, got {len(evidence_package)}"

    roles = [item["role"] for item in evidence_package]
    assert roles == ["BEFORE", "AFTER", "WHERE_CHANGE_OCCURRED"], f"Unexpected temporal roles: {roles}"

    t0_item = evidence_package[0]
    t1_item = evidence_package[1]
    overlay_item = evidence_package[2]

    assert "crop_t0_before_r18" in t0_item["image_id"]
    assert "crop_t1_after_r18" in t1_item["image_id"]
    assert "crop_overlay_change_r18" in overlay_item["image_id"]

    # 2. Verification of Original User Query preservation
    vlm_trace = [t for t in response.execution_trace if t.stage.value == "VLM_INPUT_PREPARED"][0]
    assert vlm_trace.details["query"] == user_query

    # 3. Verification of Deterministic Change Metadata preservation
    cre = meta.get("changed_region_evidence")
    assert cre["selected_region_id"] == 18
    assert cre["area_pixels"] == 3168
    assert cre["tinycd_threshold"] == 0.5
    assert cre["temporal_ordering"] == "T0->T1"


@pytest.mark.asyncio
async def test_trace_contains_vlm_input_prepared_with_artifact_ids(clean_registry):
    """Verify trace contains VLM_INPUT_PREPARED stage with explicit evidence artifact IDs."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    req = QueryRequest(
        query="Describe what changed between these two dates and where it occurred.",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    vlm_prep_entries = [t for t in response.execution_trace if t.stage == ExecutionStage.VLM_INPUT_PREPARED]
    assert len(vlm_prep_entries) == 1
    prep_entry = vlm_prep_entries[0]

    assert prep_entry.status == "COMPLETED"
    details = prep_entry.details

    # Must contain evidence_artifact_ids
    assert "evidence_artifact_ids" in details
    artifact_ids = details["evidence_artifact_ids"]
    assert len(artifact_ids) == 3
    assert any("before" in a for a in artifact_ids)
    assert any("after" in a for a in artifact_ids)
    assert any("change" in a for a in artifact_ids)

    # Must contain paths and roles
    assert details["temporal_roles"] == ["BEFORE", "AFTER", "WHERE_CHANGE_OCCURRED"]
    assert Path(details["crop_t0_path"]).exists()
    assert Path(details["crop_t1_path"]).exists()
    assert Path(details["overlay_path"]).exists()


@pytest.mark.asyncio
async def test_t1_only_fallback_when_contract_requires(clean_registry):
    """Verify preserved T1-only fallback path when single-image contract is explicitly requested."""
    t0_path = "data/official_levir_cd/test/A/test_10.png"
    t1_path = "data/official_levir_cd/test/B/test_10.png"

    # Request with single_image_t1 contract metadata
    req = QueryRequest(
        query="Describe what changed between these two dates and where it occurred.",
        images=[
            ImageInput(image_id="t0", path_or_uri=t0_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
            ImageInput(image_id="t1", path_or_uri=t1_path, format=ImageFormat.PNG, modality=ImageModality.OPTICAL),
        ],
        metadata={"vlm_contract": "single_image_t1"},
    )
    controller = AgentController(registry=clean_registry)
    response = await controller.process_query(req)

    vlm_prep_entries = [t for t in response.execution_trace if t.stage == ExecutionStage.VLM_INPUT_PREPARED]
    assert len(vlm_prep_entries) == 1
    prep_entry = vlm_prep_entries[0]

    details = prep_entry.details
    assert details["evidence_artifact_ids"] == ["crop_t1_after_r18"]
    assert details["temporal_roles"] == ["AFTER"]
    assert "SINGLE_IMAGE_CONTRACT_ENFORCED" in details["fallback_reason"]

    # VLM honestly returns CHANGE_VQA_MODEL_NOT_READY
    assert response.status == ToolStatus.PARTIAL_SUCCESS
    assert "CHANGE_VQA_MODEL_NOT_READY" in response.answer
