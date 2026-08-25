"""Comprehensive Tests for Division 2: Single-Image Remote-Sensing Intelligence.

Tests:
- GroundingCoordinateParser token parsing and IoU math
- SingleImageRSSpecialistTool contract adherence
- VQA inference execution
- Visual Grounding normalized bounding box generation
- Scene captioning
- Resource metrics recording (latency, memory, device)
- Graceful failure and validation handling
- End-to-end integration with Division 1 AgentController
- Reproducibility verification (adapter weights loadability & dataset leakage audit)
"""

from pathlib import Path
import pytest

from core.interfaces import BaseSpecialistTool
from core.schemas import (
    EvidenceType,
    ExecutionStage,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    QueryResponse,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from agent.controller import AgentController
from registry.registry import ToolRegistry
from specialists.single_image.evaluation.reproducibility import (
    audit_dataset_splits_and_leakage,
    profile_synchronized_mps_latency,
    verify_adapter_tensor_architecture,
)
from specialists.single_image.grounding import GroundingCoordinateParser
from specialists.single_image.specialist import SingleImageRSSpecialistTool


def test_grounding_parser_location_tokens():
    raw = "<loc0082><loc0399><loc0942><loc0624> runway"
    evidence = GroundingCoordinateParser.parse_location_tokens(raw, label="runway", image_id="img-01")

    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.type == EvidenceType.BOUNDING_BOX
    assert ev.label == "runway"
    assert ev.image_id == "img-01"

    bbox = ev.data["bbox"]
    assert len(bbox) == 4
    # Bounding box should be normalized between 0.0 and 1.0
    ymin, xmin, ymax, xmax = bbox
    assert 0.0 <= ymin < ymax <= 1.0
    assert 0.0 <= xmin < xmax <= 1.0
    assert ev.data["format"] == "[ymin, xmin, ymax, xmax]"


def test_grounding_parser_iou_calculation():
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.1, 0.1, 0.5, 0.5]
    # Identical boxes should have IoU = 1.0
    assert GroundingCoordinateParser.calculate_iou(box1, box2) == 1.0

    # Non-overlapping boxes should have IoU = 0.0
    box_disjoint = [0.6, 0.6, 0.9, 0.9]
    assert GroundingCoordinateParser.calculate_iou(box1, box_disjoint) == 0.0

    # Overlapping boxes
    box_overlap = [0.3, 0.3, 0.7, 0.7]
    iou = GroundingCoordinateParser.calculate_iou(box1, box_overlap)
    assert 0.0 < iou < 1.0


def test_specialist_contract_conformance():
    specialist = SingleImageRSSpecialistTool()
    assert isinstance(specialist, BaseSpecialistTool)
    assert specialist.name == "single_image_rs_specialist"
    assert TaskType.SINGLE_IMAGE_VQA in specialist.supported_tasks
    assert TaskType.SINGLE_IMAGE_GROUNDING in specialist.supported_tasks
    assert TaskType.SINGLE_IMAGE_CAPTION in specialist.supported_tasks
    assert specialist.metadata.author_or_division == "Division 2 (Sruthi)"
    assert specialist.health_check() is True


@pytest.mark.asyncio
async def test_specialist_vqa_execution(optical_image_input: ImageInput):
    specialist = SingleImageRSSpecialistTool()
    req = ToolRequest(
        request_id="req-vqa-test",
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What is the dominant land cover in this scene?",
        images=[optical_image_input],
    )

    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert result.confidence is not None and result.confidence >= 0.85
    assert len(result.answer) > 20
    assert any(term in result.answer.lower() for term in ["infrastructure", "commercial", "agricultural", "land cover"])
    assert result.model_info["author"] == "Division 2 (Sruthi)"

    # Check execution trace
    stages = [entry.stage for entry in result.execution_trace]
    assert ExecutionStage.INFERENCE_EXECUTED in stages


@pytest.mark.asyncio
async def test_specialist_grounding_execution(optical_image_input: ImageInput):
    specialist = SingleImageRSSpecialistTool()
    req = ToolRequest(
        request_id="req-ground-test",
        task=TaskType.SINGLE_IMAGE_GROUNDING,
        query="Where is the airport runway?",
        images=[optical_image_input],
    )

    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert len(result.evidence) >= 1

    bbox_ev = result.evidence[0]
    assert bbox_ev.type == EvidenceType.BOUNDING_BOX
    assert "bbox" in bbox_ev.data
    bbox = bbox_ev.data["bbox"]
    assert len(bbox) == 4
    assert 0.0 <= bbox[0] < bbox[2] <= 1.0


@pytest.mark.asyncio
async def test_specialist_caption_execution(optical_image_input: ImageInput):
    specialist = SingleImageRSSpecialistTool()
    req = ToolRequest(
        request_id="req-caption-test",
        task=TaskType.SINGLE_IMAGE_CAPTION,
        query="Describe this scene",
        images=[optical_image_input],
    )

    result = await specialist.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert len(result.answer) > 25
    assert result.confidence is not None


@pytest.mark.asyncio
async def test_specialist_validation_rejection(optical_image_input: ImageInput, optical_image_t1_input: ImageInput):
    specialist = SingleImageRSSpecialistTool()
    # Reject 2 images (Single-Image specialist requires exactly 1 image)
    req = ToolRequest(
        request_id="req-invalid-count",
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What is here?",
        images=[optical_image_input, optical_image_t1_input],
    )

    result = await specialist.execute(req)
    assert result.status == ToolStatus.FAILED
    assert "Validation failed" in result.answer


@pytest.mark.asyncio
async def test_agent_controller_integration_with_division2_specialist(
    optical_image_input: ImageInput,
):
    # Setup registry with Division 2 Real Specialist
    custom_registry = ToolRegistry()
    div2_tool = SingleImageRSSpecialistTool()
    custom_registry.register(div2_tool)

    controller = AgentController(registry=custom_registry)
    query_req = QueryRequest(
        query="What is the dominant infrastructure in this satellite scene?",
        images=[optical_image_input],
    )

    response = await controller.process_query(query_req)
    assert isinstance(response, QueryResponse)
    assert response.status == ToolStatus.SUCCESS
    assert response.resolved_task == TaskType.SINGLE_IMAGE_VQA
    assert "single_image_rs_specialist" in response.selected_tools
    assert response.agent_decision is not None
    assert response.agent_decision.selected_specialist == "single_image_rs_specialist"


def test_reproducibility_adapter_verification():
    """Verify that adapter weights exist on disk, are loadable, and match LoRA rank and tensor count."""
    audit = verify_adapter_tensor_architecture()
    assert "VERIFIED" in audit["status"]
    assert audit["total_tensors_in_file"] == 56
    assert audit["lora_rank"] == 8
    assert audit["adapted_layer_count"] == 4


def test_reproducibility_dataset_leakage_audit():
    """Verify zero overlap between train and test evaluation splits."""
    audit = audit_dataset_splits_and_leakage()
    assert audit["data_leakage_detected"] is False
    assert audit["leakage_count"] == 0
    assert audit["train_samples"] == 60
    assert audit["val_samples"] == 15
    assert audit["test_samples"] == 25
