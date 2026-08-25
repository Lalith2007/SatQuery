"""Unit tests for canonical schemas and contracts."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from core.schemas import (
    Artifact,
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    GeoSpatialMetadata,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    QueryResponse,
    SatQueryErrorDetail,
    TaskIntent,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


def test_image_modality_enum():
    assert ImageModality.OPTICAL == "optical"
    assert ImageModality.SAR == "sar"
    assert ImageModality.MULTISPECTRAL == "multispectral"
    assert ImageModality.UNKNOWN == "unknown"


def test_image_format_restrictions():
    # Only approved remote sensing formats
    assert ImageFormat.GEOTIFF == "geotiff"
    assert ImageFormat.TIFF == "tiff"
    assert ImageFormat.PNG == "png"
    assert ImageFormat.JPEG == "jpeg"
    with pytest.raises(ValueError):
        ImageFormat("webp")


def test_geospatial_metadata_optionality():
    # Never fabricated: all fields can be None
    meta = GeoSpatialMetadata()
    assert meta.crs is None
    assert meta.geo_bounds is None
    assert meta.acquisition_timestamp is None


def test_image_input_validation():
    img = ImageInput(
        path_or_uri="/path/to/test.tif",
        format=ImageFormat.TIFF,
        modality=ImageModality.OPTICAL,
        width=512,
        height=512,
        channel_count=4,
    )
    assert img.image_id is not None
    assert img.width == 512
    assert img.format == ImageFormat.TIFF

    # Test invalid dimensions
    with pytest.raises(ValidationError):
        ImageInput(path_or_uri="/path", format=ImageFormat.PNG, width=-10)


def test_task_intent_model():
    intent = TaskIntent(
        task=TaskType.CHANGE_ANALYSIS,
        confidence=0.95,
        intent_explanation="Bi-temporal change requested.",
        target_features=["urban", "deforestation"],
    )
    assert intent.task == TaskType.CHANGE_ANALYSIS
    assert intent.confidence == 0.95
    assert "urban" in intent.target_features
    # Verify no private CoT field
    assert not hasattr(intent, "reasoning")


def test_evidence_and_artifact_serialization():
    ev = Evidence(
        type=EvidenceType.BOUNDING_BOX,
        label="Airport Runway",
        confidence=0.96,
        data={"bbox": [0.1, 0.2, 0.3, 0.4]},
    )
    assert ev.type == EvidenceType.BOUNDING_BOX
    assert ev.confidence == 0.96

    art = Artifact(
        name="change_mask.png",
        type="change_map",
        uri_or_path="/storage/change_mask.png",
    )
    assert art.artifact_id is not None
    assert art.name == "change_mask.png"


def test_query_response_serialization():
    resp = QueryResponse(
        request_id="req-123",
        query="What is in this scene?",
        resolved_task=TaskType.SINGLE_IMAGE_VQA,
        status=ToolStatus.SUCCESS,
        answer="A large container terminal with cargo vessels.",
        confidence=0.91,
        evidence=[],
        artifacts=[],
        execution_trace=[
            ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component="vqa_specialist",
                status="COMPLETED",
            )
        ],
    )
    json_data = resp.model_dump_json()
    assert "req-123" in json_data
    assert "single_image_vqa" in json_data
