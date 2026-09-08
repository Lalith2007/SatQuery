"""Contract and integration conformance tests for Division 5."""

import json
from pathlib import Path
import pytest

from core.schemas import (
    Artifact,
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryResponse,
    TaskType,
    ToolResult,
    ToolStatus,
)
from presentation.confidence import ConfidencePresenter
from presentation.evidence_renderer import EvidenceRenderer
from presentation.trace_presenter import TracePresenter
from reports.generator import ReportGenerator


def test_tool_result_to_division5_pipeline():
    """Verify that a canonical ToolResult flows smoothly into Division 5 presentation and reports."""
    tool_result = ToolResult(
        request_id="req-contract-001",
        task=TaskType.SINGLE_IMAGE_GROUNDING,
        status=ToolStatus.SUCCESS,
        answer="Detected 1 industrial cooling tower.",
        confidence=0.91,
        evidence=[
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Cooling Tower",
                confidence=0.91,
                data={"bbox": [0.2, 0.3, 0.6, 0.7], "format": "[ymin, xmin, ymax, xmax]"},
            )
        ],
        artifacts=[
            Artifact(
                name="cooling_tower.png",
                type="annotated_image",
                uri_or_path="demo_assets/demo_optical_single.png",
                description="Cooling tower grounding map.",
            )
        ],
        model_info={"name": "PaliGemma-3B-RS", "version": "1.0"},
        execution_trace=[
            ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component="PaliGemmaRSInferenceEngine",
                status="COMPLETED",
                duration_ms=85.0,
            )
        ],
    )

    # 1. Test confidence formatting
    conf_disp = ConfidencePresenter.format_confidence(tool_result.confidence)
    assert conf_disp.tier.value == "HIGH"
    assert conf_disp.formatted_percentage == "91.0%"

    # 2. Test trace formatting
    trace_summary = TracePresenter.format_trace(tool_result.execution_trace)
    assert trace_summary.total_stages == 1
    assert trace_summary.total_duration_ms == 85.0

    # 3. Test report generation
    query_resp = QueryResponse(
        request_id=tool_result.request_id,
        query="Where is the cooling tower?",
        resolved_task=tool_result.task,
        status=tool_result.status,
        answer=tool_result.answer,
        confidence=tool_result.confidence,
        evidence=tool_result.evidence,
        artifacts=tool_result.artifacts,
        execution_trace=tool_result.execution_trace,
    )

    json_rep = ReportGenerator.generate_json_report(query_resp)
    assert json_rep.artifact.name.endswith(".json")
    
    html_rep = ReportGenerator.generate_html_report(query_resp)
    assert html_rep.artifact.name.endswith(".html")


def test_contract_serialization_json_compatibility():
    """Verify serialized JSON contains no binary payloads and strictly string references."""
    query_resp = QueryResponse(
        request_id="req-json-compat",
        query="Test query",
        resolved_task=TaskType.SINGLE_IMAGE_VQA,
        status=ToolStatus.SUCCESS,
        answer="Valid response text.",
        confidence=0.88,
        artifacts=[
            Artifact(
                artifact_id="art-compat-001",
                name="test_artifact.png",
                type="change_map",
                uri_or_path="artifacts_storage/test.png",
                description="Test map.",
            )
        ],
    )

    dumped = query_resp.model_dump()
    json_str = json.dumps(dumped)
    reloaded = json.loads(json_str)

    assert reloaded["request_id"] == "req-json-compat"
    assert reloaded["artifacts"][0]["artifact_id"] == "art-compat-001"
    assert reloaded["artifacts"][0]["uri_or_path"] == "artifacts_storage/test.png"
