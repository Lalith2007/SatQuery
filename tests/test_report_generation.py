"""Unit tests for multi-format report generator (Division 5)."""

import json
import os
from pathlib import Path
import pytest

from core.schemas import (
    Artifact,
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    QueryResponse,
    TaskType,
    ToolStatus,
)
from reports.generator import ReportGenerator


@pytest.fixture
def mock_query_response() -> QueryResponse:
    """Fixture providing a rich standard QueryResponse for report generation testing."""
    return QueryResponse(
        request_id="req-test-9999",
        query="Where is the airport runway in this optical image?",
        resolved_task=TaskType.SINGLE_IMAGE_GROUNDING,
        status=ToolStatus.SUCCESS,
        answer="Successfully localized Airport Runway at coordinates [0.1, 0.2, 0.5, 0.8].",
        confidence=0.94,
        evidence=[
            Evidence(
                id="ev-001",
                type=EvidenceType.BOUNDING_BOX,
                label="Airport Runway",
                confidence=0.95,
                data={"bbox": [0.1, 0.2, 0.5, 0.8], "format": "[ymin, xmin, ymax, xmax]"},
            )
        ],
        artifacts=[
            Artifact(
                artifact_id="art-001",
                name="runway_grounding_map.png",
                type="annotated_image",
                uri_or_path="demo_assets/demo_airport_grounding.png",
                description="Visual grounding annotation mask.",
            )
        ],
        execution_trace=[
            ExecutionTraceEntry(
                stage=ExecutionStage.INPUT_VALIDATED,
                component="InputValidator",
                status="COMPLETED",
                duration_ms=3.2,
            ),
            ExecutionTraceEntry(
                stage=ExecutionStage.TASK_RESOLVED,
                component="IntentResolver",
                status="COMPLETED",
                duration_ms=2.1,
            ),
            ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component="PaliGemmaRSInferenceEngine",
                status="COMPLETED",
                duration_ms=145.0,
            ),
        ],
        selected_tools=["single_image_grounding_specialist"],
        metadata={"sensor": "Optical RGB", "resolution_m": 0.5},
    )


def test_generate_json_report(mock_query_response):
    """Verify JSON report generation contains all required canonical metadata."""
    report = ReportGenerator.generate_json_report(mock_query_response, save_to_disk=True)
    assert report.format_type == "json"
    assert os.path.exists(report.file_path)

    data = json.loads(report.content)
    assert data["request_id"] == "req-test-9999"
    assert data["query"] == mock_query_response.query
    assert data["answer"] == mock_query_response.answer
    assert data["confidence"]["formatted_percentage"] == "94.0%"
    assert data["confidence"]["tier"] == "HIGH"
    assert len(data["evidence"]) == 1
    assert len(data["artifacts"]) == 1
    assert "limitations" in data


def test_generate_markdown_report(mock_query_response):
    """Verify Markdown report generation format and content."""
    report = ReportGenerator.generate_markdown_report(mock_query_response, save_to_disk=True)
    assert report.format_type == "markdown"
    assert os.path.exists(report.file_path)

    md = report.content
    assert "# 🛰️ SatQuery AI — Intelligence Analysis Report" in md
    assert "req-test-9999" in md
    assert "Airport Runway" in md
    assert "94.0%" in md
    assert "Auditable Operational Execution Trace" in md


def test_generate_html_report(mock_query_response):
    """Verify standalone HTML report generation and structure."""
    report = ReportGenerator.generate_html_report(mock_query_response, save_to_disk=True)
    assert report.format_type == "html"
    assert os.path.exists(report.file_path)

    html = report.content
    assert "<!DOCTYPE html>" in html
    assert "SatQuery AI Analysis Report" in html
    assert "req-test-9999" in html
    assert "Airport Runway" in html
    assert "window.print()" in html


def test_derive_limitations():
    """Verify automatic derivation of authentic operational limitations."""
    # Test response with uncalibrated confidence and failed status
    uncalibrated_resp = QueryResponse(
        request_id="req-lim-001",
        query="Test query",
        resolved_task=TaskType.SINGLE_IMAGE_VQA,
        status=ToolStatus.FAILED,
        answer="Error encountered.",
        confidence=None,
    )

    limitations = ReportGenerator.derive_limitations(uncalibrated_resp)
    assert any("uncalibrated or unavailable" in lim for lim in limitations)
    assert any("non-optimal status" in lim for lim in limitations)
    assert any("No explicit spatial bounding boxes" in lim for lim in limitations)
