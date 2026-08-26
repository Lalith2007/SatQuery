"""Failure handling and defensive degradation tests for Division 5."""

import pytest
from core.schemas import Evidence, EvidenceType, QueryResponse, TaskType, ToolStatus
from presentation.confidence import ConfidencePresenter, ConfidenceTier
from presentation.evidence_renderer import EvidenceRenderer
from presentation.trace_presenter import TracePresenter
from reports.generator import ReportGenerator
from evaluation.runner import BenchmarkRunner


def test_missing_evidence_graceful_handling():
    """Verify system handles empty evidence gracefully without errors."""
    resp = QueryResponse(
        request_id="req-fail-001",
        query="Test query",
        resolved_task=TaskType.SINGLE_IMAGE_VQA,
        status=ToolStatus.SUCCESS,
        answer="Normal text answer without visual evidence.",
        confidence=0.88,
        evidence=[],
    )

    rendered = EvidenceRenderer.render_all_evidence([], [])
    assert rendered == []

    rep = ReportGenerator.generate_markdown_report(resp)
    assert "No spatial bounding boxes" in rep.content


def test_malformed_bounding_box_coordinates():
    """Verify evidence renderer handles malformed bbox arrays without crashing."""
    malformed_ev = [
        Evidence(
            type=EvidenceType.BOUNDING_BOX,
            label="Broken Box",
            data={"bbox": [0.1, 0.2]},  # Only 2 coordinates instead of 4
        )
    ]

    # Should safely skip the malformed box and return valid fallback
    res = EvidenceRenderer.render_bounding_boxes("non_existent.png", malformed_ev)
    assert res.is_fallback


def test_failed_specialist_result_handling():
    """Verify presentation layer handles FAILED status appropriately."""
    failed_resp = QueryResponse(
        request_id="req-fail-002",
        query="What is this object?",
        resolved_task=TaskType.SINGLE_IMAGE_VQA,
        status=ToolStatus.FAILED,
        answer="Specialist execution failed due to internal error.",
        confidence=None,
    )

    conf_disp = ConfidencePresenter.format_confidence(failed_resp.confidence)
    assert conf_disp.tier == ConfidenceTier.UNAVAILABLE

    report = ReportGenerator.generate_html_report(failed_resp)
    assert "FAILED" in report.content
    assert "non-optimal status" in report.content


def test_unknown_benchmark_lookup_error():
    """Verify BenchmarkRunner raises ValueError on unknown benchmark name."""
    with pytest.raises(ValueError, match="Unknown benchmark"):
        BenchmarkRunner.get_evaluator("unknown_benchmark_xyz")


def test_empty_benchmark_evaluation():
    """Verify evaluators handle empty predictions and ground truths gracefully."""
    evaluator = BenchmarkRunner.get_evaluator("vrsbench")
    res = evaluator.evaluate([], [])
    assert res.total_samples == 0
    assert res.aggregate_normalized_score == 0.0
