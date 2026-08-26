"""Unit tests for operational execution trace presenter and auditor (Division 5)."""

from datetime import datetime, timezone
import pytest

from core.schemas import ExecutionStage, ExecutionTraceEntry
from presentation.trace_presenter import ExecutionTraceSummary, TracePresenter


def test_empty_trace_summary():
    """Verify handling of empty raw execution traces."""
    summary = TracePresenter.format_trace([])
    assert summary.total_stages == 0
    assert summary.total_duration_ms == 0.0
    assert summary.is_successful


def test_trace_duration_and_percentage_calculation():
    """Verify stage duration summation and percentage latency calculation."""
    raw_trace = [
        ExecutionTraceEntry(
            stage=ExecutionStage.INPUT_VALIDATED,
            component="InputValidator",
            status="COMPLETED",
            duration_ms=10.0,
        ),
        ExecutionTraceEntry(
            stage=ExecutionStage.TASK_RESOLVED,
            component="IntentResolver",
            status="COMPLETED",
            duration_ms=10.0,
        ),
        ExecutionTraceEntry(
            stage=ExecutionStage.INFERENCE_EXECUTED,
            component="SpecialistModel",
            status="COMPLETED",
            duration_ms=80.0,
        ),
    ]

    summary = TracePresenter.format_trace(raw_trace)
    assert summary.total_stages == 3
    assert summary.total_duration_ms == 100.0
    assert summary.formatted_total_duration == "100.0 ms"
    assert summary.is_successful

    # Stage 3 should account for 80% of total latency
    inference_stage = summary.stages[2]
    assert inference_stage.stage == ExecutionStage.INFERENCE_EXECUTED
    assert inference_stage.percentage_of_total == 80.0
    assert inference_stage.icon == "⚡"


def test_trace_privacy_enforcement_no_cot_leakage():
    """Strict privacy rule: Ensure private chain-of-thought keys are stripped from public traces."""
    raw_trace = [
        ExecutionTraceEntry(
            stage=ExecutionStage.TASK_RESOLVED,
            component="IntentResolver",
            status="COMPLETED",
            duration_ms=5.0,
            details={
                "public_parameter": "resolved_task_vqa",
                "chain_of_thought": "Private internal model step-by-step reasoning that must not be exposed",
                "thought": "Another private reasoning string",
                "cot": "Do not show this",
            },
        )
    ]

    summary = TracePresenter.format_trace(raw_trace)
    stage_details = summary.stages[0].details
    assert "public_parameter" in stage_details
    assert "chain_of_thought" not in stage_details
    assert "thought" not in stage_details
    assert "cot" not in stage_details


def test_trace_failure_detection():
    """Verify failed stages are properly recorded and flagged."""
    raw_trace = [
        ExecutionTraceEntry(
            stage=ExecutionStage.INPUT_VALIDATED,
            component="InputValidator",
            status="COMPLETED",
            duration_ms=5.0,
        ),
        ExecutionTraceEntry(
            stage=ExecutionStage.INFERENCE_EXECUTED,
            component="SpecialistModel",
            status="FAILED",
            duration_ms=25.0,
        ),
    ]

    summary = TracePresenter.format_trace(raw_trace)
    assert not summary.is_successful
    assert len(summary.failed_stages) == 1
    assert "INFERENCE_EXECUTED" in summary.failed_stages[0]


def test_markdown_timeline_rendering():
    """Verify markdown timeline table generation."""
    raw_trace = [
        ExecutionTraceEntry(
            stage=ExecutionStage.INPUT_VALIDATED,
            component="InputValidator",
            status="COMPLETED",
            duration_ms=2.5,
        )
    ]
    summary = TracePresenter.format_trace(raw_trace)
    md = TracePresenter.render_markdown_timeline(summary)
    assert "INPUT_VALIDATED" in md
    assert "InputValidator" in md
    assert "2.5 ms" in md
