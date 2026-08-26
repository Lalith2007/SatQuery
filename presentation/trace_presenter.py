"""Operational execution trace presenter and auditor for SatQuery AI Division 5.

Provides structured, auditable operational trace summaries, latency breakdowns,
and strict privacy enforcement preventing exposure of private chain-of-thought reasoning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.schemas import ExecutionStage, ExecutionTraceEntry


class FormattedTraceItem(BaseModel):
    """Presentation-ready trace item with normalized timestamps and duration."""
    step_index: int = Field(ge=1, description="1-indexed sequence number")
    stage: ExecutionStage = Field(description="Operational pipeline stage")
    component: str = Field(description="Subsystem/tool name")
    status: str = Field(description="Status string e.g. COMPLETED, FAILED, STARTED")
    timestamp_iso: str = Field(description="ISO 8601 UTC timestamp")
    duration_ms: Optional[float] = Field(default=None, description="Stage duration in milliseconds")
    duration_formatted: str = Field(description="Human-readable duration e.g. '45.2 ms'")
    percentage_of_total: Optional[float] = Field(default=None, description="Percentage of total execution time")
    details: Dict[str, Any] = Field(default_factory=dict, description="Safe operational metadata")
    icon: str = Field(description="Visual UI icon for the stage")


class ExecutionTraceSummary(BaseModel):
    """Complete structured audit summary for an execution trace."""
    total_stages: int = Field(ge=0, description="Total number of trace steps recorded")
    total_duration_ms: float = Field(ge=0.0, description="Total pipeline latency in milliseconds")
    formatted_total_duration: str = Field(description="Formatted total runtime e.g. '348.5 ms'")
    is_successful: bool = Field(description="True if all stages completed without failure")
    stages: List[FormattedTraceItem] = Field(default_factory=list, description="Ordered formatted stages")
    stage_latencies: Dict[str, float] = Field(default_factory=dict, description="Stage name to duration mapping")
    failed_stages: List[str] = Field(default_factory=list, description="List of failed stage descriptors if any")


class TracePresenter:
    """Formatter, auditor, and presenter for operational execution traces."""

    STAGE_ICONS: Dict[ExecutionStage, str] = {
        ExecutionStage.REQUEST_RECEIVED: "📥",
        ExecutionStage.INPUT_VALIDATED: "🛡️",
        ExecutionStage.TASK_RESOLVED: "🎯",
        ExecutionStage.TOOL_SELECTED: "🛠️",
        ExecutionStage.MODEL_INITIALIZED: "⚙️",
        ExecutionStage.INFERENCE_EXECUTED: "⚡",
        ExecutionStage.EVIDENCE_GENERATED: "🔍",
        ExecutionStage.RESULT_AGGREGATED: "📊",
        ExecutionStage.RESULT_RETURNED: "📤",
        ExecutionStage.ERROR_ENCOUNTERED: "⚠️",
    }

    # Forbidden private keys that must never be exposed in public traces
    FORBIDDEN_DETAIL_KEYS = {"chain_of_thought", "thought", "cot", "reasoning", "raw_prompt", "internal_thoughts"}

    @classmethod
    def sanitize_details(cls, details: Dict[str, Any]) -> Dict[str, Any]:
        """Strip any private chain-of-thought or sensitive reasoning from trace details."""
        if not details:
            return {}
        safe_dict = {}
        for k, v in details.items():
            if k.lower() in cls.FORBIDDEN_DETAIL_KEYS:
                continue
            safe_dict[k] = v
        return safe_dict

    @classmethod
    def format_trace(cls, raw_trace: List[ExecutionTraceEntry]) -> ExecutionTraceSummary:
        """Process and format raw execution trace into an auditable presentation summary."""
        if not raw_trace:
            return ExecutionTraceSummary(
                total_stages=0,
                total_duration_ms=0.0,
                formatted_total_duration="0.0 ms",
                is_successful=True,
                stages=[],
                stage_latencies={},
                failed_stages=[],
            )

        total_dur = sum((entry.duration_ms or 0.0) for entry in raw_trace)
        formatted_items: List[FormattedTraceItem] = []
        latencies: Dict[str, float] = {}
        failed: List[str] = []

        for idx, entry in enumerate(raw_trace):
            dur = entry.duration_ms
            dur_str = f"{dur:.1f} ms" if dur is not None else "—"
            pct = round((dur / total_dur) * 100.0, 1) if (total_dur > 0 and dur is not None) else None
            
            icon = cls.STAGE_ICONS.get(entry.stage, "📌")
            safe_details = cls.sanitize_details(entry.details)

            is_failed = str(entry.status).upper() in {"FAILED", "ERROR"}
            if is_failed:
                failed.append(f"{entry.stage.value} ({entry.component})")

            item = FormattedTraceItem(
                step_index=idx + 1,
                stage=entry.stage,
                component=entry.component,
                status=str(entry.status).upper(),
                timestamp_iso=entry.timestamp.isoformat() if hasattr(entry.timestamp, "isoformat") else str(entry.timestamp),
                duration_ms=dur,
                duration_formatted=dur_str,
                percentage_of_total=pct,
                details=safe_details,
                icon=icon,
            )
            formatted_items.append(item)
            if dur is not None:
                latencies[entry.stage.value] = dur

        return ExecutionTraceSummary(
            total_stages=len(formatted_items),
            total_duration_ms=round(total_dur, 2),
            formatted_total_duration=f"{total_dur:.1f} ms" if total_dur > 0 else "N/A",
            is_successful=len(failed) == 0,
            stages=formatted_items,
            stage_latencies=latencies,
            failed_stages=failed,
        )

    @classmethod
    def render_markdown_timeline(cls, summary: ExecutionTraceSummary) -> str:
        """Render markdown table and timeline for documentation/reports."""
        if summary.total_stages == 0:
            return "_No execution trace recorded._"

        lines = [
            f"**Total Execution Stages**: {summary.total_stages} | **Total Latency**: {summary.formatted_total_duration}",
            "",
            "| Step | Stage | Component | Status | Duration | % Latency |",
            "| :---: | :--- | :--- | :---: | :---: | :---: |",
        ]

        for s in summary.stages:
            pct_str = f"{s.percentage_of_total}%" if s.percentage_of_total is not None else "—"
            lines.append(f"| {s.step_index} | {s.icon} `{s.stage.value}` | `{s.component}` | **{s.status}** | {s.duration_formatted} | {pct_str} |")

        return "\n".join(lines)
