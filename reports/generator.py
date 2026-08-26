"""Comprehensive multi-format report generator for SatQuery AI Division 5.

Produces machine-readable JSON, GitHub-flavored Markdown, and standalone interactive
HTML intelligence reports embedding query context, grounded answers, calibrated confidence,
rendered visual evidence, auditable operational traces, and verified limitations.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from core.config import settings
from core.logging import get_logger
from core.schemas import Artifact, Evidence, ExecutionTraceEntry, QueryResponse, ToolStatus
from presentation.confidence import ConfidencePresenter
from presentation.evidence_renderer import EvidenceRenderer, EvidenceRenderingResult
from presentation.trace_presenter import TracePresenter
from presentation.ui_components import UIComponents
from reports.templates import HTML_REPORT_TEMPLATE

logger = get_logger("report_generator")


class GeneratedReport:
    """Represents a finalized report file artifact."""

    def __init__(
        self,
        report_id: str,
        format_type: str,
        file_path: str,
        artifact: Artifact,
        content: str,
    ):
        self.report_id = report_id
        self.format_type = format_type
        self.file_path = file_path
        self.artifact = artifact
        self.content = content


class ReportGenerator:
    """Builds and serializes multi-format reports from QueryResponse objects."""

    @classmethod
    def get_report_storage_dir(cls) -> Path:
        """Ensure report output directory exists under settings.artifact_storage_path."""
        rep_dir = settings.artifact_storage_path / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        return rep_dir

    @classmethod
    def derive_limitations(cls, response: QueryResponse) -> List[str]:
        """Derive authentic operational limitation statements based on response data."""
        limitations = []

        if response.confidence is None:
            limitations.append("Model confidence score was uncalibrated or unavailable for this specialist model output.")
        elif response.confidence < 0.65:
            limitations.append(f"Model returned low confidence ({response.confidence * 100:.1f}%); output should be validated by a geospatial analyst.")

        if not response.evidence:
            limitations.append("No explicit spatial bounding boxes or segmentation masks were localized for this query.")

        if response.status != ToolStatus.SUCCESS:
            limitations.append(f"Pipeline executed with non-optimal status: {response.status.value}.")

        if response.errors:
            for err in response.errors:
                limitations.append(f"Pipeline warning/error recorded: [{err.error_code}] {err.message}")

        if not limitations:
            limitations.append("Standard operational execution completed within nominal parameters. Raster resolution constraints apply.")

        return limitations

    @classmethod
    def generate_json_report(cls, response: QueryResponse, save_to_disk: bool = True) -> GeneratedReport:
        """Generate structured machine-readable JSON audit report."""
        report_id = str(uuid.uuid4())
        conf_display = ConfidencePresenter.format_confidence(response.confidence)
        trace_summary = TracePresenter.format_trace(response.execution_trace)
        limitations = cls.derive_limitations(response)

        data = {
            "report_id": report_id,
            "system": "SatQuery AI",
            "division": "Division 5 (Evidence, Evaluation & Presentation)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": response.request_id,
            "query": response.query,
            "resolved_task": response.resolved_task.value,
            "status": response.status.value,
            "answer": response.answer,
            "confidence": conf_display.model_dump(),
            "selected_tools": response.selected_tools,
            "evidence": [ev.model_dump() for ev in response.evidence],
            "artifacts": [art.model_dump() for art in response.artifacts],
            "execution_trace": trace_summary.model_dump(),
            "limitations": limitations,
            "metadata": response.metadata,
        }

        content = json.dumps(data, indent=2, default=str)
        filename = f"satquery_report_{report_id}.json"
        out_path = cls.get_report_storage_dir() / filename

        if save_to_disk:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        artifact = Artifact(
            artifact_id=report_id,
            name=filename,
            type="json_report",
            uri_or_path=str(out_path),
            description="Machine-readable JSON analysis report.",
            mime_type="application/json",
        )

        return GeneratedReport(
            report_id=report_id,
            format_type="json",
            file_path=str(out_path),
            artifact=artifact,
            content=content,
        )

    @classmethod
    def generate_markdown_report(cls, response: QueryResponse, save_to_disk: bool = True) -> GeneratedReport:
        """Generate formatted GitHub-flavored Markdown intelligence report."""
        report_id = str(uuid.uuid4())
        conf_display = ConfidencePresenter.format_confidence(response.confidence)
        trace_summary = TracePresenter.format_trace(response.execution_trace)
        limitations = cls.derive_limitations(response)

        md_lines = [
            f"# 🛰️ SatQuery AI — Intelligence Analysis Report",
            f"",
            f"**Report ID**: `{report_id}` | **Request ID**: `{response.request_id}`  ",
            f"**Timestamp (UTC)**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Division**: Division 5 (Evidence, Evaluation & Presentation Layer)  ",
            f"",
            f"---",
            f"",
            f"## 💬 Natural Language Query & Answer",
            f"",
            f"> **Query**: *\"{response.query}\"*",
            f"",
            f"**Synthesized Answer**:",
            f"```text",
            f"{response.answer}",
            f"```",
            f"",
            f"- **Confidence**: **{conf_display.formatted_percentage}** (`{conf_display.tier.value}`)",
            f"- **Confidence Rationale**: {conf_display.tier_description}",
            f"- **Resolved Task**: `{response.resolved_task.value}`",
            f"- **Execution Status**: `{response.status.value.upper()}`",
            f"- **Selected Specialist(s)**: {', '.join(f'`{t}`' for t in response.selected_tools) if response.selected_tools else 'None'}",
            f"",
            f"---",
            f"",
            f"## 🔍 Grounding Visual Evidence & Spatial Artifacts",
            f"",
        ]

        if not response.evidence:
            md_lines.append("_No spatial bounding boxes or segmentation masks generated for this query._\n")
        else:
            md_lines.append("| Evidence ID | Type | Label | Confidence | Spatial Coordinates / Data |")
            md_lines.append("| :--- | :--- | :--- | :---: | :--- |")
            for ev in response.evidence:
                conf_str = f"{ev.confidence * 100:.1f}%" if ev.confidence is not None else "N/A"
                data_str = f"`{json.dumps(ev.data)}`"
                md_lines.append(f"| `{ev.id[:8]}` | `{ev.type.value}` | **{ev.label}** | {conf_str} | {data_str} |")
            md_lines.append("")

        md_lines.extend([
            f"---",
            f"",
            f"## ⏱️ Auditable Operational Execution Trace",
            f"",
            TracePresenter.render_markdown_timeline(trace_summary),
            f"",
            f"---",
            f"",
            f"## ⚠️ Operational Limitations & Disclaimers",
            f"",
        ])

        for lim in limitations:
            md_lines.append(f"- {lim}")

        content = "\n".join(md_lines)
        filename = f"satquery_report_{report_id}.md"
        out_path = cls.get_report_storage_dir() / filename

        if save_to_disk:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        artifact = Artifact(
            artifact_id=report_id,
            name=filename,
            type="markdown_report",
            uri_or_path=str(out_path),
            description="Markdown analysis report.",
            mime_type="text/markdown",
        )

        return GeneratedReport(
            report_id=report_id,
            format_type="markdown",
            file_path=str(out_path),
            artifact=artifact,
            content=content,
        )

    @classmethod
    def generate_html_report(cls, response: QueryResponse, save_to_disk: bool = True) -> GeneratedReport:
        """Generate high-fidelity self-contained standalone HTML intelligence report."""
        report_id = str(uuid.uuid4())
        conf_badge = UIComponents.render_confidence_badge_html(response.confidence)
        trace_html = UIComponents.render_trace_timeline_html(response.execution_trace)
        limitations = cls.derive_limitations(response)
        trace_summary = TracePresenter.format_trace(response.execution_trace)

        # Build evidence cards
        evidence_cards = []
        if not response.evidence and not response.artifacts:
            evidence_cards.append("""
            <div style="grid-column: 1 / -1; padding: 20px; background: var(--bg-surface); border-radius: 8px; font-size: 13px; color: var(--text-muted);">
                No visual bounding boxes or spatial artifacts were generated for this response.
            </div>
            """)
        else:
            for art in response.artifacts:
                art_path = art.uri_or_path
                img_src = f"/api/v1/artifacts/{art.artifact_id}"
                
                # If image exists locally, try embedding as base64 for standalone portability
                if os.path.exists(art_path) and art_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with open(art_path, "rb") as img_file:
                            b64_data = base64.b64encode(img_file.read()).decode("utf-8")
                            img_src = f"data:image/png;base64,{b64_data}"
                    except Exception:
                        pass

                evidence_cards.append(f"""
                <div class="evidence-card">
                    <img src="{img_src}" alt="{art.name}" class="evidence-img" onerror="this.src='https://placehold.co/400x250/1e293b/94a3b8?text=Artifact+Preview';">
                    <div class="evidence-desc">
                        <div style="font-weight: 600; color: #93c5fd;">📁 {art.name}</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">{art.description}</div>
                        <div style="font-size: 10px; color: #6ee7b7; margin-top: 6px; font-family: monospace;">Type: {art.type}</div>
                    </div>
                </div>
                """)

            for ev in response.evidence:
                evidence_cards.append(f"""
                <div class="evidence-card">
                    <div style="padding: 16px; background: var(--bg-surface);">
                        <div style="font-weight: 600; color: #f43f5e;">📌 {ev.label}</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Type: {ev.type.value}</div>
                        <div style="font-size: 11px; color: #a78bfa; margin-top: 4px; font-family: monospace;">Data: {json.dumps(ev.data)}</div>
                    </div>
                </div>
                """)

        limitations_items = "".join([f"<li>{lim}</li>" for lim in limitations])

        html_content = HTML_REPORT_TEMPLATE.format(
            request_id=response.request_id,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            query=response.query,
            answer=response.answer,
            confidence_badge_html=conf_badge,
            resolved_task=response.resolved_task.value,
            image_count=response.agent_decision.image_count if response.agent_decision else 1,
            modalities=", ".join(m.value for m in response.agent_decision.detected_modalities) if response.agent_decision else "optical",
            selected_tools=", ".join(response.selected_tools) if response.selected_tools else "N/A",
            status=response.status.value.upper(),
            total_latency=trace_summary.formatted_total_duration,
            evidence_cards_html="".join(evidence_cards),
            trace_timeline_html=trace_html,
            limitations_list_html=limitations_items,
        )

        filename = f"satquery_report_{report_id}.html"
        out_path = cls.get_report_storage_dir() / filename

        if save_to_disk:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        artifact = Artifact(
            artifact_id=report_id,
            name=filename,
            type="html_report",
            uri_or_path=str(out_path),
            description="Standalone interactive HTML intelligence report.",
            mime_type="text/html",
        )

        return GeneratedReport(
            report_id=report_id,
            format_type="html",
            file_path=str(out_path),
            artifact=artifact,
            content=html_content,
        )
