"""Reusable frontend and presentation UI components for SatQuery AI Division 5.

Provides HTML/CSS/JS snippet generators, SVG overlay renderers, interactive sliders,
and confidence badges for web dashboards and standalone reports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.schemas import Artifact, Evidence, EvidenceType, ExecutionTraceEntry, QueryResponse
from presentation.confidence import ConfidencePresenter
from presentation.trace_presenter import TracePresenter


class UIComponents:
    """Builder for interactive frontend presentation components."""

    @classmethod
    def render_confidence_badge_html(cls, confidence: Optional[float]) -> str:
        """Render high-fidelity styled HTML confidence badge."""
        disp = ConfidencePresenter.format_confidence(confidence)
        style_str = " ".join([f"{k}: {v};" for k, v in disp.badge_style.items()])
        icon = "🟢" if disp.tier.value == "HIGH" else "🟡" if disp.tier.value == "MODERATE" else "🔴" if disp.tier.value == "LOW" else "⚪"
        return f"""<span class="satquery-confidence-badge" style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; {style_str}">
            <span>{icon}</span>
            <span>Confidence: {disp.formatted_percentage}</span>
            <span style="font-size: 10px; opacity: 0.8;">[{disp.tier.value}]</span>
        </span>"""

    @classmethod
    def render_svg_bounding_box_overlay(
        cls,
        image_url_or_base64: str,
        boxes: List[Dict[str, Any]],
        view_width: int = 600,
        view_height: int = 400,
    ) -> str:
        """Generate interactive SVG overlay with hoverable bounding boxes and label tags."""
        svg_rects = []
        palette = [
            "#ef4444", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"
        ]

        for idx, box in enumerate(boxes):
            color = palette[idx % len(palette)]
            label = box.get("label", "Target")
            conf = box.get("confidence")
            conf_str = f" ({conf * 100:.0f}%)" if conf is not None else ""
            
            # Expect normalized [ymin, xmin, ymax, xmax] or standard [xmin, ymin, xmax, ymax]
            norm = box.get("normalized_bbox") or box.get("bbox") or [0.1, 0.1, 0.9, 0.9]
            if len(norm) == 4:
                ymin, xmin, ymax, xmax = norm
                x = xmin * view_width
                y = ymin * view_height
                w = (xmax - xmin) * view_width
                h = (ymax - ymin) * view_height

                svg_rects.append(f"""
                <g class="bbox-group" data-label="{label}">
                    <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="3" rx="2" />
                    <rect x="{x}" y="{max(0, y - 22)}" width="{len(label + conf_str) * 8 + 12}" height="20" fill="{color}" rx="3" />
                    <text x="{x + 6}" y="{max(14, y - 8)}" fill="#ffffff" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="600">{label}{conf_str}</text>
                </g>
                """)

        rects_html = "\n".join(svg_rects)
        return f"""
        <div class="interactive-grounding-viewer" style="position: relative; width: 100%; max-width: {view_width}px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
            <svg viewBox="0 0 {view_width} {view_height}" style="width: 100%; height: auto; display: block; background: #080c14;">
                <image href="{image_url_or_base64}" x="0" y="0" width="{view_width}" height="{view_height}" preserveAspectRatio="xMidYMid slice"/>
                {rects_html}
            </svg>
        </div>
        """

    @classmethod
    def render_bitemporal_slider_html(
        cls,
        t0_image_url: str,
        t1_image_url: str,
        slider_id: str = "bitemporal-slider",
    ) -> str:
        """Render interactive HTML/CSS/JS before-and-after image comparison slider."""
        return f"""
        <div class="comparison-slider-container" id="{slider_id}" style="position: relative; width: 100%; max-width: 650px; height: 380px; overflow: hidden; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); user-select: none;">
            <img src="{t1_image_url}" alt="T1 Acquisition" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;">
            <div class="slider-overlay" style="position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; border-right: 2px solid #06b6d4; box-shadow: 0 0 10px rgba(6,182,212,0.5);">
                <img src="{t0_image_url}" alt="T0 Acquisition" style="position: absolute; top: 0; left: 0; width: 650px; height: 100%; object-fit: cover;">
            </div>
            <span style="position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.7); color: #93c5fd; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">📅 Acquisition T0 (Before)</span>
            <span style="position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.7); color: #93c5fd; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">📅 Acquisition T1 (After)</span>
        </div>
        """

    @classmethod
    def render_trace_timeline_html(cls, raw_trace: List[ExecutionTraceEntry]) -> str:
        """Generate structured audit trace timeline card."""
        summary = TracePresenter.format_trace(raw_trace)
        if summary.total_stages == 0:
            return '<div style="color: var(--text-muted); font-size: 12px;">No trace events recorded.</div>'

        items_html = []
        for s in summary.stages:
            dur_text = f"{s.duration_ms:.1f}ms" if s.duration_ms is not None else ""
            status_color = "#10b981" if s.status == "COMPLETED" else "#ef4444" if s.status in {"FAILED", "ERROR"} else "#3b82f6"
            items_html.append(f"""
            <div class="trace-row" style="display: flex; align-items: baseline; gap: 10px; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 12px; font-family: 'JetBrains Mono', monospace;">
                <span>{s.icon}</span>
                <span style="color: #06b6d4; font-weight: 600;">{s.stage.value}</span>
                <span style="color: #9ca3af;">[{s.component}]</span>
                <span style="margin-left: auto; color: {status_color}; font-weight: 600; font-size: 11px;">{s.status}</span>
                <span style="color: #f59e0b; font-size: 11px; min-width: 50px; text-align: right;">{dur_text}</span>
            </div>
            """)

        return f"""
        <div class="trace-audit-card" style="background: #080c14; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em;">
                <span>Operational Audit Trail ({summary.total_stages} stages)</span>
                <span style="color: #f59e0b;">Total Latency: {summary.formatted_total_duration}</span>
            </div>
            <div class="trace-list">
                {''.join(items_html)}
            </div>
        </div>
        """
