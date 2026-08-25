"""High-fidelity mock specialist tool for Bi-Temporal Change Intelligence.

Implements Division 3 mock (Dheeraj's domain) covering Change Analysis, Change VQA,
and Change Localization across temporal image pairs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid

from core.interfaces import BaseSpecialistTool
from core.schemas import (
    Artifact,
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class MockBiTemporalChangeTool(BaseSpecialistTool):
    """High-fidelity mock for Bi-Temporal Remote Sensing Change Intelligence."""

    def __init__(self) -> None:
        super().__init__(
            name="bi_temporal_change_mock",
            description="Mock specialist for bi-temporal change detection, change description, and spatial localization.",
            supported_tasks={TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA},
            version="1.0.0",
            metadata=ToolMetadata(
                name="bi_temporal_change_mock",
                description="Mock specialist for bi-temporal change detection, change description, and spatial localization.",
                version="1.0.0",
                supported_tasks=[TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=2,
                max_images=2,
                author_or_division="Division 3 (Dheeraj Mock)",
                metadata={"bitemporal_alignment_required": False, "difference_algorithm": "Siamese-Transformer-Diff-Mock"},
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        img_t0 = request.images[0]
        img_t1 = request.images[1]
        query_lower = request.query.lower()

        # Provide detailed change analysis
        if "built-up" in query_lower or "urban" in query_lower or "construction" in query_lower:
            answer = (
                "Bi-temporal change analysis indicates a 22.4% increase in built-up infrastructure between T0 and T1. "
                "Major newly constructed commercial facilities and paved road corridors are localized in the southwest quadrant."
            )
            change_category = "Urban Expansion"
        elif "water" in query_lower or "flood" in query_lower or "reservoir" in query_lower:
            answer = (
                "Bi-temporal analysis reveals significant water body surface expansion (+18.7%), "
                "indicating seasonal flooding along the southern river delta."
            )
            change_category = "Hydrological Change"
        elif "vegetation" in query_lower or "deforestation" in query_lower or "forest" in query_lower:
            answer = (
                "Vegetation analysis shows a net decrease in dense canopy cover (-12.3 hectares) "
                "in the north-central sector, with signs of land clearing."
            )
            change_category = "Deforestation / Canopy Loss"
        else:
            answer = (
                "Bi-temporal analysis between T0 and T1 detected prominent surface changes across 3 primary clusters: "
                "new ground development (14.2 ha), road extension (1.8 km), and reduced vegetation index in the southern sector."
            )
            change_category = "General Surface Transition"

        # Construct structured evidence
        evidence = [
            Evidence(
                type=EvidenceType.CHANGE_MAP,
                label=f"Bi-Temporal Difference Map ({change_category})",
                confidence=0.92,
                data={
                    "change_category": change_category,
                    "t0_image_id": img_t0.image_id,
                    "t1_image_id": img_t1.image_id,
                    "changed_area_ha": 14.2,
                    "change_confidence_mean": 0.91,
                },
            ),
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Primary Change Cluster 1",
                confidence=0.94,
                data={"bbox": [0.45, 0.20, 0.78, 0.55], "format": "[ymin, xmin, ymax, xmax]", "cluster": "SW Urban"},
                image_id=img_t1.image_id,
            ),
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Secondary Change Cluster 2",
                confidence=0.88,
                data={"bbox": [0.12, 0.70, 0.35, 0.92], "format": "[ymin, xmin, ymax, xmax]", "cluster": "NE Clearing"},
                image_id=img_t1.image_id,
            ),
            Evidence(
                type=EvidenceType.TEXT_EVIDENCE,
                label="Quantitative Change Metrics",
                confidence=0.95,
                data={
                    "total_changed_percentage": 7.8,
                    "expansion_rate_ha_per_year": 14.2,
                    "dominant_transition": "Vegetation -> Built-up",
                },
            ),
        ]

        # Construct realistic artifact reference
        artifacts = [
            Artifact(
                name="bi_temporal_change_map.png",
                type="change_map",
                uri_or_path="artifacts_storage/mock_change_map.png",
                description="Binary and color-coded change difference mask generated across T0 and T1 acquisitions.",
                mime_type="image/png",
                metadata={"t0_id": img_t0.image_id, "t1_id": img_t1.image_id},
            )
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.92,
            evidence=evidence,
            artifacts=artifacts,
            model_info={"name": "BiTemporal-SiamDiff-Mock", "version": "1.0"},
            parameters={"query": request.query},
            metadata={"t0_id": img_t0.image_id, "t1_id": img_t1.image_id, "change_category": change_category},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component="bi_temporal_change_mock",
                    status="COMPLETED",
                    duration_ms=88.4,
                    details={"clusters_detected": 2, "changed_pixels": 45120},
                )
            ],
        )
