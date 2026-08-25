"""High-fidelity mock specialist tools for Single-Image remote sensing intelligence.

Implements Division 2 mocks (Sruthi's domain) covering VQA, Captioning, and Grounding.
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


class MockSingleImageVQATool(BaseSpecialistTool):
    """High-fidelity mock for Single-Image Remote Sensing VQA."""

    def __init__(self) -> None:
        super().__init__(
            name="single_image_vqa_mock",
            description="Mock specialist for single-image optical/SAR visual question answering.",
            supported_tasks={TaskType.SINGLE_IMAGE_VQA},
            version="1.0.0",
            metadata=ToolMetadata(
                name="single_image_vqa_mock",
                description="Mock specialist for single-image optical/SAR visual question answering.",
                version="1.0.0",
                supported_tasks=[TaskType.SINGLE_IMAGE_VQA],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=1,
                max_images=1,
                author_or_division="Division 2 (Sruthi Mock)",
                metadata={"architecture": "RS-VQA-Transformer-Mock", "resolution_agnostic": True},
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        query_lower = request.query.lower()
        image = request.images[0]
        
        # Determine contextual response based on query keywords
        if "count" in query_lower or "how many" in query_lower:
            answer = "There are 4 distinct aircraft visible on the apron and taxiway."
            evidence = [
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Aircraft 1",
                    confidence=0.92,
                    data={"bbox": [0.22, 0.35, 0.28, 0.42], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                ),
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Aircraft 2",
                    confidence=0.89,
                    data={"bbox": [0.31, 0.38, 0.37, 0.45], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                ),
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Aircraft 3",
                    confidence=0.86,
                    data={"bbox": [0.42, 0.50, 0.48, 0.57], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                ),
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Aircraft 4",
                    confidence=0.94,
                    data={"bbox": [0.55, 0.62, 0.61, 0.69], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                ),
            ]
            confidence = 0.90
        elif "water" in query_lower or "river" in query_lower or "lake" in query_lower:
            answer = "Yes, a significant water body (reservoir/river channel) is clearly visible across the eastern quadrant."
            evidence = [
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Water body / Reservoir",
                    confidence=0.95,
                    data={"bbox": [0.10, 0.65, 0.85, 0.95], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                )
            ]
            confidence = 0.95
        elif "cloud" in query_lower:
            answer = "The scene has approximately 5% thin cirrus cloud cover with high overall surface visibility."
            evidence = [
                Evidence(
                    type=EvidenceType.TEXT_EVIDENCE,
                    label="Cloud Assessment",
                    confidence=0.88,
                    data={"cloud_cover_percentage": 5.0},
                    image_id=image.image_id,
                )
            ]
            confidence = 0.88
        else:
            answer = f"The image shows a structured remote-sensing scene. In response to '{request.query}', target features were identified."
            evidence = [
                Evidence(
                    type=EvidenceType.BOUNDING_BOX,
                    label="Detected Region of Interest",
                    confidence=0.87,
                    data={"bbox": [0.2, 0.2, 0.8, 0.8], "format": "[ymin, xmin, ymax, xmax]"},
                    image_id=image.image_id,
                )
            ]
            confidence = 0.87

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            artifacts=[],
            model_info={"name": "RS-VQA-Mock", "version": "1.0", "modality": image.modality.value},
            parameters={"query": request.query},
            metadata={"source_image_id": image.image_id},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component="single_image_vqa_mock",
                    status="COMPLETED",
                    duration_ms=45.2,
                    details={"extracted_evidence_count": len(evidence)},
                )
            ],
        )


class MockSingleImageCaptionTool(BaseSpecialistTool):
    """High-fidelity mock for Single-Image Remote Sensing Captioning."""

    def __init__(self) -> None:
        super().__init__(
            name="single_image_caption_mock",
            description="Mock specialist for generating comprehensive remote-sensing scene captions.",
            supported_tasks={TaskType.SINGLE_IMAGE_CAPTION},
            version="1.0.0",
            metadata=ToolMetadata(
                name="single_image_caption_mock",
                description="Mock specialist for generating comprehensive remote-sensing scene captions.",
                version="1.0.0",
                supported_tasks=[TaskType.SINGLE_IMAGE_CAPTION],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=1,
                max_images=1,
                author_or_division="Division 2 (Sruthi Mock)",
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        image = request.images[0]
        modality_desc = f"{image.modality.value} imagery" if image.modality != ImageModality.UNKNOWN else "remote sensing imagery"
        answer = (
            f"An aerial view of a mixed urban and industrial landscape captured in {modality_desc}. "
            "Features include interconnected road networks, structured commercial buildings, open parking lots, "
            "and adjacent vegetated agricultural plots with minimal cloud obstruction."
        )

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.91,
            evidence=[
                Evidence(
                    type=EvidenceType.TEXT_EVIDENCE,
                    label="Scene Caption Summary",
                    confidence=0.91,
                    data={"semantic_categories": ["urban", "industrial", "vegetation", "road_network"]},
                    image_id=image.image_id,
                )
            ],
            artifacts=[],
            model_info={"name": "RS-Captioner-Mock", "version": "1.0"},
            parameters={},
            metadata={"source_image_id": image.image_id},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component="single_image_caption_mock",
                    status="COMPLETED",
                    duration_ms=38.7,
                    details={"tokens_generated": 36},
                )
            ],
        )


class MockSingleImageGroundingTool(BaseSpecialistTool):
    """High-fidelity mock for Single-Image Spatial Grounding / Localization."""

    def __init__(self) -> None:
        super().__init__(
            name="single_image_grounding_mock",
            description="Mock specialist for spatial grounding, localization, and bounding box extraction.",
            supported_tasks={TaskType.SINGLE_IMAGE_GROUNDING},
            version="1.0.0",
            metadata=ToolMetadata(
                name="single_image_grounding_mock",
                description="Mock specialist for spatial grounding, localization, and bounding box extraction.",
                version="1.0.0",
                supported_tasks=[TaskType.SINGLE_IMAGE_GROUNDING],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=1,
                max_images=1,
                author_or_division="Division 2 (Sruthi Mock)",
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        image = request.images[0]
        query_lower = request.query.lower()
        
        target_name = "Target Feature"
        if "water" in query_lower:
            target_name = "Water Body"
            bbox = [0.15, 0.60, 0.75, 0.90]
        elif "runway" in query_lower or "airport" in query_lower:
            target_name = "Airport Runway"
            bbox = [0.40, 0.10, 0.60, 0.90]
        elif "building" in query_lower or "urban" in query_lower:
            target_name = "Urban Building Cluster"
            bbox = [0.25, 0.25, 0.65, 0.70]
        else:
            target_name = "Localized Feature"
            bbox = [0.30, 0.30, 0.70, 0.70]

        evidence = [
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label=target_name,
                confidence=0.94,
                data={"bbox": bbox, "format": "[ymin, xmin, ymax, xmax]"},
                image_id=image.image_id,
            ),
            Evidence(
                type=EvidenceType.HEATMAP,
                label=f"{target_name} Spatial Attention Heatmap",
                confidence=0.91,
                data={"spatial_peak": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]},
                image_id=image.image_id,
            ),
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=f"Successfully localized {target_name} at normalized coordinates {bbox}.",
            confidence=0.93,
            evidence=evidence,
            artifacts=[],
            model_info={"name": "RS-Grounding-Mock", "version": "1.0"},
            parameters={"query": request.query},
            metadata={"target_feature": target_name},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component="single_image_grounding_mock",
                    status="COMPLETED",
                    duration_ms=52.1,
                    details={"grounded_targets": [target_name]},
                )
            ],
        )
