"""Alternate mock specialist tool for demonstrating runtime tool swapping.

Clearly identified as a demo mock specialist (version 2.0.0-demo-mock).
Used to prove that specialist implementations can be swapped in the ToolRegistry
without modifying the AgentController, Router, or ExecutionEngine.
"""

from __future__ import annotations

from core.interfaces import BaseSpecialistTool
from core.schemas import (
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


class AlternateMockSingleImageVQATool(BaseSpecialistTool):
    """Explicitly named alternate mock specialist for Single-Image VQA demonstration."""

    def __init__(self) -> None:
        super().__init__(
            name="alternate_single_image_vqa_mock",
            description="Alternate demo mock specialist demonstrating pluggable tool swapping without agent code changes.",
            supported_tasks={TaskType.SINGLE_IMAGE_VQA},
            version="2.0.0-demo-mock",
            metadata=ToolMetadata(
                name="alternate_single_image_vqa_mock",
                description="Alternate demo mock specialist demonstrating pluggable tool swapping without agent code changes.",
                version="2.0.0-demo-mock",
                supported_tasks=[TaskType.SINGLE_IMAGE_VQA],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=1,
                max_images=1,
                author_or_division="Division 2 (Alternate Demo Mock)",
                metadata={"architecture": "Alternate-Mock-Pipeline", "demo_mode": True},
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        image = request.images[0]
        answer = (
            f"[Alternate Specialist v2.0-demo-mock]: High-resolution feature extraction processed '{request.query}'. "
            f"Observed 4 aircraft, 2 hangars, and active taxiway marking in {image.modality.value} imagery."
        )
        evidence = [
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Aircraft Cluster (Alternate Model)",
                confidence=0.97,
                data={"bbox": [0.20, 0.30, 0.60, 0.70], "format": "[ymin, xmin, ymax, xmax]"},
                image_id=image.image_id,
            )
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.97,
            evidence=evidence,
            artifacts=[],
            model_info={"name": "Alternate-Mock-VQA", "version": "2.0.0-demo-mock"},
            parameters={"query": request.query},
            metadata={"swapped_implementation": True, "source_image_id": image.image_id},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                    duration_ms=32.4,
                    details={"swapped_specialist_invoked": True},
                )
            ],
        )
