"""Division 2 Specialist: Single-Image Remote-Sensing Intelligence.

Owner: Sruthi
Implements BaseSpecialistTool for:
- Single-Image Visual Question Answering (VQA)
- Text-Guided Visual Grounding (Normalized Bounding Boxes)
- Scene Captioning & Land-Cover Description

Model: PaliGemma 3B — SatQuery Remote-Sensing Adapted
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.interfaces import BaseSpecialistTool, ValidationResult
from core.logging import get_logger
from core.schemas import (
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    ImageFormat,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from specialists.single_image.grounding import GroundingCoordinateParser
from specialists.single_image.model import PaliGemmaRSInferenceEngine

logger = get_logger("single_image_specialist")


class SingleImageRSSpecialistTool(BaseSpecialistTool):
    """Real Single-Image Remote-Sensing Intelligence Specialist for Division 2."""

    def __init__(
        self,
        base_model_id: str = "google/paligemma-3b-pt-224",
        adapter_path: Optional[str] = None,
    ) -> None:
        metadata = ToolMetadata(
            name="single_image_rs_specialist",
            description=(
                "Single-Image Remote-Sensing Specialist powered by PaliGemma 3B — "
                "SatQuery Remote-Sensing Adapted. Supports VQA, text-guided visual grounding, "
                "and scene description across optical and multispectral satellite imagery."
            ),
            version="1.0.0-adapted",
            supported_tasks=[
                TaskType.SINGLE_IMAGE_VQA,
                TaskType.SINGLE_IMAGE_GROUNDING,
                TaskType.SINGLE_IMAGE_CAPTION,
            ],
            required_modalities=[
                ImageModality.OPTICAL,
                ImageModality.MULTISPECTRAL,
                ImageModality.SAR,
            ],
            min_images=1,
            max_images=1,
            author_or_division="Division 2 (Sruthi)",
            metadata={
                "base_architecture": "SigLIP-So400m + Gemma-2B",
                "model_name": "PaliGemma 3B — SatQuery Remote-Sensing Adapted",
                "base_checkpoint": base_model_id,
                "adaptation_method": "PEFT / LoRA (rank=8, alpha=16)",
                "coordinate_convention": "[ymin, xmin, ymax, xmax] (normalized 0.0 - 1.0)",
            },
        )
        super().__init__(
            name="single_image_rs_specialist",
            description=metadata.description,
            supported_tasks=set(metadata.supported_tasks),
            version=metadata.version,
            metadata=metadata,
        )
        self.engine = PaliGemmaRSInferenceEngine.get_instance(
            base_model_id=base_model_id,
            adapter_path=adapter_path,
        )

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        """Validate input constraints for single-image vision-language analysis."""
        errors: List[str] = []

        if len(request.images) != 1:
            errors.append(f"SingleImageRSSpecialist requires exactly 1 image input, received {len(request.images)}.")

        if request.task not in self.supported_tasks:
            errors.append(
                f"Task '{request.task.value}' is not supported by {self.name}. "
                f"Supported: {[t.value for t in self.supported_tasks]}"
            )

        if not request.query or not request.query.strip():
            errors.append("Query string must not be empty.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute remote sensing inference and return standardized ToolResult."""
        # 1. Validate request
        val = self.validate_request(request)
        if not val.is_valid:
            logger.warning(f"Request validation failed in {self.name}: {val.errors}")
            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.FAILED,
                answer=f"Validation failed: {'; '.join(val.errors)}",
                confidence=None,
                evidence=[],
                artifacts=[],
                model_info=self.metadata.metadata,
                parameters={"query": request.query},
                metadata={"validation_errors": val.errors},
            )

        image_input = request.images[0]
        t0 = time.perf_counter()
        trace_entries: List[ExecutionTraceEntry] = []

        try:
            # 2. Route to appropriate inference mode
            if request.task == TaskType.SINGLE_IMAGE_GROUNDING:
                answer, raw_tokens, confidence, metrics = self.engine.run_grounding(
                    image_path=image_input.path_or_uri,
                    query=request.query,
                )
                evidence = GroundingCoordinateParser.parse_location_tokens(
                    raw_text=raw_tokens,
                    label=request.query,
                    image_id=image_input.image_id,
                    confidence=confidence,
                )
                # If parsed evidence, attach evidence trace
                if evidence:
                    trace_entries.append(
                        ExecutionTraceEntry(
                            stage=ExecutionStage.EVIDENCE_GENERATED,
                            component=self.name,
                            status="COMPLETED",
                            details={"bounding_box_count": len(evidence), "label": request.query},
                        )
                    )

            elif request.task == TaskType.SINGLE_IMAGE_CAPTION:
                answer, confidence, metrics = self.engine.run_captioning(
                    image_path=image_input.path_or_uri,
                )
                evidence = []

            else:  # TaskType.SINGLE_IMAGE_VQA
                answer, confidence, metrics = self.engine.run_vqa(
                    image_path=image_input.path_or_uri,
                    query=request.query,
                )
                # Ungrounded comparative/statistical VQA queries do not fabricate bounding boxes
                evidence = []

            total_dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                    duration_ms=total_dur_ms,
                    details={
                        "task": request.task.value,
                        "device": metrics.device_used,
                        "inference_ms": metrics.inference_time_ms,
                        "preprocessing_ms": metrics.preprocessing_time_ms,
                        "peak_memory_mb": metrics.peak_memory_mb,
                    },
                )
            )

            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.SUCCESS,
                answer=answer,
                confidence=confidence,
                evidence=evidence,
                artifacts=[],
                model_info={
                    "name": "PaliGemma 3B — SatQuery Remote-Sensing Adapted",
                    "base_model": self.engine.base_model_id,
                    "version": self.version,
                    "author": "Division 2 (Sruthi)",
                    "device": metrics.device_used,
                },
                parameters={
                    "query": request.query,
                    "image_id": image_input.image_id,
                    "modality": image_input.modality.value,
                },
                metadata={
                    "performance": {
                        "total_latency_ms": total_dur_ms,
                        "preprocessing_ms": metrics.preprocessing_time_ms,
                        "inference_ms": metrics.inference_time_ms,
                        "postprocessing_ms": metrics.postprocessing_time_ms,
                        "peak_memory_mb": metrics.peak_memory_mb,
                        "device": metrics.device_used,
                    }
                },
                execution_trace=trace_entries,
            )

        except Exception as e:
            logger.exception(f"Inference error in {self.name}: {e}")
            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.FAILED,
                answer=f"Specialist inference error: {str(e)}",
                confidence=None,
                evidence=[],
                artifacts=[],
                model_info=self.metadata.metadata,
                parameters={"query": request.query},
                metadata={"error": str(e)},
                execution_trace=[
                    ExecutionTraceEntry(
                        stage=ExecutionStage.ERROR_ENCOUNTERED,
                        component=self.name,
                        status="FAILED",
                        details={"error": str(e)},
                    )
                ],
            )

    def health_check(self) -> bool:
        """Check specialist health and device readiness."""
        try:
            return self.engine is not None
        except Exception:
            return False
