"""Division 2 Specialist: Single-Image Remote-Sensing Intelligence.

Implements BaseSpecialistTool for:
- Single-Image Visual Question Answering (VQA)
- Text-Guided Visual Grounding (Normalized Bounding Boxes)
- Scene Captioning & Land-Cover Description

Foundation Models:
- Primary Production Backend: Qwen2.5-VL (Qwen/Qwen2.5-VL-3B-Instruct)
- Preserved Rollback Backend: PaliGemma 3B (google/paligemma-3b-pt-224)
Controlled via environment variable: VISION_LANGUAGE_BACKEND (default: "qwen25vl")
"""

from __future__ import annotations

import os
from pathlib import Path
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
from specialists.single_image.adaptation.qwen25vl.grounding import QwenGroundingParser
from specialists.single_image.adaptation.qwen25vl.inference import QwenSingleImageEngine
from specialists.single_image.grounding import GroundingCoordinateParser
from specialists.single_image.model import PaliGemmaRSInferenceEngine

logger = get_logger("single_image_specialist")


class SingleImageRSSpecialistTool(BaseSpecialistTool):
    """Single-Image Remote-Sensing Intelligence Specialist supporting Qwen2.5-VL and PaliGemma."""

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.backend = (backend or os.getenv("VISION_LANGUAGE_BACKEND", "qwen25vl")).lower().strip()

        if self.backend == "qwen25vl":
            resolved_base_model = base_model_id or os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
            default_ad = Path("specialists/single_image/weights/qwen25vl_lora")
            resolved_adapter = adapter_path or (str(default_ad) if default_ad.exists() else None)
            desc = (
                "Single-Image Remote-Sensing Intelligence Specialist powered by Qwen2.5-VL — "
                "SatQuery Remote-Sensing Adapted. Supports VQA, text-guided visual grounding, "
                "and scene description across optical and SAR satellite imagery."
            )
            model_info_dict = {
                "base_architecture": "Qwen2.5-VL Vision-Language Transformer",
                "model_name": "Qwen2.5-VL-3B-Instruct",
                "backend": "qwen25vl",
                "base_checkpoint": resolved_base_model,
                "adaptation_method": "PEFT / QLoRA 4-bit (rank=16, alpha=32)",
                "coordinate_convention": "[ymin, xmin, ymax, xmax] (normalized 0.0 - 1.0)",
            }
        else:
            resolved_base_model = base_model_id or "google/paligemma-3b-pt-224"
            default_ad = Path("specialists/single_image/weights/satquery_paligemma_lora")
            resolved_adapter = adapter_path or (str(default_ad) if default_ad.exists() else None)
            desc = (
                "Single-Image Remote-Sensing Specialist powered by PaliGemma 3B — "
                "SatQuery Remote-Sensing Adapted. Supports VQA, text-guided visual grounding, "
                "and scene description across optical and multispectral satellite imagery."
            )
            model_info_dict = {
                "base_architecture": "SigLIP-So400m + Gemma-2B",
                "model_name": "PaliGemma 3B — SatQuery Remote-Sensing Adapted",
                "backend": "paligemma_legacy",
                "base_checkpoint": resolved_base_model,
                "adaptation_method": "PEFT / LoRA (rank=8, alpha=16)",
                "coordinate_convention": "[ymin, xmin, ymax, xmax] (normalized 0.0 - 1.0)",
            }

        metadata = ToolMetadata(
            name="single_image_rs_specialist",
            description=desc,
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
            metadata=model_info_dict,
        )
        super().__init__(
            name="single_image_rs_specialist",
            description=metadata.description,
            supported_tasks=set(metadata.supported_tasks),
            version=metadata.version,
            metadata=metadata,
        )

        # Initialize engines
        self.qwen_engine = QwenSingleImageEngine.get_instance(
            base_model_id=resolved_base_model if self.backend == "qwen25vl" else "Qwen/Qwen2.5-VL-3B-Instruct",
            adapter_path=resolved_adapter if self.backend == "qwen25vl" else None,
        )
        self.paligemma_engine = PaliGemmaRSInferenceEngine.get_instance(
            base_model_id=resolved_base_model if self.backend != "qwen25vl" else "google/paligemma-3b-pt-224",
            adapter_path=resolved_adapter if self.backend != "qwen25vl" else None,
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
            if self.backend == "qwen25vl":
                # --- QWEN2.5-VL PRIMARY INFERENCE PATH ---
                if request.task == TaskType.SINGLE_IMAGE_GROUNDING:
                    answer, evidence, confidence, metrics = self.qwen_engine.run_grounding(
                        image=image_input.path_or_uri,
                        query=request.query,
                        image_id=image_input.image_id,
                    )
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
                    answer, confidence, metrics = self.qwen_engine.run_captioning(
                        image=image_input.path_or_uri,
                    )
                    evidence = []
                else:  # SINGLE_IMAGE_VQA
                    answer, confidence, metrics = self.qwen_engine.run_vqa(
                        image=image_input.path_or_uri,
                        query=request.query,
                    )
                    evidence = []

                device_used = metrics.device_used
                inference_ms = metrics.inference_time_ms
                preprocessing_ms = metrics.preprocessing_time_ms
                peak_memory = metrics.peak_memory_mb
                active_model_name = "Qwen2.5-VL-3B-Instruct"

            else:
                # --- PALIGEMMA LEGACY ROLLBACK INFERENCE PATH ---
                if request.task == TaskType.SINGLE_IMAGE_GROUNDING:
                    answer, raw_tokens, confidence, metrics = self.paligemma_engine.run_grounding(
                        image_path=image_input.path_or_uri,
                        query=request.query,
                    )
                    evidence = GroundingCoordinateParser.parse_location_tokens(
                        raw_text=raw_tokens,
                        label=request.query,
                        image_id=image_input.image_id,
                        confidence=confidence,
                    )
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
                    answer, confidence, metrics = self.paligemma_engine.run_captioning(
                        image_path=image_input.path_or_uri,
                    )
                    evidence = []
                else:  # SINGLE_IMAGE_VQA
                    answer, confidence, metrics = self.paligemma_engine.run_vqa(
                        image_path=image_input.path_or_uri,
                        query=request.query,
                    )
                    evidence = []

                device_used = metrics.device_used
                inference_ms = metrics.inference_time_ms
                preprocessing_ms = metrics.preprocessing_time_ms
                peak_memory = metrics.peak_memory_mb
                active_model_name = "PaliGemma 3B — SatQuery Remote-Sensing Adapted"

            total_dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                    duration_ms=total_dur_ms,
                    details={
                        "task": request.task.value,
                        "backend": self.backend,
                        "device": device_used,
                        "inference_ms": inference_ms,
                        "preprocessing_ms": preprocessing_ms,
                        "peak_memory_mb": peak_memory,
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
                    "name": active_model_name,
                    "backend": self.backend,
                    "base_model": self.metadata.metadata.get("base_checkpoint"),
                    "version": self.version,
                    "author": self.metadata.author_or_division,
                    "device": device_used,
                    "is_mock": False,
                    "is_fallback": False,
                },
                parameters={
                    "query": request.query,
                    "image_id": image_input.image_id,
                    "modality": image_input.modality.value,
                },
                metadata={
                    "performance": {
                        "total_latency_ms": total_dur_ms,
                        "preprocessing_ms": preprocessing_ms,
                        "inference_ms": inference_ms,
                        "peak_memory_mb": peak_memory,
                        "device": device_used,
                    },
                    "backend": self.backend,
                    "is_mock": False,
                    "is_fallback": False,
                },
                execution_trace=trace_entries,
            )

        except Exception as e:
            logger.exception(f"Inference error in {self.name} ({self.backend}): {e}")
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
                metadata={"error": str(e), "backend": self.backend},
                execution_trace=[
                    ExecutionTraceEntry(
                        stage=ExecutionStage.ERROR_ENCOUNTERED,
                        component=self.name,
                        status="FAILED",
                        details={"error": str(e), "backend": self.backend},
                    )
                ],
            )

    def health_check(self) -> bool:
        """Check specialist health and device readiness."""
        try:
            if self.backend == "qwen25vl":
                return self.qwen_engine is not None
            return self.paligemma_engine is not None
        except Exception:
            return False
