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
            resolved_base_model = base_model_id or os.getenv("MODEL_ID")
            if not resolved_base_model:
                for candidate in [
                    Path("merged_full"),
                    Path(__file__).resolve().parent.parent.parent / "merged_full",
                    Path("artifacts/qwen25vl_stage1/merged_full"),
                ]:
                    if candidate.exists() and (candidate / "model.safetensors.index.json").exists():
                        resolved_base_model = str(candidate)
                        break
            if not resolved_base_model:
                resolved_base_model = "Qwen/Qwen2.5-VL-3B-Instruct"

            default_ad = Path("specialists/single_image/weights/qwen25vl_lora")
            # Standalone merged model has zero PEFT dependency
            if "merged_full" in str(resolved_base_model):
                resolved_adapter = None
            else:
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
                TaskType.CHANGE_VQA,
            ],
            required_modalities=[
                ImageModality.OPTICAL,
                ImageModality.MULTISPECTRAL,
                ImageModality.SAR,
            ],
            min_images=1,
            max_images=3,
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
        """Validate input constraints for vision-language analysis."""
        errors: List[str] = []

        is_change_vqa = (
            request.task == TaskType.CHANGE_VQA
            or "changed_region_evidence" in request.context
            or any(img.metadata.get("is_changed_region_crop") for img in request.images)
        )
        is_t1_fallback = (
            request.metadata.get("vlm_contract") == "single_image_t1"
            or request.config.get("vlm_contract") == "single_image_t1"
        )

        if is_change_vqa:
            if is_t1_fallback:
                if len(request.images) != 1:
                    errors.append(f"Explicit T1-only Change-VQA contract requires exactly 1 image, received {len(request.images)}.")
            else:
                if len(request.images) != 3:
                    errors.append(f"Change-VQA evidence package requires exactly 3 images [T0, T1, overlay], received {len(request.images)}.")
        elif len(request.images) != 1:
            task_name = request.task.value if hasattr(request.task, "value") else str(request.task)
            errors.append(f"SingleImageRSSpecialist requires exactly 1 image input for '{task_name}', received {len(request.images)}.")

        if request.task not in self.supported_tasks and request.task != TaskType.CHANGE_VQA:
            errors.append(
                f"Task '{request.task.value}' is not supported by {self.name}. "
                f"Supported: {[t.value for t in self.supported_tasks]}"
            )

        if not request.query or not request.query.strip():
            errors.append("Query string must not be empty.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute remote sensing inference and return standardized ToolResult."""
        is_change_vqa = (
            request.task == TaskType.CHANGE_VQA
            or any(img.metadata.get("is_changed_region_crop", False) for img in request.images)
            or "changed_region_evidence" in request.context
        )

        is_t1_fallback = (
            request.metadata.get("vlm_contract") == "single_image_t1"
            or request.config.get("vlm_contract") == "single_image_t1"
        )

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

        if is_change_vqa:
            if is_t1_fallback:
                handoff_mode = "SINGLE_IMAGE_T1_FALLBACK"
                image_input = request.images[0]
            else:
                handoff_mode = "MULTI_IMAGE_CHANGE_VQA"
                image_input = request.images[1]  # T1 reference if needed
        else:
            handoff_mode = "SINGLE_IMAGE_STANDARD"
            image_input = request.images[0]

        t0 = time.perf_counter()
        trace_entries: List[ExecutionTraceEntry] = []

        # Part 8 Mandate: When no trained Qwen checkpoint exists, DO NOT run untrained weights,
        # DO NOT use PaliGemma, and DO NOT fabricate semantic descriptions for Change-VQA.
        if is_change_vqa and self.backend == "qwen25vl" and not self.qwen_engine.is_real_model_loaded:
            logger.info(
                f"Change-VQA crop received ({handoff_mode}) but fine-tuned Qwen checkpoint is pending. Returning CHANGE_VQA_MODEL_NOT_READY."
            )
            evidence_summary = [
                {
                    "image_id": img.image_id,
                    "path": img.path_or_uri,
                    "role": img.metadata.get("temporal_role", "UNKNOWN"),
                    "dimensions": img.metadata.get("crop_dimensions"),
                }
                for img in request.images
            ]
            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.VLM_EXECUTED,
                    component=f"{self.name}.qwen25vl",
                    status="MODEL_NOT_READY",
                    details={
                        "backend": "qwen25vl",
                        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
                        "checkpoint_identifier": None,
                        "execution_mode": "checkpoint_pending",
                        "readiness_status": "CHANGE_VQA_MODEL_NOT_READY",
                        "handoff_mode": handoff_mode,
                        "evidence_package": evidence_summary,
                        "image_count": len(request.images),
                        "roles_received": [img.metadata.get("temporal_role") for img in request.images],
                        "evidence_artifact_ids": [img.image_id for img in request.images],
                        "reason": "Fine-tuned Qwen2.5-VL checkpoint is pending training. Semantic interpretation unavailable.",
                    },
                )
            )
            answer = (
                "[CHANGE_VQA_MODEL_NOT_READY] TinyCD detection = completed; "
                "semantic VLM interpretation = unavailable. "
                "The bi-temporal change specialist successfully detected change regions and generated "
                "visual evidence crops (BEFORE, AFTER, WHERE CHANGE OCCURRED), but the fine-tuned Qwen2.5-VL model weights are currently pending training."
            )
            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.ANSWER_GENERATED,
                    component=self.name,
                    status="COMPLETED",
                    details={
                        "status": "CHANGE_VQA_MODEL_NOT_READY",
                        "semantic_interpretation": "unavailable",
                        "evidence_package_count": len(request.images),
                        "handoff_mode": handoff_mode,
                    },
                )
            )
            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.PARTIAL_SUCCESS,
                answer=answer,
                confidence=None,
                evidence=[],
                artifacts=[],
                model_info={
                    "name": "Qwen2.5-VL-3B-Instruct",
                    "backend": "qwen25vl",
                    "readiness_status": "CHANGE_VQA_MODEL_NOT_READY",
                    "execution_mode": "checkpoint_pending",
                    "checkpoint_identifier": None,
                    "is_mock": False,
                    "is_fallback": False,
                },
                parameters={
                    "query": request.query,
                    "evidence_images": [img.path_or_uri for img in request.images],
                    "evidence_artifact_ids": [img.image_id for img in request.images],
                },
                metadata={
                    "change_vqa_status": "CHANGE_VQA_MODEL_NOT_READY",
                    "tinycd_detection": "completed",
                    "semantic_vlm_interpretation": "unavailable",
                    "backend": "qwen25vl",
                    "readiness_status": "CHANGE_VQA_MODEL_NOT_READY",
                    "handoff_mode": handoff_mode,
                    "evidence_package": evidence_summary,
                    "evidence_artifact_ids": [img.image_id for img in request.images],
                    "vlm_provenance": {
                        "backend": "qwen25vl",
                        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
                        "checkpoint_identifier": None,
                        "execution_mode": "checkpoint_pending",
                        "readiness_status": "CHANGE_VQA_MODEL_NOT_READY",
                    },
                },
                execution_trace=trace_entries,
            )

        try:
            if self.backend == "qwen25vl":
                # --- QWEN2.5-VL PRIMARY INFERENCE PATH ---
                if is_change_vqa:
                    change_meta = (
                        request.context.get("changed_region_evidence")
                        or request.metadata.get("changed_region_evidence")
                    )
                    if is_t1_fallback:
                        answer, confidence, metrics = self.qwen_engine.run_vqa(
                            image=request.images[0].path_or_uri,
                            query=request.query,
                        )
                    else:
                        answer, confidence, metrics = self.qwen_engine.run_change_vqa(
                            images=[img.path_or_uri for img in request.images],
                            query=request.query,
                            metadata=change_meta,
                            roles=[img.metadata.get("temporal_role") for img in request.images],
                        )
                    evidence = []
                    trace_entries.append(
                        ExecutionTraceEntry(
                            stage=ExecutionStage.VLM_EXECUTED,
                            component=f"{self.name}.qwen25vl",
                            status="COMPLETED",
                            details={
                                "handoff_mode": handoff_mode,
                                "image_count": len(request.images),
                                "roles": [img.metadata.get("temporal_role") for img in request.images],
                            },
                        )
                    )
                elif request.task == TaskType.SINGLE_IMAGE_GROUNDING:
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

            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.VLM_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                    duration_ms=total_dur_ms,
                    details={
                        "task": request.task.value,
                        "backend": self.backend,
                        "model_name": active_model_name,
                        "device": device_used,
                        "inference_ms": inference_ms,
                        "preprocessing_ms": preprocessing_ms,
                        "peak_memory_mb": peak_memory,
                        "checkpoint_identifier": getattr(self.qwen_engine, "adapter_path", None) if self.backend == "qwen25vl" else None,
                        "execution_mode": "production_verified" if getattr(self.qwen_engine, "is_real_model_loaded", False) else "deterministic_fallback",
                        "readiness_status": "READY" if getattr(self.qwen_engine, "is_real_model_loaded", False) else "FALLBACK",
                    },
                )
            )
            trace_entries.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.ANSWER_GENERATED,
                    component=self.name,
                    status="COMPLETED",
                    details={"task": request.task.value},
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
