"""Division 3: Bi-Temporal Change Intelligence Specialist.

Owner: Dheeraj
Implements BaseSpecialistTool contract for bi-temporal change detection,
change VQA, and spatial change analysis.

Architecture:
  T0 + T1 + Query
    -> 6-Point Validation
    -> Controlled Geospatial Alignment
    -> Pluggable ChangeModel
    -> Postprocessing / Localization
    -> Query Intent Classification
    -> Pluggable SemanticReasoner
    -> Evidence Packaging
    -> Canonical ToolResult
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.interfaces import BaseSpecialistTool, ValidationResult
from core.logging import get_logger
from core.schemas import (
    ExecutionStage,
    ExecutionTraceEntry,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

from specialists.temporal_change.config import TemporalChangeConfig
from specialists.temporal_change.evidence import generate_evidence
from specialists.temporal_change.interfaces import (
    ChangeModel,
    QueryIntent,
    SemanticReasoner,
)
from specialists.temporal_change.model_adapter import (
    ChangeFormerAdapter,
    MockChangeModel,
)
from specialists.temporal_change.postprocessing import postprocess_change_map
from specialists.temporal_change.preprocessing import preprocess_pair
from specialists.temporal_change.semantic_reasoning import (
    MockSemanticReasoner,
    SpatialMetricSynthesizer,
    classify_query_intent,
)
from specialists.temporal_change.validation import validate_pair

logger = get_logger("temporal_change.specialist")


class BiTemporalChangeSpecialistTool(BaseSpecialistTool):
    """Production specialist tool for bi-temporal remote sensing change analysis.

    Accepts pluggable ChangeModel and SemanticReasoner backends via
    dependency injection. Falls back to mock/default implementations
    when no explicit backends are provided.

    Strictly inherits from core.interfaces.BaseSpecialistTool.
    Consumes core.schemas.ToolRequest, returns core.schemas.ToolResult.
    """

    def __init__(
        self,
        change_model: Optional[ChangeModel] = None,
        semantic_reasoner: Optional[SemanticReasoner] = None,
        config: Optional[TemporalChangeConfig] = None,
    ) -> None:
        self._config = config or TemporalChangeConfig()

        # Select backends
        if change_model is not None:
            self._change_model = change_model
        elif self._config.use_mock_model:
            self._change_model = MockChangeModel()
        else:
            self._change_model = ChangeFormerAdapter(
                checkpoint_path=self._config.model_checkpoint_path,
                architecture=self._config.model_architecture,
            )

        if semantic_reasoner is not None:
            self._semantic_reasoner = semantic_reasoner
        elif self._config.use_mock_model:
            self._semantic_reasoner = MockSemanticReasoner()
        else:
            self._semantic_reasoner = SpatialMetricSynthesizer()

        self._initialized = False

        metadata = ToolMetadata(
            name="bitemporal_change_specialist",
            description=(
                "Production specialist for bi-temporal remote sensing change detection, "
                "spatial localization, and change-based visual question answering."
            ),
            version="1.0.0",
            supported_tasks=[TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA],
            required_modalities=[
                ImageModality.OPTICAL,
                ImageModality.MULTISPECTRAL,
            ],
            min_images=2,
            max_images=2,
            author_or_division="Division 3 (Dheeraj)",
            metadata={
                "change_model": self._change_model.model_name,
                "semantic_reasoner": type(self._semantic_reasoner).__name__,
            },
        )

        super().__init__(
            name="bitemporal_change_specialist",
            description=metadata.description,
            supported_tasks={TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA},
            version=metadata.version,
            metadata=metadata,
        )

    def _ensure_initialized(self) -> None:
        """Lazy initialization of model backends."""
        if not self._initialized:
            device = self._config.resolve_device()
            self._change_model.initialize(device=device)
            self._semantic_reasoner.initialize()
            self._config.ensure_dirs()
            self._initialized = True
            logger.info(
                f"BiTemporalChangeSpecialistTool initialized: "
                f"model={self._change_model.model_name}, device={device}"
            )

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        """Validate request against bi-temporal change analysis requirements."""
        errors: List[str] = []

        # Task validation
        if request.task not in self.supported_tasks:
            errors.append(
                f"Task '{request.task.value}' is not supported. "
                f"Supported: {[t.value for t in self.supported_tasks]}"
            )

        # Image count check
        if len(request.images) != 2:
            errors.append(
                f"Bi-temporal analysis requires exactly 2 images, received {len(request.images)}."
            )

        # Query check
        if not request.query or not request.query.strip():
            errors.append("Query string must not be empty.")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute the bi-temporal change analysis pipeline."""
        trace: List[ExecutionTraceEntry] = []
        t_start = time.perf_counter()

        # --- Stage 1: Request validation ---
        val = self.validate_request(request)
        if not val.is_valid:
            return self._build_failure_result(
                request, "; ".join(val.errors), trace, "VALIDATION_FAILED"
            )

        trace.append(ExecutionTraceEntry(
            stage=ExecutionStage.INPUT_VALIDATED,
            component=self.name,
            status="COMPLETED",
            details={"image_count": len(request.images)},
        ))

        # --- Stage 2: 6-point pair validation ---
        pair_result = validate_pair(
            request,
            allow_reprojection=self._config.allow_geospatial_reprojection,
        )

        trace.append(ExecutionTraceEntry(
            stage=ExecutionStage.INPUT_VALIDATED,
            component=f"{self.name}.pair_validation",
            status="COMPLETED" if pair_result.is_valid else "FAILED",
            details={
                "alignment_status": pair_result.alignment_status,
                "stages_passed": sum(1 for s in pair_result.stages if s.passed),
                "stages_total": len(pair_result.stages),
                "errors": pair_result.errors,
            },
        ))

        if not pair_result.is_valid:
            return self._build_failure_result(
                request,
                f"Pair validation failed: {'; '.join(pair_result.errors)}",
                trace,
                "PAIR_VALIDATION_FAILED",
            )

        t0_img, t1_img = request.images[0], request.images[1]

        # --- Stage 3: Preprocessing ---
        try:
            self._ensure_initialized()

            t0_arr, t1_arr, preproc_meta = preprocess_pair(
                t0_path=t0_img.path_or_uri,
                t1_path=t1_img.path_or_uri,
                model_input_size=self._config.model_input_size,
                normalize=self._config.normalize_to_float,
            )

            trace.append(ExecutionTraceEntry(
                stage=ExecutionStage.MODEL_INITIALIZED,
                component=f"{self.name}.preprocessing",
                status="COMPLETED",
                duration_ms=preproc_meta.get("preprocessing_time_ms"),
                details=preproc_meta,
            ))
        except Exception as e:
            return self._build_failure_result(
                request, f"Preprocessing error: {e}", trace, "PREPROCESSING_FAILED"
            )

        # --- Stage 4: Change detection inference ---
        try:
            change_output = self._change_model.detect_change(t0_arr, t1_arr)

            trace.append(ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component=f"{self.name}.change_model",
                status="COMPLETED",
                duration_ms=change_output.inference_time_ms,
                details={
                    "model": change_output.model_name,
                    "device": change_output.device_used,
                    "changed_pixel_ratio": change_output.changed_pixel_ratio,
                },
            ))
        except Exception as e:
            return self._build_failure_result(
                request, f"Change detection inference error: {e}", trace, "INFERENCE_FAILED"
            )

        # --- Stage 5: Postprocessing ---
        try:
            postproc = postprocess_change_map(
                prob_map=change_output.change_probability_map,
                binary_map=change_output.binary_change_map,
                threshold=self._config.change_threshold,
                min_region_area=self._config.min_region_area_pixels,
                max_regions=self._config.max_regions_reported,
                morphology_kernel=self._config.morphology_kernel_size,
            )

            # Attach regions to change output for semantic reasoning
            change_output.changed_regions = postproc.regions

            trace.append(ExecutionTraceEntry(
                stage=ExecutionStage.EVIDENCE_GENERATED,
                component=f"{self.name}.postprocessing",
                status="COMPLETED",
                details={
                    "regions_found": len(postproc.regions),
                    "changed_pixel_ratio": round(postproc.changed_pixel_ratio, 4),
                    "parameters": postproc.parameters,
                },
            ))
        except Exception as e:
            return self._build_failure_result(
                request, f"Postprocessing error: {e}", trace, "POSTPROCESSING_FAILED"
            )

        # --- Stage 6: Query intent classification ---
        query_intent = classify_query_intent(request.query)

        # --- Stage 7: Semantic reasoning ---
        try:
            semantic_output = self._semantic_reasoner.reason_about_change(
                query=request.query,
                change_output=change_output,
                query_intent=query_intent,
                t0=t0_arr,
                t1=t1_arr,
            )

            trace.append(ExecutionTraceEntry(
                stage=ExecutionStage.RESULT_AGGREGATED,
                component=f"{self.name}.semantic_reasoning",
                status="COMPLETED",
                details={
                    "query_intent": query_intent.value,
                    "capabilities": [c.value for c in semantic_output.supported_capabilities],
                    "has_limitations": len(semantic_output.limitations) > 0,
                },
            ))
        except Exception as e:
            # Semantic reasoning failure is non-fatal — fall back to basic answer
            logger.warning(f"Semantic reasoning failed, falling back: {e}")
            semantic_output = None

        # --- Stage 8: Evidence generation ---
        try:
            evidence, artifacts = generate_evidence(
                change_output=change_output,
                postproc_result=postproc,
                t0_image_id=t0_img.image_id,
                t1_image_id=t1_img.image_id,
                output_dir=self._config.artifact_output_dir,
                request_id=request.request_id,
            )

            trace.append(ExecutionTraceEntry(
                stage=ExecutionStage.EVIDENCE_GENERATED,
                component=f"{self.name}.evidence",
                status="COMPLETED",
                details={
                    "evidence_count": len(evidence),
                    "artifact_count": len(artifacts),
                },
            ))
        except Exception as e:
            logger.warning(f"Evidence generation failed: {e}")
            evidence = []
            artifacts = []

        # --- Build result ---
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if semantic_output:
            answer = semantic_output.answer
        else:
            pct = round(postproc.changed_pixel_ratio * 100, 1)
            answer = (
                f"Change detection completed. Approximately {pct}% of the scene "
                f"has changed, with {len(postproc.regions)} distinct region(s) detected."
            )

        # Confidence breakdown (Rule B: no fabricated combined score)
        change_confidence = change_output.change_confidence
        semantic_confidence = semantic_output.semantic_confidence if semantic_output else None
        # overall_confidence: only populated if defensible; otherwise None
        overall_confidence = None

        # Top-level confidence follows the primary task
        if request.task == TaskType.CHANGE_ANALYSIS:
            top_confidence = change_confidence
        elif request.task == TaskType.CHANGE_VQA:
            top_confidence = semantic_confidence  # May be None
        else:
            top_confidence = change_confidence

        trace.append(ExecutionTraceEntry(
            stage=ExecutionStage.RESULT_RETURNED,
            component=self.name,
            status="COMPLETED",
            duration_ms=total_ms,
        ))

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=top_confidence,
            evidence=evidence,
            artifacts=artifacts,
            model_info={
                "change_model": change_output.model_name,
                "change_model_version": change_output.model_version,
                "semantic_reasoner": type(self._semantic_reasoner).__name__,
                "device": change_output.device_used,
            },
            parameters={
                "change_threshold": self._config.change_threshold,
                "min_region_area": self._config.min_region_area_pixels,
                "morphology_kernel": self._config.morphology_kernel_size,
                "model_input_size": self._config.model_input_size,
                "allow_geospatial_reprojection": self._config.allow_geospatial_reprojection,
            },
            metadata={
                "confidence_breakdown": {
                    "change_confidence": change_confidence,
                    "semantic_confidence": semantic_confidence,
                    "overall_confidence": overall_confidence,
                },
                "alignment_status": pair_result.alignment_status,
                "query_intent": query_intent.value,
                "limitations": semantic_output.limitations if semantic_output else [],
                "t0_info": pair_result.t0_info,
                "t1_info": pair_result.t1_info,
            },
            execution_trace=trace,
        )

    def _build_failure_result(
        self,
        request: ToolRequest,
        error_message: str,
        trace: List[ExecutionTraceEntry],
        error_stage: str,
    ) -> ToolResult:
        """Build a standardized failure ToolResult."""
        trace.append(ExecutionTraceEntry(
            stage=ExecutionStage.ERROR_ENCOUNTERED,
            component=self.name,
            status="FAILED",
            details={"error": error_message, "stage": error_stage},
        ))

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.FAILED,
            answer=f"Change analysis failed: {error_message}",
            confidence=None,
            evidence=[],
            artifacts=[],
            model_info={"change_model": self._change_model.model_name},
            parameters={},
            metadata={"error_stage": error_stage},
            execution_trace=trace,
        )

    def health_check(self) -> bool:
        """Check if model backends are ready."""
        try:
            if not self._initialized:
                return True  # Not yet initialized is OK (lazy init)
            return self._change_model.is_ready() and self._semantic_reasoner.is_ready()
        except Exception:
            return False
