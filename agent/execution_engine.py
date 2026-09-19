"""Sequential Execution Engine for specialist tool workflows.

Executes workflow steps sequentially, passes context across stages,
enforces timeouts, and maintains an auditable operational execution trace.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from core.errors import (
    InferenceError,
    InputValidationError,
    OutputValidationError,
    SatQueryException,
    ToolNotFoundError,
    ToolTimeoutError,
)
from core.logging import get_logger
from core.schemas import (
    ExecutionStage,
    ExecutionTraceEntry,
    ImageInput,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from registry.registry import ToolRegistry, default_registry
from agent.workflow import WorkflowPlan, WorkflowStep

logger = get_logger("execution_engine")


class ExecutionEngine:
    """Orchestrates asynchronous execution of single or sequential tool workflows."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        default_timeout_seconds: float = 30.0,
    ) -> None:
        self.registry = registry or default_registry
        self.default_timeout_seconds = default_timeout_seconds

    async def execute_plan(
        self,
        plan: WorkflowPlan,
        base_request: ToolRequest,
        timeout_seconds: Optional[float] = None,
    ) -> List[ToolResult]:
        """Execute a planned workflow and return all intermediate and final ToolResults."""
        timeout = timeout_seconds or self.default_timeout_seconds
        results: List[ToolResult] = []
        cumulative_context: Dict[str, Any] = dict(base_request.context)

        logger.info(f"Executing workflow plan '{plan.plan_id}' with {len(plan.steps)} step(s)")

        for step in plan.steps:
            step_result = await self._execute_step(
                step=step,
                base_request=base_request,
                context=cumulative_context,
                timeout_seconds=timeout,
            )
            results.append(step_result)

            if step_result.status == ToolStatus.FAILED:
                logger.warning(f"Workflow step {step.step_index} ('{step.tool_name}') failed; halting subsequent steps.")
                break

            # Update context for subsequent steps
            step_meta = dict(step_result.metadata) if step_result.metadata else {}
            cumulative_context[f"step_{step.step_index}_result"] = {
                "task": step_result.task.value,
                "tool_name": step.tool_name,
                "answer": step_result.answer,
                "evidence_count": len(step_result.evidence),
                "artifacts": [a.model_dump() for a in step_result.artifacts],
                "metadata": step_meta,
            }
            if "changed_region_evidence" in step_meta:
                cumulative_context["changed_region_evidence"] = step_meta["changed_region_evidence"]

        return results

    async def _execute_step(
        self,
        step: WorkflowStep,
        base_request: ToolRequest,
        context: Dict[str, Any],
        timeout_seconds: float,
    ) -> ToolResult:
        """Execute a single workflow step with timeout protection and trace recording."""
        tool = self.registry.get(step.tool_name)
        
        # Prepare step-specific request
        step_images = base_request.images
        meta = getattr(tool, "metadata", None)
        vlm_prep_trace: Optional[ExecutionTraceEntry] = None

        # Check for visual handoff from change detection to VLM
        if step.pass_context_from_previous and "changed_region_evidence" in context:
            cre_data = context["changed_region_evidence"]
            crop_t1_path = cre_data.get("cropped_t1_path")
            crop_t0_path = cre_data.get("cropped_t0_path")
            overlay_path = cre_data.get("visualization_overlay_path")
            reg_id = cre_data.get("selected_region_id", 1)

            # Contract check: Preserve T1-only path as a documented fallback if multi-image is disabled
            force_single_image_t1 = (
                context.get("force_t1_only_vlm", False)
                or base_request.metadata.get("vlm_contract") == "single_image_t1"
                or base_request.config.get("vlm_contract") == "single_image_t1"
                or (not crop_t0_path or not Path(crop_t0_path).exists())
            )

            from core.schemas import ImageFormat, ImageModality

            if force_single_image_t1 and crop_t1_path and Path(crop_t1_path).exists():
                logger.info(f"Visual handoff to {step.tool_name}: routing T1-only crop (single-image contract enforced)")
                t1_img_id = f"crop_t1_after_r{reg_id}"
                step_images = [
                    ImageInput(
                        image_id=t1_img_id,
                        path_or_uri=str(crop_t1_path),
                        format=ImageFormat.PNG,
                        modality=ImageModality.OPTICAL,
                        metadata={
                            "is_changed_region_crop": True,
                            "temporal_role": "AFTER",
                            "acquisition_phase": "T1",
                            "source_t0": cre_data.get("source_t0_path"),
                            "source_t1": cre_data.get("source_t1_path"),
                            "selected_region_id": reg_id,
                            "selected_region_pixel_bbox": cre_data.get("selected_region_pixel_bbox"),
                            "selected_region_normalized_bbox": cre_data.get("selected_region_normalized_bbox"),
                            "area_pixels": cre_data.get("area_pixels"),
                            "has_change": cre_data.get("has_change", True),
                            "fallback_reason": cre_data.get("fallback_reason") or "SINGLE_IMAGE_CONTRACT_ENFORCED: T1-only crop routed.",
                        },
                    )
                ]
                vlm_prep_trace = ExecutionTraceEntry(
                    stage=ExecutionStage.VLM_INPUT_PREPARED,
                    component="ExecutionEngine.visual_handoff",
                    status="COMPLETED",
                    details={
                        "target_tool": step.tool_name,
                        "handoff_mode": "SINGLE_IMAGE_T1_FALLBACK",
                        "evidence_artifact_ids": [t1_img_id],
                        "temporal_roles": ["AFTER"],
                        "crop_image_path": str(crop_t1_path),
                        "selected_region_id": reg_id,
                        "area_pixels": cre_data.get("area_pixels"),
                        "crop_dimensions": cre_data.get("crop_dimensions"),
                        "has_change": cre_data.get("has_change", True),
                        "fallback_reason": "SINGLE_IMAGE_CONTRACT_ENFORCED: T1-only visual patch routed.",
                        "source_t1": cre_data.get("source_t1_path"),
                        "source_t0": cre_data.get("source_t0_path"),
                        "query": base_request.query,
                    },
                )
            elif crop_t1_path and Path(crop_t1_path).exists():
                logger.info(f"Visual handoff to {step.tool_name}: packaging multi-image Change-VQA evidence (BEFORE, AFTER, WHERE_CHANGE_OCCURRED)")
                t0_img_id = f"crop_t0_before_r{reg_id}"
                t1_img_id = f"crop_t1_after_r{reg_id}"
                overlay_img_id = f"crop_overlay_change_r{reg_id}"

                img_t0 = ImageInput(
                    image_id=t0_img_id,
                    path_or_uri=str(crop_t0_path),
                    format=ImageFormat.PNG,
                    modality=ImageModality.OPTICAL,
                    metadata={
                        "is_changed_region_crop": True,
                        "temporal_role": "BEFORE",
                        "acquisition_phase": "T0",
                        "source_t0": cre_data.get("source_t0_path"),
                        "source_t1": cre_data.get("source_t1_path"),
                        "selected_region_id": reg_id,
                        "selected_region_pixel_bbox": cre_data.get("selected_region_pixel_bbox"),
                        "crop_dimensions": cre_data.get("crop_dimensions"),
                    },
                )
                img_t1 = ImageInput(
                    image_id=t1_img_id,
                    path_or_uri=str(crop_t1_path),
                    format=ImageFormat.PNG,
                    modality=ImageModality.OPTICAL,
                    metadata={
                        "is_changed_region_crop": True,
                        "temporal_role": "AFTER",
                        "acquisition_phase": "T1",
                        "source_t0": cre_data.get("source_t0_path"),
                        "source_t1": cre_data.get("source_t1_path"),
                        "selected_region_id": reg_id,
                        "selected_region_pixel_bbox": cre_data.get("selected_region_pixel_bbox"),
                        "crop_dimensions": cre_data.get("crop_dimensions"),
                    },
                )
                img_overlay = ImageInput(
                    image_id=overlay_img_id,
                    path_or_uri=str(overlay_path),
                    format=ImageFormat.PNG,
                    modality=ImageModality.OPTICAL,
                    metadata={
                        "is_changed_region_crop": True,
                        "temporal_role": "WHERE_CHANGE_OCCURRED",
                        "acquisition_phase": "CHANGE_MASK_OVERLAY",
                        "source_t0": cre_data.get("source_t0_path"),
                        "source_t1": cre_data.get("source_t1_path"),
                        "selected_region_id": reg_id,
                        "selected_region_pixel_bbox": cre_data.get("selected_region_pixel_bbox"),
                        "crop_dimensions": cre_data.get("crop_dimensions"),
                    },
                )
                step_images = [img_t0, img_t1, img_overlay]

                vlm_prep_trace = ExecutionTraceEntry(
                    stage=ExecutionStage.VLM_INPUT_PREPARED,
                    component="ExecutionEngine.visual_handoff",
                    status="COMPLETED",
                    details={
                        "target_tool": step.tool_name,
                        "handoff_mode": "MULTI_IMAGE_CHANGE_VQA",
                        "evidence_artifact_ids": [t0_img_id, t1_img_id, overlay_img_id],
                        "temporal_roles": ["BEFORE", "AFTER", "WHERE_CHANGE_OCCURRED"],
                        "crop_image_path": str(crop_t1_path),
                        "crop_t0_path": str(crop_t0_path),
                        "crop_t1_path": str(crop_t1_path),
                        "overlay_path": str(overlay_path),
                        "selected_region_id": reg_id,
                        "area_pixels": cre_data.get("area_pixels"),
                        "area_fraction": cre_data.get("area_fraction"),
                        "crop_dimensions": cre_data.get("crop_dimensions"),
                        "has_change": cre_data.get("has_change", True),
                        "fallback_reason": cre_data.get("fallback_reason"),
                        "source_t0": cre_data.get("source_t0_path"),
                        "source_t1": cre_data.get("source_t1_path"),
                        "query": base_request.query,
                        "deterministic_change_metadata": {
                            "selected_region_pixel_bbox": cre_data.get("selected_region_pixel_bbox"),
                            "selected_region_normalized_bbox": cre_data.get("selected_region_normalized_bbox"),
                            "tinycd_threshold": cre_data.get("tinycd_threshold", 0.5),
                            "temporal_ordering": cre_data.get("temporal_ordering", "T0->T1"),
                        },
                    },
                )
        elif meta and meta.max_images == 1 and len(step_images) > 1:
            step_images = [step_images[-1]]  # Target post-change / latest acquisition image

        step_request = ToolRequest(
            request_id=base_request.request_id,
            task=step.task,
            query=base_request.query,
            images=step_images,
            metadata=base_request.metadata,
            config=base_request.config,
            context=context if step.pass_context_from_previous else {},
        )

        # 1. Pre-execution Tool Validation
        val_res = tool.validate_request(step_request)
        if not val_res.is_valid:
            error_msg = "; ".join(val_res.errors)
            logger.error(f"Validation failed for tool '{tool.name}': {error_msg}")
            raise InputValidationError(f"Tool '{tool.name}' rejected request: {error_msg}")

        # 2. Asynchronous Execution with Timeout
        start_time = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                tool.execute(step_request),
                timeout=timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Validate result integrity
            if not isinstance(result, ToolResult):
                raise OutputValidationError(
                    f"Tool '{tool.name}' returned invalid result type: {type(result)}. Expected ToolResult."
                )

            # Insert VLM input preparation trace entry before tool trace if applicable
            if vlm_prep_trace:
                result.execution_trace.insert(0, vlm_prep_trace)
            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Tool '{tool.name}' timed out after {timeout_seconds}s")
            raise ToolTimeoutError(
                f"Execution of specialist tool '{tool.name}' timed out after {timeout_seconds}s",
                details={"step_index": step.step_index, "duration_ms": elapsed_ms},
            )
        except SatQueryException:
            # Re-raise known domain exceptions
            raise
        except Exception as err:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(f"Unhandled exception during execution of tool '{tool.name}': {err}")
            raise InferenceError(
                f"Inference error in specialist tool '{tool.name}': {str(err)}",
                details={"step_index": step.step_index, "duration_ms": elapsed_ms},
            )
