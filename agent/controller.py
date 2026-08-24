"""Agent Controller for SatQuery AI (Division 1 Core Orchestration).

Coordinates the end-to-end vision-language pipeline:
Input Validation -> Intent Resolution -> TaskPlan Generation -> Specialist Execution -> Result Aggregation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import List, Optional
import uuid

from core.errors import SatQueryException
from core.logging import get_logger, set_request_id
from core.schemas import (
    AgentDecision,
    ExecutionStage,
    ExecutionTraceEntry,
    QueryRequest,
    QueryResponse,
    SatQueryErrorDetail,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from registry.registry import ToolRegistry, default_registry
from validation.validator import InputValidator
from agent.aggregator import ResultAggregator
from agent.execution_engine import ExecutionEngine
from agent.intent_resolver import IntentResolver
from agent.router import TaskRouter
from agent.workflow import WorkflowPlanner

logger = get_logger("controller")


class AgentController:
    """Master orchestrator for SatQuery AI."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.registry = registry or default_registry
        self.router = TaskRouter(self.registry)
        self.execution_engine = ExecutionEngine(self.registry, default_timeout_seconds=timeout_seconds)

    async def process_query(
        self,
        request: QueryRequest,
        request_id: Optional[str] = None,
    ) -> QueryResponse:
        """Process a user query through the full agentic pipeline."""
        req_id = request_id or str(uuid.uuid4())
        set_request_id(req_id)
        system_trace: List[ExecutionTraceEntry] = []
        resolved_task = request.task_hint or TaskType.SINGLE_IMAGE_VQA

        logger.info(f"Received query request [{req_id}]: '{request.query}' with {len(request.images)} image(s)")

        # 1. Trace: REQUEST_RECEIVED
        system_trace.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.REQUEST_RECEIVED,
                component="AgentController",
                status="COMPLETED",
                details={"image_count": len(request.images), "has_task_hint": bool(request.task_hint)},
            )
        )

        try:
            # 2. Query Interpretation & Intent Resolution
            t0 = time.perf_counter()
            intent = IntentResolver.resolve_intent(
                query=request.query,
                images=request.images,
                task_hint=request.task_hint,
            )
            resolved_task = intent.task
            dur_intent = round((time.perf_counter() - t0) * 1000.0, 2)
            
            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.TASK_RESOLVED,
                    component="IntentResolver",
                    status="COMPLETED",
                    duration_ms=dur_intent,
                    details={
                        "resolved_task": intent.task.value,
                        "confidence": intent.confidence,
                        "intent_explanation": intent.intent_explanation,
                        "target_features": intent.target_features,
                    },
                )
            )

            # 3. Input Validation against Resolved Task
            t0 = time.perf_counter()
            InputValidator.validate_task_compatibility(
                task=resolved_task,
                images=request.images,
                inspect_disk=False,  # Disk inspection happens during upload/input ingestion
            )
            dur_val = round((time.perf_counter() - t0) * 1000.0, 2)

            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.INPUT_VALIDATED,
                    component="InputValidator",
                    status="COMPLETED",
                    duration_ms=dur_val,
                    details={"task": resolved_task.value, "validated_images": len(request.images)},
                )
            )

            # 4. Tool Selection via Router
            t0 = time.perf_counter()
            selected_tool = self.router.select_tool(intent=intent, images=request.images)
            
            # Check if multi-step composite query
            is_composite = intent.extracted_parameters.get("is_composite", False)
            secondary_tool = None
            if is_composite:
                secondary_task = intent.extracted_parameters.get("secondary_task", TaskType.SINGLE_IMAGE_VQA)
                sec_candidates = self.registry.find_tools_for_task(secondary_task)
                if sec_candidates:
                    secondary_tool = sec_candidates[0]

            dur_router = round((time.perf_counter() - t0) * 1000.0, 2)
            selected_tool_names = [selected_tool.name] + ([secondary_tool.name] if secondary_tool else [])

            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.TOOL_SELECTED,
                    component="TaskRouter",
                    status="COMPLETED",
                    duration_ms=dur_router,
                    details={"selected_tools": selected_tool_names, "is_composite": is_composite},
                )
            )

            # 5. Canonical TaskPlan Formulation & Workflow Derivation
            task_plan = WorkflowPlanner.create_task_plan(
                intent=intent,
                selected_tool=selected_tool,
                secondary_tool=secondary_tool,
                is_composite_query=is_composite,
                images=request.images,
            )
            workflow_plan = WorkflowPlanner.derive_workflow_plan(task_plan, intent)

            # Construct AgentDecision Card
            agent_decision = AgentDecision(
                task=resolved_task,
                task_display_name=resolved_task.value.replace("_", " ").title(),
                image_count=len(request.images),
                detected_modalities=[img.modality for img in request.images],
                selected_specialist=" -> ".join(selected_tool_names),
                workflow_summary=" -> ".join([s.tool_name for s in task_plan.steps]),
                confidence=intent.confidence,
                why_this_tool=intent.intent_explanation,
            )

            # 6. Specialist Execution
            base_tool_request = ToolRequest(
                request_id=req_id,
                task=resolved_task,
                query=request.query,
                images=request.images,
                metadata={"user_config": request.config},
                config=request.config,
            )

            # Mark steps as running/completed during execution
            for step in task_plan.steps:
                step.status = "running"

            tool_results = await self.execution_engine.execute_plan(
                plan=workflow_plan,
                base_request=base_tool_request,
            )

            for idx, res in enumerate(tool_results):
                if idx < len(task_plan.steps):
                    task_plan.steps[idx].status = "completed" if res.status == ToolStatus.SUCCESS else "failed"

            # 7. Result Aggregation
            response = ResultAggregator.aggregate(
                request_id=req_id,
                query=request.query,
                resolved_task=resolved_task,
                tool_results=tool_results,
                system_trace_entries=system_trace,
                task_intent=intent,
                task_plan=task_plan,
                agent_decision=agent_decision,
                selected_tools=selected_tool_names,
            )

            response.execution_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.RESULT_RETURNED,
                    component="AgentController",
                    status="COMPLETED",
                    details={"status": response.status.value},
                )
            )
            logger.info(f"Query [{req_id}] completed successfully with status: {response.status.value}")
            return response

        except SatQueryException as err:
            logger.error(f"SatQueryException encountered while processing [{req_id}]: {err.message}")
            err_detail = err.to_error_detail(request_id_override=req_id)
            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.ERROR_ENCOUNTERED,
                    component="AgentController",
                    status="FAILED",
                    details={"error_code": err.error_code.value, "message": err.message},
                )
            )
            return QueryResponse(
                request_id=req_id,
                query=request.query,
                resolved_task=resolved_task,
                status=ToolStatus.FAILED,
                answer=f"Error: {err.message}",
                confidence=None,
                evidence=[],
                artifacts=[],
                execution_trace=system_trace,
                errors=[err_detail],
            )
        except Exception as unhandled:
            logger.exception(f"Unexpected system error while processing [{req_id}]: {unhandled}")
            err_detail = SatQueryErrorDetail(
                error_code="INTERNAL_SYSTEM_ERROR",
                message="An unexpected internal error occurred during execution.",
                request_id=req_id,
                details={"error_type": type(unhandled).__name__},
            )
            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.ERROR_ENCOUNTERED,
                    component="AgentController",
                    status="FAILED",
                    details={"error_type": type(unhandled).__name__},
                )
            )
            return QueryResponse(
                request_id=req_id,
                query=request.query,
                resolved_task=resolved_task,
                status=ToolStatus.FAILED,
                answer="An unexpected internal error occurred during analysis.",
                confidence=None,
                evidence=[],
                artifacts=[],
                execution_trace=system_trace,
                errors=[err_detail],
            )
