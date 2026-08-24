"""Agent Controller for SatQuery AI (Division 1 Core Orchestration).

Coordinates the end-to-end vision-language pipeline:
Input Validation -> Intent Resolution -> Workflow Planning -> Specialist Execution -> Result Aggregation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import List, Optional
import uuid

from core.errors import SatQueryException
from core.logging import get_logger, set_request_id
from core.schemas import (
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
            dur_router = round((time.perf_counter() - t0) * 1000.0, 2)

            system_trace.append(
                ExecutionTraceEntry(
                    stage=ExecutionStage.TOOL_SELECTED,
                    component="TaskRouter",
                    status="COMPLETED",
                    duration_ms=dur_router,
                    details={"selected_tool": selected_tool.name, "tool_version": selected_tool.version},
                )
            )

            # 5. Workflow Planning
            plan = WorkflowPlanner.create_plan(
                intent=intent,
                selected_tool=selected_tool,
                is_composite_query=False,
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

            tool_results = await self.execution_engine.execute_plan(
                plan=plan,
                base_request=base_tool_request,
            )

            # 7. Result Aggregation
            response = ResultAggregator.aggregate(
                request_id=req_id,
                query=request.query,
                resolved_task=resolved_task,
                tool_results=tool_results,
                system_trace_entries=system_trace,
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
