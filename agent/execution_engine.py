"""Sequential Execution Engine for specialist tool workflows.

Executes workflow steps sequentially, passes context across stages,
enforces timeouts, and maintains an auditable operational execution trace.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
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
            cumulative_context[f"step_{step.step_index}_result"] = {
                "task": step_result.task.value,
                "tool_name": step.tool_name,
                "answer": step_result.answer,
                "evidence_count": len(step_result.evidence),
                "artifacts": [a.model_dump() for a in step_result.artifacts],
            }

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
        step_request = ToolRequest(
            request_id=base_request.request_id,
            task=step.task,
            query=base_request.query,
            images=base_request.images,
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

            # Ensure trace entry is present
            trace_entry = ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component=tool.name,
                status="COMPLETED",
                duration_ms=round(elapsed_ms, 2),
                details={"task": step.task.value, "step_index": step.step_index},
            )
            result.execution_trace.insert(0, trace_entry)
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
