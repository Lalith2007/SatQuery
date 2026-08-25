"""Unit tests for failure boundaries, timeouts, and error handling."""

import pytest

from core.errors import InferenceError, ToolNotFoundError, ToolTimeoutError
from core.schemas import (
    ExecutionStage,
    ImageInput,
    QueryRequest,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from agent.controller import AgentController
from agent.execution_engine import ExecutionEngine
from agent.workflow import WorkflowPlanner, WorkflowStep, WorkflowPlan
from registry.registry import ToolRegistry
from specialists.mock import MockFailingTool


@pytest.mark.asyncio
async def test_tool_timeout_handling(optical_image_input: ImageInput):
    registry = ToolRegistry()
    failing_tool = MockFailingTool(name="timeout_tool", failure_mode="timeout", delay_seconds=0.1)
    registry.register(failing_tool)

    engine = ExecutionEngine(registry=registry, default_timeout_seconds=0.01)
    plan = WorkflowPlan(
        intent={"task": TaskType.SINGLE_IMAGE_VQA, "confidence": 1.0, "intent_explanation": "Test timeout"},
        steps=[WorkflowStep(step_index=0, task=TaskType.SINGLE_IMAGE_VQA, tool_name="timeout_tool")],
    )

    req = ToolRequest(
        request_id="req-timeout",
        task=TaskType.SINGLE_IMAGE_VQA,
        query="Test timeout",
        images=[optical_image_input],
    )

    with pytest.raises(ToolTimeoutError):
        await engine.execute_plan(plan=plan, base_request=req, timeout_seconds=0.01)


@pytest.mark.asyncio
async def test_tool_inference_error_handling(optical_image_input: ImageInput):
    registry = ToolRegistry()
    failing_tool = MockFailingTool(name="error_tool", failure_mode="inference_error")
    registry.register(failing_tool)

    engine = ExecutionEngine(registry=registry)
    plan = WorkflowPlan(
        intent={"task": TaskType.SINGLE_IMAGE_VQA, "confidence": 1.0, "intent_explanation": "Test error"},
        steps=[WorkflowStep(step_index=0, task=TaskType.SINGLE_IMAGE_VQA, tool_name="error_tool")],
    )

    req = ToolRequest(
        request_id="req-error",
        task=TaskType.SINGLE_IMAGE_VQA,
        query="Test error",
        images=[optical_image_input],
    )

    with pytest.raises(InferenceError):
        await engine.execute_plan(plan=plan, base_request=req)


@pytest.mark.asyncio
async def test_agent_controller_graceful_error_response(optical_image_input: ImageInput):
    # Empty registry with no tools
    empty_registry = ToolRegistry()
    controller = AgentController(registry=empty_registry)

    query_req = QueryRequest(
        query="What is in this image?",
        images=[optical_image_input],
    )

    # Controller should catch domain exception and return structured error response without crashing
    response = await controller.process_query(query_req)
    assert response.status == ToolStatus.FAILED
    assert len(response.errors) > 0
    assert response.errors[0].error_code == "TOOL_NOT_FOUND"
    assert any(t.stage == ExecutionStage.ERROR_ENCOUNTERED for t in response.execution_trace)
