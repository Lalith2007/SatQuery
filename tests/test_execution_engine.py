"""Unit tests for sequential workflow planning and ExecutionEngine."""

import pytest

from core.schemas import (
    ImageInput,
    TaskIntent,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from agent.aggregator import ResultAggregator
from agent.execution_engine import ExecutionEngine
from agent.workflow import WorkflowPlanner
from registry.registry import ToolRegistry


@pytest.mark.asyncio
async def test_single_step_execution(
    optical_image_input: ImageInput,
    test_registry: ToolRegistry,
):
    engine = ExecutionEngine(registry=test_registry)
    tool = test_registry.get("single_image_vqa_mock")

    intent = TaskIntent(task=TaskType.SINGLE_IMAGE_VQA, confidence=0.9)
    plan = WorkflowPlanner.create_plan(intent=intent, selected_tool=tool)

    req = ToolRequest(
        request_id="req-single-test",
        task=TaskType.SINGLE_IMAGE_VQA,
        query="What is in the image?",
        images=[optical_image_input],
    )

    results = await engine.execute_plan(plan=plan, base_request=req)
    assert len(results) == 1
    assert results[0].status == ToolStatus.SUCCESS
    assert results[0].execution_trace[0].duration_ms is not None


@pytest.mark.asyncio
async def test_multi_step_sequential_execution_and_aggregation(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
    test_registry: ToolRegistry,
):
    engine = ExecutionEngine(registry=test_registry)
    tool1 = test_registry.get("bi_temporal_change_mock")
    tool2 = test_registry.get("single_image_vqa_mock")

    intent = TaskIntent(task=TaskType.CHANGE_ANALYSIS, confidence=0.95)
    plan = WorkflowPlanner.create_plan(
        intent=intent,
        selected_tool=tool1,
        secondary_tool=tool2,
        is_composite_query=True,
    )
    assert plan.is_multi_step is True
    assert len(plan.steps) == 2

    req = ToolRequest(
        request_id="req-multi-test",
        task=TaskType.CHANGE_ANALYSIS,
        query="Analyze change and interpret newly constructed features.",
        images=[optical_image_input, optical_image_t1_input],
    )

    results = await engine.execute_plan(plan=plan, base_request=req)
    assert len(results) == 2
    assert results[0].status == ToolStatus.SUCCESS
    assert results[1].status == ToolStatus.SUCCESS

    # Test Aggregator on multi-step results
    response = ResultAggregator.aggregate(
        request_id=req.request_id,
        query=req.query,
        resolved_task=TaskType.CHANGE_ANALYSIS,
        tool_results=results,
    )
    assert response.status == ToolStatus.SUCCESS
    assert "Step 1" in response.answer
    assert "Step 2" in response.answer
    assert len(response.evidence) > 0
    assert len(response.artifacts) > 0
