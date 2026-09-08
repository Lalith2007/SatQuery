"""Comprehensive tests for Division 1 Enhancement Pass.

Tests TaskPlan, AgentDecision, live execution trace fields,
"Why this tool?" operational rationale, multi-tool workflows,
and specialist tool swapping demonstration.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from core.interfaces import BaseSpecialistTool
from core.schemas import (
    AgentDecision,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    QueryResponse,
    TaskIntent,
    TaskPlan,
    TaskPlanStep,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from agent.controller import AgentController
from agent.intent_resolver import IntentResolver
from agent.workflow import WorkflowPlanner
from registry.registry import ToolRegistry
from specialists.mock import (
    AlternateMockSingleImageVQATool,
    MockBiTemporalChangeTool,
    MockOpticalSARAnalysisTool,
    MockSingleImageGroundingTool,
    MockSingleImageVQATool,
)


def test_task_plan_schema_and_derivation(optical_image_input: ImageInput):
    intent = TaskIntent(
        task=TaskType.SINGLE_IMAGE_VQA,
        confidence=0.92,
        intent_explanation="Single image visual question answering requested.",
    )
    vqa_tool = MockSingleImageVQATool()

    task_plan = WorkflowPlanner.create_task_plan(
        intent=intent,
        selected_tool=vqa_tool,
        images=[optical_image_input],
    )

    assert isinstance(task_plan, TaskPlan)
    assert task_plan.is_multi_step is False
    assert len(task_plan.steps) == 1
    assert task_plan.steps[0].tool_name == "single_image_vqa_mock"
    assert "single image vqa" in task_plan.goal.lower()

    # Verify WorkflowPlan derivation
    workflow_plan = WorkflowPlanner.derive_workflow_plan(task_plan, intent)
    assert len(workflow_plan.steps) == 1
    assert workflow_plan.steps[0].tool_name == "single_image_vqa_mock"


def test_multi_step_task_plan_creation(optical_image_input: ImageInput, optical_image_t1_input: ImageInput):
    intent = TaskIntent(
        task=TaskType.CHANGE_ANALYSIS,
        confidence=0.95,
        intent_explanation="Multi-step change and characterization requested.",
        extracted_parameters={"is_composite": True, "secondary_task": TaskType.SINGLE_IMAGE_GROUNDING},
    )
    change_tool = MockBiTemporalChangeTool()
    grounding_tool = MockSingleImageGroundingTool()

    task_plan = WorkflowPlanner.create_task_plan(
        intent=intent,
        selected_tool=change_tool,
        secondary_tool=grounding_tool,
        is_composite_query=True,
        images=[optical_image_input, optical_image_t1_input],
    )

    assert task_plan.is_multi_step is True
    assert len(task_plan.steps) == 2
    assert task_plan.steps[0].tool_name == "bi_temporal_change_mock"
    assert task_plan.steps[1].tool_name == "single_image_grounding_mock"
    assert len(task_plan.steps[1].dependencies) == 1


@pytest.mark.asyncio
async def test_agent_controller_decision_and_taskplan(
    optical_image_input: ImageInput,
    test_registry: ToolRegistry,
):
    controller = AgentController(registry=test_registry)
    query_req = QueryRequest(
        query="What is in this scene?",
        images=[optical_image_input],
    )

    response = await controller.process_query(query_req)
    assert isinstance(response, QueryResponse)
    assert response.status == ToolStatus.SUCCESS

    # 1. Agent Decision Card
    assert response.agent_decision is not None
    assert response.agent_decision.task == TaskType.SINGLE_IMAGE_VQA
    assert response.agent_decision.image_count == 1
    assert ImageModality.OPTICAL in response.agent_decision.detected_modalities
    assert "single_image_vqa_mock" in response.agent_decision.selected_specialist
    assert len(response.agent_decision.why_this_tool) > 0

    # 2. Canonical TaskPlan
    assert response.task_plan is not None
    assert len(response.task_plan.steps) == 1
    assert response.task_plan.steps[0].status == "completed"

    # 3. Selected Tools
    assert "single_image_vqa_mock" in response.selected_tools


@pytest.mark.asyncio
async def test_composite_multi_tool_execution(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
    test_registry: ToolRegistry,
):
    controller = AgentController(registry=test_registry)
    # Composite query matching pattern
    query_req = QueryRequest(
        query="What changed, where did it happen, and was the new region built-up?",
        images=[optical_image_input, optical_image_t1_input],
    )

    response = await controller.process_query(query_req)
    assert response.status == ToolStatus.SUCCESS
    assert response.task_plan.is_multi_step is True
    assert len(response.task_plan.steps) == 2
    assert "Step 1" in response.answer
    assert "Step 2" in response.answer
    assert len(response.selected_tools) == 2


def test_tool_swapping_in_registry(optical_image_input: ImageInput):
    registry = ToolRegistry()
    standard_tool = MockSingleImageVQATool()
    alternate_tool = AlternateMockSingleImageVQATool()

    # Initial registration
    registry.register(standard_tool)
    assert registry.get("single_image_vqa_mock").version == "1.0.0"

    # Swap with alternate mock specialist
    old_tool = registry.swap_tool("single_image_vqa_mock", alternate_tool)
    assert old_tool.name == "single_image_vqa_mock"
    assert registry.has_tool("alternate_single_image_vqa_mock")
    assert not registry.has_tool("single_image_vqa_mock")

    # Find tool for task returns the newly swapped tool
    tools = registry.find_tools_for_task(TaskType.SINGLE_IMAGE_VQA)
    assert len(tools) == 1
    assert tools[0].name == "alternate_single_image_vqa_mock"
    assert tools[0].version == "2.0.0-demo-mock"


def test_api_tool_swap_endpoint(api_client: TestClient):
    # Swap to alternate
    res = api_client.post("/api/v1/tools/swap", json={"target_tool": "single_image_vqa_mock", "use_alternate": True})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "swapped"
    assert data["active_tool"] == "alternate_single_image_vqa_mock"

    # Restore standard
    res_restore = api_client.post("/api/v1/tools/swap", json={"target_tool": "alternate_single_image_vqa_mock", "use_alternate": False})
    assert res_restore.status_code == 200
    restore_data = res_restore.json()
    assert restore_data["status"] == "restored"
    assert restore_data["active_tool"] == "single_image_vqa_mock"


def test_api_presentation_ui(api_client: TestClient):
    res = api_client.get("/")
    assert res.status_code == 200
    assert "SatQuery AI" in res.text

    res_demo = api_client.get("/demo")
    assert res_demo.status_code == 200
    assert "SatQuery AI" in res_demo.text
    assert "Specialist Tool Registry" in res_demo.text
