"""Comprehensive Division 3 Integration & Verification Test Suite.

Rigorously verifies:
1. Division 3 tool registration in default_registry alongside Division 2.
2. API discovery of both single_image_rs_specialist and bitemporal_change_specialist.
3. Task vocabulary discovery across all remote-sensing task types.
4. Single-image routing to Division 2 (PaliGemma LoRA specialist).
5. Bi-temporal routing to Division 3 (Bi-Temporal Change specialist).
6. Multi-step composite workflow routing (Division 3 -> Division 2).
7. FastAPI multipart dual-image upload query endpoint.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.main import app
from agent.controller import AgentController
from core.interfaces import BaseSpecialistTool
from core.schemas import (
    EvidenceType,
    ExecutionStage,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    TaskType,
    ToolRequest,
    ToolStatus,
)
from registry.registry import ToolRegistry, default_registry
from specialists.mock import register_default_mocks
from specialists.single_image.specialist import SingleImageRSSpecialistTool
from specialists.temporal_change import register_temporal_change_specialist
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool


@pytest.fixture
def integrated_registry() -> ToolRegistry:
    """Populated registry containing real Division 2, real Division 3, and mock partners."""
    reg = ToolRegistry()
    register_default_mocks(reg)
    div2_tool = SingleImageRSSpecialistTool()
    reg.register(div2_tool, overwrite=True)
    div3_tool = register_temporal_change_specialist(reg)
    return reg


@pytest.fixture
def integrated_controller(integrated_registry: ToolRegistry) -> AgentController:
    """AgentController backed by the integrated registry."""
    return AgentController(registry=integrated_registry)


@pytest.fixture
def client():
    """FastAPI TestClient with live app lifespan ensuring both specialists are registered."""
    with TestClient(app) as c:
        yield c


def test_division3_tool_registration(integrated_registry: ToolRegistry):
    """Verify Section 3.A: Division 3 specialist is registered and healthy."""
    assert integrated_registry.has_tool("bitemporal_change_specialist")
    assert integrated_registry.has_tool("single_image_rs_specialist")

    tool = integrated_registry.get("bitemporal_change_specialist")
    assert isinstance(tool, BaseSpecialistTool)
    assert isinstance(tool, BiTemporalChangeSpecialistTool)
    assert tool.version == "1.0.0"
    assert TaskType.CHANGE_ANALYSIS in tool.supported_tasks
    assert TaskType.CHANGE_VQA in tool.supported_tasks
    assert tool.health_check() is True


def test_api_tools_discovery(client):
    """Verify Section 3.B: GET /api/v1/tools exposes both Division 2 and Division 3 specialists."""
    resp = client.get("/api/v1/tools")
    assert resp.status_code == 200
    tools = resp.json()
    tool_names = [t["name"] for t in tools]
    assert "single_image_rs_specialist" in tool_names
    assert "bitemporal_change_specialist" in tool_names


def test_api_tasks_discovery(client):
    """Verify Section 3.C: GET /api/v1/tasks exposes all supported task types."""
    resp = client.get("/api/v1/tasks")
    assert resp.status_code == 200
    tasks_data = resp.json()["tasks"]
    task_keys = [t["task"] for t in tasks_data]
    assert "single_image_vqa" in task_keys
    assert "single_image_grounding" in task_keys
    assert "single_image_caption" in task_keys
    assert "change_analysis" in task_keys
    assert "change_vqa" in task_keys


@pytest.mark.asyncio
async def test_agent_routing_single_image(integrated_controller: AgentController):
    """Verify Section 3.D: Single-image query routes to Division 2 specialist."""
    req = QueryRequest(
        query="What land cover dominates this scene?",
        images=[
            ImageInput(
                path_or_uri="demo_assets/demo_optical_single.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            )
        ],
    )
    resp = await integrated_controller.process_query(req)
    assert resp.status == ToolStatus.SUCCESS
    assert resp.task_intent.task == TaskType.SINGLE_IMAGE_VQA
    assert resp.agent_decision.selected_specialist == "single_image_rs_specialist"
    assert len(resp.answer) > 0


@pytest.mark.asyncio
async def test_agent_routing_bitemporal_change(integrated_controller: AgentController):
    """Verify Section 3.E: Bi-temporal query routes to Division 3 specialist."""
    req = QueryRequest(
        query="What changed between these two dates?",
        images=[
            ImageInput(
                image_id="img-t0",
                path_or_uri="demo_assets/demo_change_t0.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
            ImageInput(
                image_id="img-t1",
                path_or_uri="demo_assets/demo_change_t1.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
        ],
    )
    resp = await integrated_controller.process_query(req)
    assert resp.status == ToolStatus.SUCCESS
    assert resp.task_intent.task in {TaskType.CHANGE_VQA, TaskType.CHANGE_ANALYSIS}
    assert resp.agent_decision.selected_specialist == "bitemporal_change_specialist"
    assert len(resp.answer) > 0
    assert len(resp.evidence) >= 1
    assert any(e.type in {EvidenceType.CHANGE_MAP, EvidenceType.BOUNDING_BOX, EvidenceType.HEATMAP} for e in resp.evidence)


@pytest.mark.asyncio
async def test_composite_workflow_execution(integrated_controller: AgentController):
    """Verify Section 3.F: Multi-step composite workflow routes through Div 3 then Div 2."""
    req = QueryRequest(
        query="What changed, where did it happen, and what is present in the change?",
        images=[
            ImageInput(
                image_id="img-t0",
                path_or_uri="demo_assets/demo_change_t0.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
            ImageInput(
                image_id="img-t1",
                path_or_uri="demo_assets/demo_change_t1.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            ),
        ],
    )
    resp = await integrated_controller.process_query(req)
    assert resp.status == ToolStatus.SUCCESS
    assert resp.task_plan.is_multi_step is True
    assert len(resp.task_plan.steps) == 2
    assert resp.task_plan.steps[0].tool_name == "bitemporal_change_specialist"
    assert resp.task_plan.steps[1].tool_name == "single_image_rs_specialist"
    assert resp.task_plan.steps[0].status == "completed"
    assert resp.task_plan.steps[1].status == "completed"
    assert len(resp.evidence) >= 1


def test_fastapi_multipart_bitemporal_upload(client):
    """Verify Section 5: Live FastAPI multipart upload executes bi-temporal pipeline."""
    t0_path = Path("demo_assets/demo_change_t0.png")
    t1_path = Path("demo_assets/demo_change_t1.png")

    with open(t0_path, "rb") as f0, open(t1_path, "rb") as f1:
        files = [
            ("files", ("demo_change_t0.png", f0.read(), "image/png")),
            ("files", ("demo_change_t1.png", f1.read(), "image/png")),
        ]

    resp = client.post(
        "/api/v1/query/multipart",
        data={"query": "Detect surface difference between these two acquisitions."},
        files=files,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["task_intent"]["task"] in {"change_analysis", "change_vqa"}
    assert body["agent_decision"]["selected_specialist"] == "bitemporal_change_specialist"
    assert len(body["answer"]) > 0
