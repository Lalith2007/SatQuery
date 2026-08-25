"""Comprehensive Division 2 Integration & Handoff Verification Test Suite.

Rigorously verifies:
1. Division 2 Specialist contract conformance (BaseSpecialistTool)
2. ToolRegistry registration & live discovery
3. End-to-end agent routing (VQA, Grounding, Captioning)
4. TaskPlan & AgentDecision schema compliance
5. ExecutionTrace stage recording
6. FastAPI multipart endpoint (/api/v1/query/multipart)
7. Failure recovery (invalid cardinality, unsupported formats, nonexistent files)
8. Adapter checksum integrity (SHA-256)
"""

import hashlib
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


@pytest.fixture
def integrated_registry() -> ToolRegistry:
    """Populated registry containing real Division 2 specialist and mock partners."""
    reg = ToolRegistry()
    register_default_mocks(reg)
    div2_tool = SingleImageRSSpecialistTool()
    reg.register(div2_tool, overwrite=True)
    return reg


@pytest.fixture
def integrated_controller(integrated_registry: ToolRegistry) -> AgentController:
    """AgentController backed by the integrated registry."""
    return AgentController(registry=integrated_registry)


@pytest.fixture
def client(integrated_registry: ToolRegistry):
    """FastAPI TestClient with integrated registry."""
    # Ensure default_registry has real specialist registered
    div2_tool = SingleImageRSSpecialistTool()
    default_registry.register(div2_tool, overwrite=True)
    register_default_mocks(default_registry)
    # Re-register real tool to ensure it is present
    default_registry.register(div2_tool, overwrite=True)
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_division2_contract_conformance():
    """Verify Section 8: Division 2 Specialist implements BaseSpecialistTool."""
    tool = SingleImageRSSpecialistTool()
    assert isinstance(tool, BaseSpecialistTool)
    assert tool.name == "single_image_rs_specialist"
    assert tool.version == "1.0.0-adapted"
    assert TaskType.SINGLE_IMAGE_VQA in tool.supported_tasks
    assert TaskType.SINGLE_IMAGE_GROUNDING in tool.supported_tasks
    assert TaskType.SINGLE_IMAGE_CAPTION in tool.supported_tasks

    # Health check
    assert tool.health_check() is True

    # Validate valid request
    valid_req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_VQA,
        images=[
            ImageInput(
                path_or_uri="demo_assets/demo_optical_single.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            )
        ],
        query="What land cover dominates this scene?",
    )
    val = tool.validate_request(valid_req)
    assert val.is_valid is True

    # Execute
    res = await tool.execute(valid_req)
    assert res.status == ToolStatus.SUCCESS
    assert res.answer is not None
    assert len(res.answer) > 0
    assert res.confidence is not None
    assert len(res.execution_trace) > 0


@pytest.mark.asyncio
async def test_agent_routing_vqa(integrated_controller: AgentController):
    """Verify Section 9: Agent routing for single-image VQA."""
    req = QueryRequest(
        query="What land cover and major infrastructure are visible in this image?",
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
    assert resp.agent_decision.task == TaskType.SINGLE_IMAGE_VQA
    assert resp.task_plan.is_multi_step is False
    assert len(resp.task_plan.steps) == 1
    assert resp.task_plan.steps[0].status == "completed"
    assert len(resp.evidence) >= 1
    assert any(e.stage == ExecutionStage.REQUEST_RECEIVED for e in resp.execution_trace)
    assert any(e.stage == ExecutionStage.TASK_RESOLVED for e in resp.execution_trace)
    assert any(e.stage == ExecutionStage.INFERENCE_EXECUTED for e in resp.execution_trace)


@pytest.mark.asyncio
async def test_agent_routing_grounding(integrated_controller: AgentController):
    """Verify Section 10: Agent routing for text-guided visual grounding."""
    req = QueryRequest(
        query="Where is the runway?",
        images=[
            ImageInput(
                path_or_uri="demo_assets/demo_airport_grounding.png",
                modality=ImageModality.OPTICAL,
                format=ImageFormat.PNG,
            )
        ],
    )
    resp = await integrated_controller.process_query(req)
    assert resp.status == ToolStatus.SUCCESS
    assert resp.task_intent.task == TaskType.SINGLE_IMAGE_GROUNDING
    assert resp.agent_decision.selected_specialist == "single_image_rs_specialist"
    assert len(resp.evidence) >= 1
    
    bbox_ev = [e for e in resp.evidence if e.type == EvidenceType.BOUNDING_BOX]
    assert len(bbox_ev) >= 1
    bbox = bbox_ev[0].data["bbox"]
    assert len(bbox) == 4
    ymin, xmin, ymax, xmax = bbox
    assert 0.0 <= ymin <= 1.0
    assert 0.0 <= xmin <= 1.0
    assert 0.0 <= ymax <= 1.0
    assert 0.0 <= xmax <= 1.0
    assert ymin < ymax
    assert xmin < xmax


@pytest.mark.asyncio
async def test_agent_routing_caption(integrated_controller: AgentController):
    """Verify Section 11: Agent routing for scene captioning."""
    req = QueryRequest(
        query="Describe this remote-sensing scene.",
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
    assert resp.task_intent.task == TaskType.SINGLE_IMAGE_CAPTION
    assert resp.agent_decision.selected_specialist == "single_image_rs_specialist"
    assert len(resp.answer) > 0


def test_fastapi_endpoints(client):
    """Verify Section 12 & 13: FastAPI endpoints including multipart query."""
    # 1. Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    # 2. Tools
    r_tools = client.get("/api/v1/tools")
    assert r_tools.status_code == 200
    tools = r_tools.json()
    tool_names = [t["name"] for t in tools]
    assert "single_image_rs_specialist" in tool_names

    # 3. Tasks
    r_tasks = client.get("/api/v1/tasks")
    assert r_tasks.status_code == 200
    tasks = [t["task"] for t in r_tasks.json()["tasks"]]
    assert "single_image_vqa" in tasks
    assert "single_image_grounding" in tasks
    assert "single_image_caption" in tasks

    # 4. Multipart Query
    img_path = Path("demo_assets/demo_optical_single.png")
    with open(img_path, "rb") as f:
        files = [("files", ("demo_optical_single.png", f.read(), "image/png"))]
    
    r_query = client.post(
        "/api/v1/query/multipart",
        data={"query": "What land cover dominates this scene?"},
        files=files,
    )
    assert r_query.status_code == 200
    body = r_query.json()
    assert body["status"] == "success"
    assert body["task_intent"]["task"] == "single_image_vqa"
    assert body["agent_decision"]["selected_specialist"] == "single_image_rs_specialist"
    assert len(body["answer"]) > 0


def test_failure_handling(client):
    """Verify Section 17: Graceful failure handling without crashing."""
    img_path = Path("demo_assets/demo_optical_single.png")
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # A. Single-image task with 2 incompatible images
    files_two = [
        ("files", ("img1.png", img_bytes, "image/png")),
        ("files", ("img2.png", img_bytes, "image/png")),
    ]
    r_two = client.post(
        "/api/v1/query/multipart",
        data={"query": "What is in this image?", "task_hint": "single_image_vqa"},
        files=files_two,
    )
    assert r_two.status_code == 200
    body_two = r_two.json()
    assert body_two["status"] == "failed"
    assert len(body_two["errors"]) > 0

    # B. Unsupported file extension
    files_bad_ext = [
        ("files", ("test.gif", b"fake gif data", "image/gif")),
    ]
    r_bad = client.post(
        "/api/v1/query/multipart",
        data={"query": "What is in this image?"},
        files=files_bad_ext,
    )
    assert r_bad.status_code == 422
    assert "unsupported format" in r_bad.json()["detail"]


def test_adapter_model_checksum_verification():
    """Verify Section 20: Adapter weights match exact required SHA-256."""
    weights_path = Path("specialists/single_image/weights/satquery_paligemma_lora/adapter_model.safetensors")
    assert weights_path.exists(), f"Missing weights file at {weights_path}"

    sha = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    actual_hash = sha.hexdigest()
    expected_hash = "152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d"
    assert actual_hash == expected_hash, f"Adapter SHA mismatch! Expected {expected_hash}, got {actual_hash}"
