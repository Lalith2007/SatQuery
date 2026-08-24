"""End-to-end integration tests for FastAPI backend."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from core.schemas import ImageFormat, ImageInput, ImageModality


def test_api_health(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert "version" in data
    assert data["registered_tools_count"] >= 5


def test_api_list_tasks(api_client: TestClient):
    response = api_client.get("/api/v1/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    tasks = [t["task"] for t in data["tasks"]]
    assert "single_image_vqa" in tasks
    assert "change_analysis" in tasks
    assert "optical_sar_analysis" in tasks


def test_api_list_tools(api_client: TestClient):
    response = api_client.get("/api/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    tool_names = [t["name"] for t in data]
    assert "single_image_vqa_mock" in tool_names
    assert "bi_temporal_change_mock" in tool_names
    assert "optical_sar_cross_modal_mock" in tool_names


def test_api_query_single_image_vqa(api_client: TestClient, sample_png_path: Path):
    payload = {
        "query": "How many aircraft are parked on the apron?",
        "images": [
            {
                "path_or_uri": str(sample_png_path),
                "format": "png",
                "modality": "optical",
            }
        ],
    }
    response = api_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["resolved_task"] == "single_image_vqa"
    assert "aircraft" in data["answer"].lower()
    assert len(data["evidence"]) > 0
    assert len(data["execution_trace"]) > 0


def test_api_query_bitemporal_change(
    api_client: TestClient,
    sample_png_path: Path,
    temp_dir: Path,
):
    t1_path = temp_dir / "t1.png"
    t1_path.write_bytes(sample_png_path.read_bytes())

    payload = {
        "query": "What changed between these two acquisitions?",
        "images": [
            {"path_or_uri": str(sample_png_path), "format": "png", "modality": "optical"},
            {"path_or_uri": str(t1_path), "format": "png", "modality": "optical"},
        ],
    }
    response = api_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["resolved_task"] in {"change_analysis", "change_vqa"}
    assert len(data["evidence"]) > 0
    assert len(data["artifacts"]) > 0


def test_api_query_optical_sar_cross_modal(
    api_client: TestClient,
    sample_png_path: Path,
    sar_image_input: ImageInput,
):
    payload = {
        "query": "Combine optical and SAR to identify structures beneath clouds.",
        "images": [
            {"path_or_uri": str(sample_png_path), "format": "png", "modality": "optical"},
            {"path_or_uri": sar_image_input.path_or_uri, "format": "tiff", "modality": "sar"},
        ],
    }
    response = api_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["resolved_task"] == "optical_sar_analysis"
    assert "optical" in data["answer"].lower() and "sar" in data["answer"].lower()


def test_api_multipart_upload_query(api_client: TestClient, sample_png_path: Path):
    with open(sample_png_path, "rb") as f:
        files = [("files", ("test_aerial.png", f, "image/png"))]
        data = {
            "query": "Where is the runway located?",
            "task_hint": "single_image_grounding",
        }
        response = api_client.post("/api/v1/query/multipart", data=data, files=files)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["resolved_task"] == "single_image_grounding"
    assert len(res_data["evidence"]) > 0


def test_api_artifact_not_found(api_client: TestClient):
    response = api_client.get("/api/v1/artifacts/non_existent_artifact_id_12345")
    assert response.status_code == 404
