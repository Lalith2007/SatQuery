"""End-to-end integration tests for Division 5 FastAPI presentation, artifacts, and evaluation endpoints."""

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.main import app
from core.config import settings
from core.schemas import Evidence, EvidenceType, QueryRequest, QueryResponse, TaskType, ToolStatus
from presentation.evidence_renderer import ArtifactRegistry, EvidenceRenderer


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


def test_api_list_evaluation_benchmarks(client):
    """Verify GET /api/v1/evaluation/benchmarks returns supported benchmark suites."""
    response = client.get("/api/v1/evaluation/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "supported_benchmarks" in data
    bench_ids = [b["id"] for b in data["supported_benchmarks"]]
    assert "vrsbench" in bench_ids
    assert "rsvqa" in bench_ids
    assert "cdvqa" in bench_ids
    assert "isro_sac" in bench_ids


def test_api_run_benchmark_evaluation(client):
    """Verify POST /api/v1/evaluation/run executes live evaluation."""
    payload = {
        "benchmark": "vrsbench",
        "predictions": [
            {"answer": "Airport runway with 4 planes.", "bbox": [0.1, 0.1, 0.5, 0.5]},
        ],
        "ground_truths": [
            {"answer": "Airport runway with 4 aircraft.", "bbox": [0.11, 0.09, 0.51, 0.49], "category": "presence"},
        ],
    }

    response = client.post("/api/v1/evaluation/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["benchmark"] == "VRSBench"
    assert data["samples_evaluated"] == 1
    assert "scoreboard_markdown" in data
    assert "vqa_accuracy" in data["metrics"]


def test_api_generate_and_download_report(client):
    """Verify report generation and downloading via REST endpoints."""
    # 1. Create mock response payload
    mock_resp = {
        "request_id": "req-api-report-001",
        "query": "What is in this optical image?",
        "resolved_task": "single_image_vqa",
        "status": "success",
        "answer": "Agricultural crop field with center-pivot irrigation.",
        "confidence": 0.89,
        "evidence": [],
        "artifacts": [],
        "execution_trace": [],
    }

    gen_payload = {
        "response": mock_resp,
        "format": "html",
    }

    # 2. Generate report
    gen_res = client.post("/api/v1/reports/generate", json=gen_payload)
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["status"] == "success"
    report_id = gen_data["report_id"]
    download_url = gen_data["download_url"]

    # 3. Download generated report file
    dl_res = client.get(download_url)
    assert dl_res.status_code == 200
    assert "<!DOCTYPE html>" in dl_res.text
    assert "Agricultural crop field" in dl_res.text


def test_api_presentation_dashboard_accessible(client):
    """Verify GET / and GET /demo return the presentation dashboard with Division 5 components."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "SatQuery AI" in res_root.text
    assert "Benchmark Evaluation & Scoreboards" in res_root.text
    assert "Developer: Laksh" in res_root.text

    res_demo = client.get("/demo")
    assert res_demo.status_code == 200
    assert "SatQuery AI" in res_demo.text


# ---------------------------------------------------------------------------
# Artifact Serving Regression Tests (Division 5 Bug Fix Verification)
# ---------------------------------------------------------------------------

def test_artifact_serving_test_a_existing_artifact(client):
    """Test A — Existing registered artifact returns 200 with correct media type."""
    # Create test artifact file in evidence directory
    evidence_dir = settings.artifact_storage_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    test_art_path = evidence_dir / "test_artifact_serving.png"

    # Write small 1x1 PNG
    from PIL import Image
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(test_art_path)

    test_art_id = "test-serving-uuid-001"
    ArtifactRegistry.register(test_art_id, test_art_path, name=test_art_path.name)

    res = client.get(f"/api/v1/artifacts/{test_art_id}")
    assert res.status_code == 200
    assert res.headers.get("content-type") == "image/png"
    assert len(res.content) > 0


def test_artifact_serving_test_b_missing_artifact(client):
    """Test B — Nonexistent artifact returns 404 Not Found."""
    res = client.get("/api/v1/artifacts/nonexistent-artifact-uuid-999999")
    assert res.status_code == 404
    data = res.json()
    assert "not found" in data["detail"].lower()


def test_artifact_serving_test_c_generated_division5_evidence(client):
    """Test C — Generated Division 5 evidence artifact returned in query response can be requested via API."""
    query_payload = {
        "query": "Where is the airport runway and apron in this image?",
        "images": [
            {
                "path_or_uri": "demo_assets/demo_airport_grounding.png",
                "format": "png",
                "modality": "optical",
            }
        ],
    }

    # Execute query
    res = client.post("/api/v1/query", json=query_payload)
    assert res.status_code == 200
    resp_data = res.json()

    artifacts = resp_data.get("artifacts", [])
    assert len(artifacts) > 0, "Expected at least 1 rendered visual artifact in response."

    for art in artifacts:
        art_id = art["artifact_id"]
        art_res = client.get(f"/api/v1/artifacts/{art_id}")
        assert art_res.status_code == 200, f"Failed to retrieve artifact '{art_id}': {art_res.status_code}"
        assert art_res.headers.get("content-type") in {"image/png", "image/jpeg", "image/tiff"}


def test_artifact_serving_test_d_multiple_evidence_types(client):
    """Test D — Multiple evidence types (Grounding bbox, Bi-temporal change, Optical-SAR blend, Crop)."""
    # 1. Bi-Temporal Change Query
    change_query = {
        "query": "What changed between these two acquisition dates?",
        "images": [
            {"path_or_uri": "demo_assets/demo_change_t0.png", "format": "png", "modality": "optical"},
            {"path_or_uri": "demo_assets/demo_change_t1.png", "format": "png", "modality": "optical"},
        ],
    }
    res_change = client.post("/api/v1/query", json=change_query)
    assert res_change.status_code == 200
    change_arts = res_change.json().get("artifacts", [])
    assert len(change_arts) > 0
    for art in change_arts:
        r = client.get(f"/api/v1/artifacts/{art['artifact_id']}")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "image/png"

    # 2. Optical-SAR Fusion Query
    fusion_query = {
        "query": "Identify metallic structures beneath cloud cover using optical and SAR.",
        "images": [
            {"path_or_uri": "demo_assets/demo_optical_cross.png", "format": "png", "modality": "optical"},
            {"path_or_uri": "demo_assets/demo_sar_cross.tif", "format": "tiff", "modality": "sar"},
        ],
    }
    res_fusion = client.post("/api/v1/query", json=fusion_query)
    assert res_fusion.status_code == 200
    fusion_arts = res_fusion.json().get("artifacts", [])
    assert len(fusion_arts) > 0
    for art in fusion_arts:
        r = client.get(f"/api/v1/artifacts/{art['artifact_id']}")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "image/png"


def test_artifact_serving_test_e_path_safety(client):
    """Test E — Directory traversal and arbitrary path access are blocked."""
    # Test with parent directory traversal
    res_traversal = client.get("/api/v1/artifacts/..%2F..%2F..%2Fetc%2Fpasswd")
    assert res_traversal.status_code in {400, 404, 422}

    res_traversal2 = client.get("/api/v1/artifacts/..%5C..%5CWindows%5Cwin.ini")
    assert res_traversal2.status_code in {400, 404, 422}
