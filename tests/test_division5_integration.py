"""End-to-end integration tests for Division 5 FastAPI presentation and evaluation endpoints."""

import pytest
from starlette.testclient import TestClient

from app.main import app
from core.schemas import QueryResponse, TaskType, ToolStatus


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
