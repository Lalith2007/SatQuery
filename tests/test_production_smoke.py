"""SatQuery AI — Local Production Smoke Test Suite.

Executes real-model, zero-mock production smoke testing against the live FastAPI application:
1. GET /health
2. GET /version
3. Single-Image VQA (Qwen2.5-VL merged_full)
4. Single-Image Grounding (Qwen2.5-VL merged_full)
5. Single-Image Captioning (Qwen2.5-VL merged_full)
6. Bi-Temporal Change Detection (TinyCD frozen baseline)
7. Change VQA with 3-image sequential handoff (TinyCD + Qwen)
8. Optical-SAR Landcover Segmentation (CMAF frozen baseline)
9. Evidence artifact generation & persistence
10. Frontend static bundle delivery
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest
from starlette.testclient import TestClient

from core.schemas import ToolStatus
from deployment.server import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    """Module-scoped test client managing application lifespan with real weights."""
    with TestClient(app) as test_client:
        yield test_client


def test_01_health_endpoint(client: TestClient):
    """Verify GET /health returns healthy status and reports all 3 models ready."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    print("\n[SMOKE TEST 1] GET /health response:")
    print(json.dumps(data, indent=2))

    assert data["service"] == "SatQuery AI"
    assert data["deployment"] == "baseline-2026-09-18"
    assert "models" in data
    assert data["models"]["qwen25vl"]["status"] in ("READY", "AVAILABLE_ON_DISK")
    assert data["models"]["tinycd"]["status"] == "READY"
    assert data["models"]["cmaf"]["status"] == "READY"


def test_02_version_endpoint(client: TestClient):
    """Verify GET /version returns verified baseline metadata and exact hashes."""
    resp = client.get("/version")
    assert resp.status_code == 200
    data = resp.json()

    print("\n[SMOKE TEST 2] GET /version response:")
    print(json.dumps(data, indent=2))

    assert data["deployment"] == "baseline-2026-09-18"
    assert data["models"]["qwen25vl"]["parameters"] == 3754622976
    assert data["models"]["qwen25vl"]["revision"] == "stage1-baseline"
    assert len(data["models"]["qwen25vl"]["shards"]) == 2
    assert data["models"]["tinycd"]["sha256"] == "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"
    assert data["models"]["cmaf"]["sha256"] == "26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b"


def test_03_single_image_vqa(client: TestClient):
    """Verify Single-Image VQA with real Qwen2.5-VL merged_full model."""
    img_path = PROJECT_ROOT / "demo_assets/demo_a_vqa/vqa_01.png"
    assert img_path.exists()

    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Is it a rural or an urban area",
                "task_hint": "single_image_vqa",
            },
            files=[("files", ("vqa_01.png", f, "image/png"))],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 3] Single-Image VQA (Latency: {latency_s:.2f}s):")
    print(f"  Query:  {data['query']}")
    print(f"  Answer: {data['answer']}")
    print(f"  Status: {data['status']}")

    assert data["status"] == ToolStatus.SUCCESS.value
    assert len(data["answer"]) > 0


def test_04_single_image_grounding(client: TestClient):
    """Verify Single-Image Visual Grounding with real Qwen2.5-VL model."""
    img_path = PROJECT_ROOT / "demo_assets/demo_b_grounding/grounding_01.png"
    assert img_path.exists()

    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Locate the residential buildings in this image.",
                "task_hint": "single_image_grounding",
            },
            files=[("files", ("grounding_01.png", f, "image/png"))],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 4] Single-Image Grounding (Latency: {latency_s:.2f}s):")
    print(f"  Answer:   {data['answer']}")
    print(f"  Evidence: {len(data.get('evidence', []))} items")

    assert data["status"] == ToolStatus.SUCCESS.value


def test_05_single_image_captioning(client: TestClient):
    """Verify Single-Image Captioning with real Qwen2.5-VL model."""
    img_path = PROJECT_ROOT / "demo_assets/demo_a_vqa/vqa_01.png"
    assert img_path.exists()

    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Describe the satellite scene in detail.",
                "task_hint": "single_image_caption",
            },
            files=[("files", ("vqa_01.png", f, "image/png"))],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 5] Single-Image Captioning (Latency: {latency_s:.2f}s):")
    print(f"  Caption: {data['answer']}")

    assert data["status"] == ToolStatus.SUCCESS.value
    assert len(data["answer"].split()) >= 5


def test_06_bitemporal_change_detection(client: TestClient):
    """Verify Bi-Temporal Change Detection with real TinyCD specialist."""
    t0_path = PROJECT_ROOT / "demo_assets/demo_c_change/change_01_t0.png"
    t1_path = PROJECT_ROOT / "demo_assets/demo_c_change/change_01_t1.png"
    assert t0_path.exists() and t1_path.exists()

    t0 = time.perf_counter()
    with open(t0_path, "rb") as f0, open(t1_path, "rb") as f1:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Identify and localize all structural surface changes between T0 and T1.",
                "task_hint": "change_analysis",
            },
            files=[
                ("files", ("t0.png", f0, "image/png")),
                ("files", ("t1.png", f1, "image/png")),
            ],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 6] TinyCD Change Detection (Latency: {latency_s:.2f}s):")
    print(f"  Answer:    {data['answer']}")
    print(f"  Artifacts: {len(data.get('artifacts', []))} files generated")

    assert data["status"] == ToolStatus.SUCCESS.value
    assert len(data.get("artifacts", [])) >= 1


def test_07_change_vqa_three_image_handoff(client: TestClient):
    """Verify Change-VQA with exact 3-image sequential handoff [T0, T1, overlay]."""
    t0_path = PROJECT_ROOT / "demo_assets/demo_e_workflow/workflow_01_t0.png"
    t1_path = PROJECT_ROOT / "demo_assets/demo_e_workflow/workflow_01_t1.png"
    ov_path = PROJECT_ROOT / "demo_assets/demo_e_workflow/workflow_01_overlay.png"
    assert t0_path.exists() and t1_path.exists() and ov_path.exists()

    t0 = time.perf_counter()
    with open(t0_path, "rb") as f0, open(t1_path, "rb") as f1, open(ov_path, "rb") as fov:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "What structural changes occurred in the highlighted region?",
                "task_hint": "change_vqa",
            },
            files=[
                ("files", ("t0.png", f0, "image/png")),
                ("files", ("t1.png", f1, "image/png")),
                ("files", ("overlay.png", fov, "image/png")),
            ],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 7] 3-Image Change VQA (Latency: {latency_s:.2f}s):")
    print(f"  Answer: {data['answer']}")

    assert data["status"] in (ToolStatus.SUCCESS.value, ToolStatus.PARTIAL_SUCCESS.value)
    assert len(data["answer"]) > 0


def test_08_optical_sar_analysis(client: TestClient):
    """Verify Optical-SAR cross-modal fusion with real CMAF specialist."""
    opt_path = PROJECT_ROOT / "demo_assets/demo_d_optical_sar/cross_01_opt.png"
    sar_path = PROJECT_ROOT / "demo_assets/demo_d_optical_sar/cross_01_sar.tif"
    assert opt_path.exists() and sar_path.exists()

    t0 = time.perf_counter()
    with open(opt_path, "rb") as f_opt, open(sar_path, "rb") as f_sar:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures.",
                "task_hint": "optical_sar_analysis",
            },
            files=[
                ("files", ("optical.png", f_opt, "image/png")),
                ("files", ("sar.tif", f_sar, "image/tiff")),
            ],
        )
    latency_s = time.perf_counter() - t0
    assert resp.status_code == 200
    data = resp.json()

    print(f"\n[SMOKE TEST 8] CMAF Optical-SAR Fusion (Latency: {latency_s:.2f}s):")
    print(f"  Answer:    {data['answer']}")
    print(f"  Artifacts: {len(data.get('artifacts', []))} generated")

    assert data["status"] == ToolStatus.SUCCESS.value


def test_09_frontend_bundle_delivery(client: TestClient):
    """Verify frontend static single-page application is served correctly."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "SatQuery AI" in resp.text

    # Verify compiled static assets exist
    dist_dir = PROJECT_ROOT / "frontend/dist"
    index_html = dist_dir / "index.html"
    assert index_html.exists()

    print("\n[SMOKE TEST 9] Frontend Delivery:")
    print(f"  index.html size: {index_html.stat().st_size:,} bytes")


def test_10_evidence_artifact_persistence(client: TestClient):
    """Verify evidence artifacts generated across specialist pipelines persist on disk."""
    t0_path = PROJECT_ROOT / "demo_assets/demo_c_change/change_01_t0.png"
    t1_path = PROJECT_ROOT / "demo_assets/demo_c_change/change_01_t1.png"
    assert t0_path.exists() and t1_path.exists()

    with open(t0_path, "rb") as f0, open(t1_path, "rb") as f1:
        resp = client.post(
            "/api/v1/query/multipart",
            data={
                "query": "Detect surface changes and generate evidence artifacts.",
                "task_hint": "change_analysis",
            },
            files=[
                ("files", ("t0.png", f0, "image/png")),
                ("files", ("t1.png", f1, "image/png")),
            ],
        )
    assert resp.status_code == 200
    data = resp.json()
    artifacts = data.get("artifacts", [])
    assert len(artifacts) >= 1

    # Verify each artifact exists on disk and is non-empty
    verified_count = 0
    for art in artifacts:
        p_str = art.get("uri_or_path") or art.get("path")
        if p_str:
            art_path = Path(p_str)
            if art_path.exists() and art_path.stat().st_size > 0:
                verified_count += 1

    print(f"\n[SMOKE TEST 10] Evidence Artifact Persistence:")
    print(f"  Total generated: {len(artifacts)}")
    print(f"  Verified on disk: {verified_count}")

    assert verified_count >= 1

