"""Live preset end-to-end verification script for Demos A, B, C, D, and E."""

from starlette.testclient import TestClient
from app.main import app

def test_all_live_demo_presets_artifact_serving():
    with TestClient(app) as client:
        presets = [
            ("Demo A: Single-Image VQA", {
                "query": "What is the dominant land cover and infrastructure in this scene?",
                "images": [{"path_or_uri": "demo_assets/demo_optical_single.png", "format": "png", "modality": "optical"}]
            }),
            ("Demo B: Spatial Grounding", {
                "query": "Where is the airport runway and apron in this image?",
                "images": [{"path_or_uri": "demo_assets/demo_airport_grounding.png", "format": "png", "modality": "optical"}]
            }),
            ("Demo C: Bi-Temporal Change", {
                "query": "What changed between these two acquisition dates?",
                "images": [
                    {"path_or_uri": "demo_assets/demo_change_t0.png", "format": "png", "modality": "optical"},
                    {"path_or_uri": "demo_assets/demo_change_t1.png", "format": "png", "modality": "optical"}
                ]
            }),
            ("Demo D: Optical-SAR Fusion", {
                "query": "Use optical and SAR images together to identify structures beneath clouds.",
                "images": [
                    {"path_or_uri": "demo_assets/demo_optical_cross.png", "format": "png", "modality": "optical"},
                    {"path_or_uri": "demo_assets/demo_sar_cross.tif", "format": "tiff", "modality": "sar"}
                ]
            }),
            ("Demo E: Multi-Tool Workflow", {
                "query": "What changed, where did it happen, and was the new region built-up?",
                "images": [
                    {"path_or_uri": "demo_assets/demo_change_t0.png", "format": "png", "modality": "optical"},
                    {"path_or_uri": "demo_assets/demo_change_t1.png", "format": "png", "modality": "optical"}
                ]
            }),
        ]

        for name, payload in presets:
            res = client.post("/api/v1/query", json=payload)
            assert res.status_code == 200, f"{name} query failed: {res.text}"
            data = res.json()
            assert data["status"] == "success", f"{name} status not success: {data}"
            artifacts = data.get("artifacts", [])
            assert len(artifacts) > 0, f"Expected visual artifacts for {name}"
            print(f"[{name}] returned {len(artifacts)} artifact(s).")
            for art in artifacts:
                art_id = art["artifact_id"]
                art_res = client.get(f"/api/v1/artifacts/{art_id}")
                assert art_res.status_code == 200, f"Artifact '{art['name']}' with ID '{art_id}' returned {art_res.status_code}"
                assert art_res.headers.get("content-type") in {"image/png", "image/jpeg", "image/tiff"}
                assert len(art_res.content) > 100
                print(f"  --> Successfully retrieved artifact '{art['name']}' (ID: {art_id}) [Status: 200, Type: {art_res.headers.get('content-type')}, Size: {len(art_res.content)} bytes]")

if __name__ == "__main__":
    test_all_live_demo_presets_artifact_serving()
    print("\nALL PRESETS VERIFIED AND SERVING REAL ARTIFACTS WITH 200 OK!")
