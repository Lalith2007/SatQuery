# Division 5: Evidence, Evaluation, and Presentation Layer

**Owner**: Manoj  
**Directory**: `presentation/`, `evaluation/`, `reports/`  
**Shared Contract**: `core.schemas.QueryResponse`, `core.schemas.Evidence`, `core.schemas.Artifact`

---

## 1. Overview & Responsibilities
Division 5 is responsible for:
- **Presentation & Web UI**: Consuming the FastAPI backend (`/api/v1/query`, `/health`, `/api/v1/artifacts/{id}`) and rendering interactive visual evidence to the user.
- **Evidence Rendering**: Displaying bounding boxes, change maps, heatmaps, and false-color composites.
- **Evaluation & Benchmarks**: Evaluating model accuracy, confidence calibration, and execution latency.

---

## 2. API Contract for Frontend Integration

### Query Endpoint
- **URL**: `POST /api/v1/query` (JSON) or `POST /api/v1/query/multipart` (File Upload)
- **Response Model**: `QueryResponse` (defined in `core/schemas.py`)

### Sample Response Structure
```json
{
  "request_id": "c7a80a2b-f06b-4e92-8051-7f9f75d6910a",
  "query": "Where is the airport runway?",
  "resolved_task": "single_image_grounding",
  "status": "success",
  "answer": "Successfully localized Airport Runway at normalized coordinates [0.4, 0.1, 0.6, 0.9].",
  "confidence": 0.93,
  "evidence": [
    {
      "id": "e1-456",
      "type": "bounding_box",
      "label": "Airport Runway",
      "confidence": 0.94,
      "data": {
        "bbox": [0.40, 0.10, 0.60, 0.90],
        "format": "[ymin, xmin, ymax, xmax]"
      },
      "image_id": "img-001"
    }
  ],
  "artifacts": [
    {
      "artifact_id": "art-789",
      "name": "bi_temporal_change_map.png",
      "type": "change_map",
      "uri_or_path": "artifacts_storage/mock_change_map.png",
      "description": "Binary change difference mask."
    }
  ],
  "execution_trace": [
    {
      "stage": "INPUT_VALIDATED",
      "component": "InputValidator",
      "status": "COMPLETED",
      "duration_ms": 2.1
    },
    {
      "stage": "TASK_RESOLVED",
      "component": "IntentResolver",
      "status": "COMPLETED",
      "duration_ms": 1.4,
      "details": {"resolved_task": "single_image_grounding"}
    },
    {
      "stage": "INFERENCE_EXECUTED",
      "component": "single_image_grounding_mock",
      "status": "COMPLETED",
      "duration_ms": 52.1
    }
  ],
  "errors": []
}
```

---

## 3. Artifact Retrieval Endpoint
- **URL**: `GET /api/v1/artifacts/{artifact_id}`
- Returns raw image / binary file for visualization directly in the browser/frontend.
