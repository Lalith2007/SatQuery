# SatQuery AI: Developer & Specialist Integration Guide (INTEGRATION.md)

This document provides the definitive integration manual for **Division 2 (Sruthi)**, **Division 3 (Dheeraj)**, **Division 4 (Laksh)**, and **Division 5 (Manoj)** to plug their specialist modules and presentation components into the **SatQuery AI Division 1 Backbone** developed by **Lalith**.

---

## 1. Directory Placement & Division Ownership

| Division | Owner | Domain | Target Directory |
| :--- | :--- | :--- | :--- |
| **Division 1** | **Lalith** | Agent Core, Backend, Orchestration, Schemas, Validation, Registry | `core/`, `agent/`, `validation/`, `registry/`, `app/` |
| **Division 2** | **Sruthi** | Single-Image Intelligence (VQA, Caption, Grounding) | `specialists/single_image/` |
| **Division 3** | **Dheeraj** | Bi-Temporal Change Intelligence (Change Analysis, Change VQA) | `specialists/temporal_change/` |
| **Division 4** | **Laksh** | Optical-SAR Cross-Modal Intelligence | `specialists/optical_sar/` |
| **Division 5** | **Manoj** | Evidence Presentation, GUI, Evaluation, Benchmarks | `presentation/`, `evaluation/`, `reports/` |

---

## 2. The Specialist Contract Interface

All specialist tools must inherit from `core.interfaces.BaseSpecialistTool`:

```python
from core.interfaces import BaseSpecialistTool, ValidationResult
from core.schemas import (
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

class MySpecialistTool(BaseSpecialistTool):
    def __init__(self):
        super().__init__(
            name="my_unique_tool_name",
            description="Clear explanation of model capability.",
            supported_tasks={TaskType.SINGLE_IMAGE_VQA},  # or CHANGE_ANALYSIS, OPTICAL_SAR_ANALYSIS, etc.
            version="1.0.0",
            metadata=ToolMetadata(
                name="my_unique_tool_name",
                description="Clear explanation of model capability.",
                version="1.0.0",
                supported_tasks=[TaskType.SINGLE_IMAGE_VQA],
                required_modalities=[ImageModality.OPTICAL],
                min_images=1,
                max_images=1,
                author_or_division="Division X (Name)",
            ),
        )

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        """Optional: Add custom domain parameter checks."""
        if len(request.images) != 1:
            return ValidationResult(is_valid=False, errors=["Requires 1 image."])
        return ValidationResult(is_valid=True)

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute model inference and return standardized ToolResult."""
        # 1. Access request inputs
        image = request.images[0]
        query = request.query
        
        # 2. Run inference (PyTorch, ONNX, HuggingFace, etc.)
        answer_text = "Your grounded answer here."
        
        # 3. Construct Evidence
        evidence_list = [
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Target Name",
                confidence=0.92,
                data={"bbox": [0.2, 0.3, 0.4, 0.5], "format": "[ymin, xmin, ymax, xmax]"},
                image_id=image.image_id,
            )
        ]

        # 4. Return canonical ToolResult
        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer_text,
            confidence=0.92,
            evidence=evidence_list,
            artifacts=[],
            model_info={"name": "MyModel", "version": "1.0"},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                )
            ],
        )

    def health_check(self) -> bool:
        """Return True if model checkpoints are loaded and GPU is healthy."""
        return True
```

---

## 3. What Request Object Will You Receive? (`ToolRequest`)

The Agent passes a strongly-typed `ToolRequest` (`core.schemas.ToolRequest`):

```python
class ToolRequest(BaseModel):
    request_id: str                   # Unique correlation ID
    task: TaskType                    # e.g., TaskType.SINGLE_IMAGE_VQA
    query: str                        # Natural language query
    images: List[ImageInput]          # Canonical ImageInput list (contains path, format, modality, geospatial tags)
    metadata: Dict[str, Any]          # Request-level metadata
    config: Dict[str, Any]            # Runtime parameters
    context: Dict[str, Any]           # Outputs from prior workflow steps (for multi-tool pipelines)
```

---

## 4. What Result Object Must You Return? (`ToolResult`)

Specialists must return a canonical `ToolResult` (`core.schemas.ToolResult`):

```python
class ToolResult(BaseModel):
    request_id: str                          # Matching request_id
    task: TaskType                           # Executed task
    status: ToolStatus                       # ToolStatus.SUCCESS, PARTIAL_SUCCESS, or FAILED
    answer: str                              # Human-readable evidence-grounded answer
    confidence: Optional[float]              # Confidence score between 0.0 and 1.0 (None if uncalibrated)
    evidence: List[Evidence]                 # Bounding boxes, masks, change maps, heatmaps, text snippets
    artifacts: List[Artifact]                # References to saved images/masks in storage
    model_info: Dict[str, Any]               # Model name, version, architecture
    parameters: Dict[str, Any]               # Parameters utilized during inference
    metadata: Dict[str, Any]                 # Arbitrary extra metadata
    execution_trace: List[ExecutionTraceEntry] # Component trace logs
```

---

## 5. Tool Registration & Discovery

### How to Register Your Tool
In your module or at app startup, register your tool with the `default_registry`:

```python
from registry.registry import default_registry
from specialists.single_image.my_tool import MySpecialistTool

default_registry.register(MySpecialistTool())
```

### How the Agent Discovers It
1. The user sends a query + image(s).
2. `IntentResolver` resolves the query into a structured `TaskIntent`.
3. `TaskRouter` queries `default_registry.find_tools_for_task(intent.task, images)`.
4. Your tool is matched based on `supported_tasks`, `min_images`, `max_images`, and `required_modalities`.
5. The `ExecutionEngine` invokes your tool's `execute(request)` method.

---

## 6. How to Run Tests

### Running Contract Tests for Your Tool
Verify your tool satisfies the SatQuery system contract:
```bash
# Run all contract tests
pytest tests/test_contracts.py -v

# Run contract test for a specific tool
pytest tests/test_contracts.py -k "my_unique_tool_name" -v
```

### Running with Mock Specialists
To test the complete agent pipeline without GPU or heavy checkpoints:
```bash
pytest tests/test_routing.py tests/test_execution_engine.py tests/test_api.py -v
```

### Running the End-to-End Test Suite
```bash
pytest -v
```

---

## 7. Starting the API Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`.
