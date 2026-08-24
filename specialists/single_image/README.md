# Division 2: Single-Image Remote-Sensing Intelligence

**Owner**: Sruthi  
**Directory**: `specialists/single_image/`  
**Shared Contract**: `core.interfaces.BaseSpecialistTool`

---

## 1. Overview & Responsibilities
Division 2 is responsible for implementing specialist intelligence models for single remote-sensing images:
- **Visual Question Answering (VQA)**: Single-image queries across Optical, Multispectral, and SAR imagery.
- **Scene Captioning**: Natural-language descriptions of remote-sensing scenes.
- **Spatial Grounding / Localization**: Bounding-box and heatmap localization for spatial queries (e.g., "Where is the water?").

---

## 2. Interface to Implement
Your specialist classes must inherit from `core.interfaces.BaseSpecialistTool`:

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

class SingleImageVQASpecialist(BaseSpecialistTool):
    def __init__(self, checkpoint_path: str = "models/vqa_weights.pth"):
        super().__init__(
            name="single_image_vqa_specialist",
            description="Production specialist for single-image remote sensing VQA.",
            supported_tasks={TaskType.SINGLE_IMAGE_VQA},
            version="1.0.0",
            metadata=ToolMetadata(
                name="single_image_vqa_specialist",
                description="Production specialist for single-image remote sensing VQA.",
                version="1.0.0",
                supported_tasks=[TaskType.SINGLE_IMAGE_VQA],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=1,
                max_images=1,
                author_or_division="Division 2 (Sruthi)",
            ),
        )
        self.checkpoint_path = checkpoint_path
        # Load your model checkpoint here

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        if len(request.images) != 1:
            return ValidationResult(is_valid=False, errors=["Requires exactly 1 image."])
        return ValidationResult(is_valid=True)

    async def execute(self, request: ToolRequest) -> ToolResult:
        image = request.images[0]
        # Run your model inference...
        answer = "Detected 4 aircraft parked on the taxiway."
        evidence = [
            Evidence(
                type=EvidenceType.BOUNDING_BOX,
                label="Aircraft 1",
                confidence=0.94,
                data={"bbox": [0.22, 0.35, 0.28, 0.42], "format": "[ymin, xmin, ymax, xmax]"},
                image_id=image.image_id,
            )
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.92,
            evidence=evidence,
            artifacts=[],
            model_info={"name": "RS-VQA-Transformer", "version": "1.0"},
            execution_trace=[
                ExecutionTraceEntry(
                    stage=ExecutionStage.INFERENCE_EXECUTED,
                    component=self.name,
                    status="COMPLETED",
                )
            ],
        )
```

---

## 3. How to Register Your Tool
In `registry/registry.py` or during app startup:

```python
from registry.registry import default_registry
from specialists.single_image.vqa import SingleImageVQASpecialist

default_registry.register(SingleImageVQASpecialist())
```

---

## 4. Running Contract Tests
Ensure your tool satisfies all system contracts by running:
```bash
pytest tests/test_contracts.py -k "single_image"
```
