# Division 3: Bi-Temporal Change Intelligence

**Owner**: Dheeraj  
**Directory**: `specialists/temporal_change/`  
**Shared Contract**: `core.interfaces.BaseSpecialistTool`

---

## 1. Overview & Responsibilities
Division 3 is responsible for bi-temporal remote-sensing intelligence across pairs of acquisitions (T0 and T1) over the same geographic region:
- **Change Analysis**: Difference detection, change segmentation, and quantitative expansion metrics.
- **Change VQA**: Question answering regarding changes between dates (e.g. "Has built-up area increased?").
- **Change Localization & Maps**: Generating change difference map artifacts and bounding boxes for changed clusters.

---

## 2. Key Architectural Guidelines
1. **Resolution & Dimension Invariance**: Do not require identical pixel dimensions between T0 and T1 images. Different acquisition angles, sensors, or resamplings may yield different raster shapes.
2. **First-Class Evidence & Artifacts**: Return `EvidenceType.CHANGE_MAP` and `EvidenceType.BOUNDING_BOX` for spatial change areas, and write output masks to `settings.artifact_storage_path` as `Artifact` references.

---

## 3. Tool Implementation Example
```python
from core.interfaces import BaseSpecialistTool, ValidationResult
from core.schemas import (
    Artifact,
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

class BiTemporalChangeSpecialist(BaseSpecialistTool):
    def __init__(self):
        super().__init__(
            name="bitemporal_change_specialist",
            description="Production specialist for bi-temporal remote sensing change detection.",
            supported_tasks={TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA},
            version="1.0.0",
            metadata=ToolMetadata(
                name="bitemporal_change_specialist",
                description="Production specialist for bi-temporal remote sensing change detection.",
                version="1.0.0",
                supported_tasks=[TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA],
                required_modalities=[ImageModality.OPTICAL, ImageModality.MULTISPECTRAL, ImageModality.SAR],
                min_images=2,
                max_images=2,
                author_or_division="Division 3 (Dheeraj)",
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        img_t0, img_t1 = request.images[0], request.images[1]
        
        # Perform change detection inference...
        answer = "Detected a 15% increase in urban construction in the south sector."
        evidence = [
            Evidence(
                type=EvidenceType.CHANGE_MAP,
                label="Urban Growth Mask",
                confidence=0.93,
                data={"changed_area_ha": 12.5, "t0_id": img_t0.image_id, "t1_id": img_t1.image_id},
            )
        ]
        artifacts = [
            Artifact(
                name="change_mask.png",
                type="change_map",
                uri_or_path="artifacts_storage/change_mask.png",
                description="Binary change mask for T0/T1.",
            )
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.93,
            evidence=evidence,
            artifacts=artifacts,
            model_info={"name": "BiTemporal-SiamDiff", "version": "1.0"},
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

## 4. Testing Your Specialist
Run contract tests:
```bash
pytest tests/test_contracts.py -k "change"
```
