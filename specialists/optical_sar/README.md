# Division 4: Optical-SAR Cross-Modal Intelligence

**Owner**: Laksh  
**Directory**: `specialists/optical_sar/`  
**Shared Contract**: `core.interfaces.BaseSpecialistTool`

---

## 1. Overview & Responsibilities
Division 4 is responsible for cross-modal intelligence that jointly processes co-registered Optical/Multispectral and Synthetic Aperture Radar (SAR) imagery:
- **Joint Analysis & Cross-Modal Fusion**: Combining optical spectral richness (RGB/NIR) with SAR physical structural backscatter (VV/VH, roughness, moisture, cloud penetration).
- **All-Weather Feature Detection**: Resolving targets obscured by clouds, haze, or shadows in optical imagery using SAR data.

---

## 2. Key Architectural Guidelines
1. **Heterogeneous Raster Dimensions**: Optical and SAR rasters often have different spatial resolutions and pixel grid sizes. Do not assume identical dimensions.
2. **Modality Validation**: The agent ensures that input images include at least one Optical/Multispectral image and one SAR image.

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

class OpticalSARSpecialist(BaseSpecialistTool):
    def __init__(self):
        super().__init__(
            name="optical_sar_cross_modal_specialist",
            description="Production specialist for cross-modal optical and SAR feature fusion.",
            supported_tasks={TaskType.OPTICAL_SAR_ANALYSIS},
            version="1.0.0",
            metadata=ToolMetadata(
                name="optical_sar_cross_modal_specialist",
                description="Production specialist for cross-modal optical and SAR feature fusion.",
                version="1.0.0",
                supported_tasks=[TaskType.OPTICAL_SAR_ANALYSIS],
                required_modalities=[ImageModality.OPTICAL, ImageModality.SAR],
                min_images=2,
                max_images=2,
                author_or_division="Division 4 (Laksh)",
            ),
        )

    async def execute(self, request: ToolRequest) -> ToolResult:
        optical_img = next(img for img in request.images if img.modality != ImageModality.SAR)
        sar_img = next(img for img in request.images if img.modality == ImageModality.SAR)

        # Run cross-modal fusion inference...
        answer = "Fused optical and SAR features successfully resolved metallic structures beneath cloud cover."
        evidence = [
            Evidence(
                type=EvidenceType.HIGHLIGHTED_IMAGE,
                label="SAR Enhanced Structural Mask",
                confidence=0.95,
                data={"sar_intensity_db": -10.5},
                image_id=sar_img.image_id,
            )
        ]

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=0.95,
            evidence=evidence,
            artifacts=[],
            model_info={"name": "CrossModal-Attention-Fusion", "version": "1.0"},
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

## 4. Running Contract Tests
```bash
pytest tests/test_contracts.py -k "optical_sar"
```
