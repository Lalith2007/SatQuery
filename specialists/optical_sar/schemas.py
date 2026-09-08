"""Validation schemas, adapters, and helper models for Division 4 Optical-SAR Specialist.

Extends canonical core schemas (core.schemas.ToolRequest, core.schemas.ToolResult)
with Optical-SAR domain specific validation rules.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.schemas import ImageInput, ImageModality, TaskType, ToolRequest


class OpticalSarValidationResult(BaseModel):
    """Validation result specific to Optical-SAR cross-modal inputs."""
    is_valid: bool = Field(description="True if request contains valid co-registered Optical and SAR images")
    errors: List[str] = Field(default_factory=list, description="Validation failure messages")
    optical_image: Optional[ImageInput] = Field(default=None, description="Extracted Optical/Multispectral image input")
    sar_image: Optional[ImageInput] = Field(default=None, description="Extracted SAR image input")


class OpticalSarInputs(BaseModel):
    """Container for validated Optical and SAR image inputs."""
    optical: ImageInput = Field(description="Optical/Multispectral image input")
    sar: ImageInput = Field(description="SAR image input")
    query: str = Field(description="Natural language query string")
    request_id: str = Field(description="Correlated request ID")


def extract_and_validate_optical_sar_inputs(request: ToolRequest) -> OpticalSarValidationResult:
    """Extract and validate Optical and SAR image inputs from a canonical ToolRequest.
    
    Verifies:
    1. Task compatibility (TaskType.OPTICAL_SAR_ANALYSIS)
    2. Presence of at least 1 Optical/Multispectral image
    3. Presence of at least 1 SAR image
    """
    errors: List[str] = []

    if request.task != TaskType.OPTICAL_SAR_ANALYSIS:
        errors.append(
            f"Task '{request.task.value}' is not supported. Division 4 requires '{TaskType.OPTICAL_SAR_ANALYSIS.value}'."
        )

    if not request.images or len(request.images) < 2:
        errors.append(
            f"Optical-SAR specialist requires at least 2 input images (1 Optical + 1 SAR). Received {len(request.images)}."
        )
        return OpticalSarValidationResult(is_valid=False, errors=errors)

    optical_img: Optional[ImageInput] = None
    sar_img: Optional[ImageInput] = None

    for img in request.images:
        if img.modality in {ImageModality.OPTICAL, ImageModality.MULTISPECTRAL} and optical_img is None:
            optical_img = img
        elif img.modality == ImageModality.SAR and sar_img is None:
            sar_img = img

    # Fallback heuristic if modalities are UNKNOWN: inspect file extension or sequence
    if optical_img is None or sar_img is None:
        for idx, img in enumerate(request.images):
            path_lower = img.path_or_uri.lower()
            if optical_img is None and ("opt" in path_lower or "optical" in path_lower or "rgb" in path_lower or idx == 0):
                optical_img = img
            elif sar_img is None and ("sar" in path_lower or "rad" in path_lower or "vv" in path_lower or idx == 1):
                sar_img = img

    if optical_img is None:
        errors.append("No Optical or Multispectral image input identified in request.")
    if sar_img is None:
        errors.append("No Synthetic Aperture Radar (SAR) image input identified in request.")

    if errors:
        return OpticalSarValidationResult(is_valid=False, errors=errors)

    return OpticalSarValidationResult(
        is_valid=True,
        errors=[],
        optical_image=optical_img,
        sar_image=sar_img,
    )
