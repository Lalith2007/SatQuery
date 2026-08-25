"""Division 2: Single-Image Remote-Sensing Intelligence.

Owner: Sruthi
Model: PaliGemma 3B — SatQuery Remote-Sensing Adapted
"""

from __future__ import annotations

from registry.registry import ToolRegistry, default_registry
from specialists.single_image.grounding import GroundingCoordinateParser
from specialists.single_image.model import PaliGemmaRSInferenceEngine
from specialists.single_image.specialist import SingleImageRSSpecialistTool


def register_single_image_specialist(registry: ToolRegistry | None = None) -> SingleImageRSSpecialistTool:
    """Register the real Division 2 SingleImageRSSpecialistTool into the provided or default registry."""
    reg = registry or default_registry
    tool = SingleImageRSSpecialistTool()
    reg.register(tool, overwrite=True)
    return tool


__all__ = [
    "SingleImageRSSpecialistTool",
    "PaliGemmaRSInferenceEngine",
    "GroundingCoordinateParser",
    "register_single_image_specialist",
]
