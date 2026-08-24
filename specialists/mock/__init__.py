"""High-fidelity mock specialists package for SatQuery AI."""

from registry.registry import ToolRegistry, default_registry
from specialists.mock.failing_mock import MockFailingTool
from specialists.mock.optical_sar_mock import MockOpticalSARAnalysisTool
from specialists.mock.single_image_mock import (
    MockSingleImageCaptionTool,
    MockSingleImageGroundingTool,
    MockSingleImageVQATool,
)
from specialists.mock.temporal_change_mock import MockBiTemporalChangeTool


def register_default_mocks(registry: ToolRegistry | None = None) -> None:
    """Register standard mock specialist tools into the provided or default registry."""
    reg = registry or default_registry
    reg.register(MockSingleImageVQATool(), overwrite=True)
    reg.register(MockSingleImageCaptionTool(), overwrite=True)
    reg.register(MockSingleImageGroundingTool(), overwrite=True)
    reg.register(MockBiTemporalChangeTool(), overwrite=True)
    reg.register(MockOpticalSARAnalysisTool(), overwrite=True)


__all__ = [
    "MockSingleImageVQATool",
    "MockSingleImageCaptionTool",
    "MockSingleImageGroundingTool",
    "MockBiTemporalChangeTool",
    "MockOpticalSARAnalysisTool",
    "MockFailingTool",
    "register_default_mocks",
]
