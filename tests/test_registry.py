"""Unit tests for the specialist ToolRegistry."""

import pytest

from core.errors import ToolNotFoundError
from core.schemas import ImageFormat, ImageInput, ImageModality, TaskType
from registry.registry import ToolRegistry
from specialists.mock import (
    MockBiTemporalChangeTool,
    MockOpticalSARAnalysisTool,
    MockSingleImageVQATool,
)


def test_registry_register_and_lookup():
    registry = ToolRegistry()
    vqa_tool = MockSingleImageVQATool()
    registry.register(vqa_tool)

    assert registry.has_tool("single_image_vqa_mock")
    retrieved = registry.get("single_image_vqa_mock")
    assert retrieved.name == "single_image_vqa_mock"


def test_registry_unregister():
    registry = ToolRegistry()
    vqa_tool = MockSingleImageVQATool()
    registry.register(vqa_tool)

    unregistered = registry.unregister("single_image_vqa_mock")
    assert unregistered is not None
    assert not registry.has_tool("single_image_vqa_mock")


def test_registry_tool_not_found():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("non_existent_tool")


def test_registry_find_tools_for_task():
    registry = ToolRegistry()
    registry.register(MockSingleImageVQATool())
    registry.register(MockBiTemporalChangeTool())
    registry.register(MockOpticalSARAnalysisTool())

    # Find for single image VQA
    tools = registry.find_tools_for_task(TaskType.SINGLE_IMAGE_VQA)
    assert len(tools) == 1
    assert tools[0].name == "single_image_vqa_mock"

    # Find for change analysis
    change_tools = registry.find_tools_for_task(TaskType.CHANGE_ANALYSIS)
    assert len(change_tools) == 1
    assert change_tools[0].name == "bi_temporal_change_mock"


def test_registry_health_check_all():
    registry = ToolRegistry()
    registry.register(MockSingleImageVQATool())
    health = registry.health_check_all()
    assert health.get("single_image_vqa_mock") is True
