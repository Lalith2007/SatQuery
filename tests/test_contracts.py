"""Contract tests for specialist tools.

Ensures that all specialist tools (and their mocks) strictly adhere to the
BaseSpecialistTool interface and return schema-compliant ToolResults.
"""

import pytest

from core.interfaces import BaseSpecialistTool
from core.schemas import (
    EvidenceType,
    ImageInput,
    TaskType,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from specialists.mock import (
    MockBiTemporalChangeTool,
    MockOpticalSARAnalysisTool,
    MockSingleImageCaptionTool,
    MockSingleImageGroundingTool,
    MockSingleImageVQATool,
)


@pytest.mark.asyncio
async def test_single_image_vqa_contract(optical_image_input: ImageInput):
    tool = MockSingleImageVQATool()
    assert isinstance(tool, BaseSpecialistTool)

    req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_VQA,
        query="How many aircraft are on the runway?",
        images=[optical_image_input],
    )
    val = tool.validate_request(req)
    assert val.is_valid is True

    result = await tool.execute(req)
    assert isinstance(result, ToolResult)
    assert result.status == ToolStatus.SUCCESS
    assert result.task == TaskType.SINGLE_IMAGE_VQA
    assert len(result.answer) > 0
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.evidence) > 0
    assert result.evidence[0].type == EvidenceType.BOUNDING_BOX
    assert result.evidence[0].confidence is not None
    assert len(result.execution_trace) > 0


@pytest.mark.asyncio
async def test_single_image_caption_contract(optical_image_input: ImageInput):
    tool = MockSingleImageCaptionTool()
    req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_CAPTION,
        query="Describe this scene.",
        images=[optical_image_input],
    )
    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert "landscape" in result.answer.lower() or "view" in result.answer.lower()
    assert len(result.evidence) > 0


@pytest.mark.asyncio
async def test_single_image_grounding_contract(optical_image_input: ImageInput):
    tool = MockSingleImageGroundingTool()
    req = ToolRequest(
        task=TaskType.SINGLE_IMAGE_GROUNDING,
        query="Where is the water body?",
        images=[optical_image_input],
    )
    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert any(e.type == EvidenceType.BOUNDING_BOX for e in result.evidence)
    assert any(e.type == EvidenceType.HEATMAP for e in result.evidence)


@pytest.mark.asyncio
async def test_bitemporal_change_contract(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
):
    tool = MockBiTemporalChangeTool()
    req = ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="What changed between these acquisitions?",
        images=[optical_image_input, optical_image_t1_input],
    )
    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert len(result.evidence) > 0
    assert any(e.type == EvidenceType.CHANGE_MAP for e in result.evidence)
    assert len(result.artifacts) > 0
    assert result.artifacts[0].type == "change_map"


@pytest.mark.asyncio
async def test_optical_sar_contract(
    optical_image_input: ImageInput,
    sar_image_input: ImageInput,
):
    tool = MockOpticalSARAnalysisTool()
    req = ToolRequest(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Combine optical and SAR imagery to identify structures.",
        images=[optical_image_input, sar_image_input],
    )
    result = await tool.execute(req)
    assert result.status == ToolStatus.SUCCESS
    assert any(e.type == EvidenceType.HIGHLIGHTED_IMAGE for e in result.evidence)
    assert len(result.artifacts) > 0
