"""Unit and routing tests for IntentResolver and TaskRouter."""

import pytest

from core.schemas import ImageInput, TaskType
from agent.intent_resolver import IntentResolver
from agent.router import TaskRouter
from registry.registry import ToolRegistry


def test_routing_single_image_vqa(optical_image_input: ImageInput, test_registry: ToolRegistry):
    query = "What is in this image?"
    intent = IntentResolver.resolve_intent(query, images=[optical_image_input])
    assert intent.task == TaskType.SINGLE_IMAGE_VQA
    assert intent.confidence >= 0.85
    assert len(intent.intent_explanation) > 0

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, [optical_image_input])
    assert tool.name == "single_image_vqa_mock"


def test_routing_single_image_caption(optical_image_input: ImageInput, test_registry: ToolRegistry):
    query = "Describe this image."
    intent = IntentResolver.resolve_intent(query, images=[optical_image_input])
    assert intent.task == TaskType.SINGLE_IMAGE_CAPTION

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, [optical_image_input])
    assert tool.name == "single_image_caption_mock"


def test_routing_single_image_grounding(optical_image_input: ImageInput, test_registry: ToolRegistry):
    query = "Where is the water?"
    intent = IntentResolver.resolve_intent(query, images=[optical_image_input])
    assert intent.task == TaskType.SINGLE_IMAGE_GROUNDING
    assert "water" in intent.target_features

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, [optical_image_input])
    assert tool.name == "single_image_grounding_mock"


def test_routing_bitemporal_change_query(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
    test_registry: ToolRegistry,
):
    query = "What changed between these dates?"
    images = [optical_image_input, optical_image_t1_input]
    intent = IntentResolver.resolve_intent(query, images=images)
    assert intent.task in {TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA}

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, images)
    assert tool.name == "bi_temporal_change_mock"


def test_routing_bitemporal_builtup_increase(
    optical_image_input: ImageInput,
    optical_image_t1_input: ImageInput,
    test_registry: ToolRegistry,
):
    query = "Has built-up area increased?"
    images = [optical_image_input, optical_image_t1_input]
    intent = IntentResolver.resolve_intent(query, images=images)
    assert intent.task in {TaskType.CHANGE_ANALYSIS, TaskType.CHANGE_VQA}

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, images)
    assert tool.name == "bi_temporal_change_mock"


def test_routing_optical_sar_cross_modal(
    optical_image_input: ImageInput,
    sar_image_input: ImageInput,
    test_registry: ToolRegistry,
):
    query = "Use optical and SAR together to detect buildings."
    images = [optical_image_input, sar_image_input]
    intent = IntentResolver.resolve_intent(query, images=images)
    assert intent.task == TaskType.OPTICAL_SAR_ANALYSIS

    router = TaskRouter(test_registry)
    tool = router.select_tool(intent, images)
    assert tool.name == "optical_sar_cross_modal_mock"
