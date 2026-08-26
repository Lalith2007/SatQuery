"""Comprehensive test suite for Division 3: Bi-Temporal Change Intelligence.

6 Test Gates:
  Gate 1: Unit tests (validation, preprocessing, postprocessing)
  Gate 2: Contract tests (BaseSpecialistTool, ToolResult schema)
  Gate 3: Mock-service tests (full pipeline without production weights)
  Gate 4: Real-model/sample inference test (public sample data)
  Gate 5: Registry invocation test
  Gate 6: Agent-level integration test (ExecutionEngine)
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from core.interfaces import BaseSpecialistTool, ValidationResult
from core.schemas import (
    EvidenceType,
    GeoSpatialMetadata,
    ImageFormat,
    ImageInput,
    ImageModality,
    TaskType,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def t0_image_path(temp_dir: Path) -> Path:
    """Create a T0 sample image."""
    path = temp_dir / "t0.png"
    arr = np.random.randint(0, 200, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def t1_image_path(temp_dir: Path) -> Path:
    """Create a T1 sample image (slightly different from T0)."""
    path = temp_dir / "t1.png"
    arr = np.random.randint(50, 255, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def t0_input(t0_image_path: Path) -> ImageInput:
    return ImageInput(
        image_id="t0-test",
        path_or_uri=str(t0_image_path),
        format=ImageFormat.PNG,
        modality=ImageModality.OPTICAL,
        width=128,
        height=128,
        channel_count=3,
        dtype="uint8",
        geospatial=GeoSpatialMetadata(
            crs="EPSG:4326",
            geo_bounds=[-122.5, 37.7, -122.3, 37.9],
            acquisition_timestamp=datetime(2023, 6, 1, tzinfo=timezone.utc),
            sensor="Sentinel-2",
        ),
    )


@pytest.fixture
def t1_input(t1_image_path: Path) -> ImageInput:
    return ImageInput(
        image_id="t1-test",
        path_or_uri=str(t1_image_path),
        format=ImageFormat.PNG,
        modality=ImageModality.OPTICAL,
        width=128,
        height=128,
        channel_count=3,
        dtype="uint8",
        geospatial=GeoSpatialMetadata(
            crs="EPSG:4326",
            geo_bounds=[-122.5, 37.7, -122.3, 37.9],
            acquisition_timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
            sensor="Sentinel-2",
        ),
    )


@pytest.fixture
def change_request(t0_input, t1_input) -> ToolRequest:
    return ToolRequest(
        task=TaskType.CHANGE_ANALYSIS,
        query="What changed between these two dates?",
        images=[t0_input, t1_input],
    )


@pytest.fixture
def change_vqa_request(t0_input, t1_input) -> ToolRequest:
    return ToolRequest(
        task=TaskType.CHANGE_VQA,
        query="Has the built-up area increased?",
        images=[t0_input, t1_input],
    )


# ============================================================
# Gate 1: Unit Tests
# ============================================================

class TestGate1Validation:
    """Unit tests for validation, preprocessing, postprocessing."""

    def test_validate_image_count_valid(self, change_request):
        from specialists.temporal_change.validation import validate_image_count
        result = validate_image_count(change_request)
        assert result.passed is True

    def test_validate_image_count_invalid(self, t0_input):
        from specialists.temporal_change.validation import validate_image_count
        req = ToolRequest(task=TaskType.CHANGE_ANALYSIS, query="test", images=[t0_input])
        result = validate_image_count(req)
        assert result.passed is False
        assert "2 images" in result.message

    def test_validate_file_readability(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_file_readability
        result = validate_file_readability([t0_input, t1_input])
        assert result.passed is True

    def test_validate_file_readability_missing(self, t0_input):
        bad = ImageInput(
            image_id="bad",
            path_or_uri="/nonexistent/path.png",
            format=ImageFormat.PNG,
        )
        from specialists.temporal_change.validation import validate_file_readability
        result = validate_file_readability([t0_input, bad])
        assert result.passed is False

    def test_validate_dimensions_match(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_dimensions
        result = validate_dimensions(t0_input, t1_input)
        assert result.passed is True
        assert result.details.get("dimensions_match") is True

    def test_validate_dimensions_channel_mismatch(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_dimensions
        t1_input.channel_count = 4
        result = validate_dimensions(t0_input, t1_input)
        assert result.passed is False
        assert "Channel count" in result.message

    def test_validate_crs_match(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_crs
        result = validate_crs(t0_input, t1_input)
        assert result.passed is True

    def test_validate_crs_mismatch(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_crs
        t1_input.geospatial.crs = "EPSG:32610"
        result = validate_crs(t0_input, t1_input)
        assert result.passed is False

    def test_validate_timestamps(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_timestamps
        result = validate_timestamps(t0_input, t1_input)
        assert result.passed is True
        assert result.details.get("temporal_gap_days") == 366

    def test_spatial_compatibility_aligned(self, t0_input, t1_input):
        from specialists.temporal_change.validation import validate_spatial_compatibility
        result, status = validate_spatial_compatibility(t0_input, t1_input)
        assert result.passed is True
        assert status == "aligned"

    def test_spatial_compatibility_dim_mismatch_no_geo(self, temp_dir):
        """Dimension mismatch without geo metadata -> incompatible."""
        from specialists.temporal_change.validation import validate_spatial_compatibility
        t0 = ImageInput(image_id="a", path_or_uri="x", format=ImageFormat.PNG, width=128, height=128)
        t1 = ImageInput(image_id="b", path_or_uri="y", format=ImageFormat.PNG, width=256, height=256)
        result, status = validate_spatial_compatibility(t0, t1)
        assert result.passed is False
        assert status == "incompatible"

    def test_full_pair_validation(self, change_request):
        from specialists.temporal_change.validation import validate_pair
        result = validate_pair(change_request)
        assert result.is_valid is True
        assert len(result.stages) == 6
        assert result.alignment_status == "aligned"


class TestGate1Preprocessing:
    """Unit tests for image preprocessing."""

    def test_load_image(self, t0_image_path):
        from specialists.temporal_change.preprocessing import load_image_as_array
        arr = load_image_as_array(str(t0_image_path))
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8

    def test_resize_image(self):
        from specialists.temporal_change.preprocessing import resize_image
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize_image(arr, 256)
        assert resized.shape == (256, 256, 3)

    def test_preprocess_pair(self, t0_image_path, t1_image_path):
        from specialists.temporal_change.preprocessing import preprocess_pair
        t0, t1, meta = preprocess_pair(str(t0_image_path), str(t1_image_path), model_input_size=64)
        assert t0.shape == (64, 64, 3)
        assert t1.shape == (64, 64, 3)
        assert t0.dtype == np.float32
        assert "preprocessing_time_ms" in meta


class TestGate1Postprocessing:
    """Unit tests for postprocessing."""

    def test_threshold_probability_map(self):
        from specialists.temporal_change.postprocessing import threshold_probability_map
        prob = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=np.float32)
        binary = threshold_probability_map(prob, threshold=0.5)
        assert binary.dtype == np.uint8
        assert binary[0, 0] == 0
        assert binary[0, 1] == 1

    def test_postprocess_change_map(self):
        from specialists.temporal_change.postprocessing import postprocess_change_map
        prob = np.random.rand(64, 64).astype(np.float32)
        result = postprocess_change_map(prob_map=prob, threshold=0.5, min_region_area=1)
        assert result.binary_map.shape == (64, 64)
        assert 0.0 <= result.changed_pixel_ratio <= 1.0
        assert result.total_pixel_count == 64 * 64

    def test_extract_connected_regions(self):
        from specialists.temporal_change.postprocessing import extract_connected_regions
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[10:20, 10:20] = 1  # 100-pixel region
        binary[40:50, 40:50] = 1  # 100-pixel region
        regions = extract_connected_regions(binary, min_area_pixels=50)
        assert len(regions) == 2
        assert all(r.area_pixels >= 50 for r in regions)


class TestGate1QueryIntent:
    """Unit tests for query intent classification."""

    def test_general_change(self):
        from specialists.temporal_change.semantic_reasoning import classify_query_intent
        from specialists.temporal_change.interfaces import QueryIntent
        assert classify_query_intent("What changed?") == QueryIntent.GENERAL_CHANGE

    def test_localization(self):
        from specialists.temporal_change.semantic_reasoning import classify_query_intent
        from specialists.temporal_change.interfaces import QueryIntent
        assert classify_query_intent("Where did the change occur?") == QueryIntent.CHANGE_LOCALIZATION

    def test_built_up(self):
        from specialists.temporal_change.semantic_reasoning import classify_query_intent
        from specialists.temporal_change.interfaces import QueryIntent
        assert classify_query_intent("Has the built-up area increased?") == QueryIntent.BUILT_UP_CHANGE

    def test_vegetation(self):
        from specialists.temporal_change.semantic_reasoning import classify_query_intent
        from specialists.temporal_change.interfaces import QueryIntent
        assert classify_query_intent("Did vegetation decrease?") == QueryIntent.VEGETATION_CHANGE

    def test_water(self):
        from specialists.temporal_change.semantic_reasoning import classify_query_intent
        from specialists.temporal_change.interfaces import QueryIntent
        assert classify_query_intent("Did the water body expand?") == QueryIntent.WATER_CHANGE


# ============================================================
# Gate 2: Contract Tests
# ============================================================

class TestGate2Contract:
    """Contract tests: BaseSpecialistTool compliance and ToolResult schema."""

    def test_inherits_base_specialist(self):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        assert isinstance(tool, BaseSpecialistTool)

    def test_supported_tasks(self):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        assert TaskType.CHANGE_ANALYSIS in tool.supported_tasks
        assert TaskType.CHANGE_VQA in tool.supported_tasks

    def test_metadata_fields(self):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        meta = tool.metadata
        assert meta.min_images == 2
        assert meta.max_images == 2
        assert "Division 3" in meta.author_or_division

    def test_validate_request_valid(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        result = tool.validate_request(change_request)
        assert result.is_valid is True

    def test_validate_request_wrong_task(self, t0_input, t1_input):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        req = ToolRequest(
            task=TaskType.SINGLE_IMAGE_VQA,
            query="test",
            images=[t0_input, t1_input],
        )
        result = tool.validate_request(req)
        assert result.is_valid is False

    def test_validate_request_wrong_image_count(self, t0_input):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        req = ToolRequest(task=TaskType.CHANGE_ANALYSIS, query="test", images=[t0_input])
        result = tool.validate_request(req)
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        result = await tool.execute(change_request)
        assert isinstance(result, ToolResult)
        assert result.request_id == change_request.request_id
        assert result.task == TaskType.CHANGE_ANALYSIS

    @pytest.mark.asyncio
    async def test_result_has_evidence(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        result = await tool.execute(change_request)
        assert result.status == ToolStatus.SUCCESS
        assert len(result.evidence) > 0
        assert any(e.type == EvidenceType.CHANGE_MAP for e in result.evidence)

    @pytest.mark.asyncio
    async def test_result_has_execution_trace(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        result = await tool.execute(change_request)
        assert len(result.execution_trace) > 0

    @pytest.mark.asyncio
    async def test_confidence_breakdown_not_fabricated(self, change_request):
        """Verify confidence breakdown exposes separate metrics without fabrication."""
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        result = await tool.execute(change_request)
        breakdown = result.metadata.get("confidence_breakdown", {})
        assert "change_confidence" in breakdown
        assert "semantic_confidence" in breakdown
        assert "overall_confidence" in breakdown
        # Overall must be None unless defensible calibration exists
        assert breakdown["overall_confidence"] is None


# ============================================================
# Gate 3: Mock-Service Tests
# ============================================================

class TestGate3MockService:
    """Mock-service tests: full pipeline without production weights."""

    @pytest.mark.asyncio
    async def test_mock_change_analysis(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool, MockChangeModel, MockSemanticReasoner
        tool = BiTemporalChangeSpecialistTool(
            change_model=MockChangeModel(),
            semantic_reasoner=MockSemanticReasoner(),
        )
        result = await tool.execute(change_request)
        assert result.status == ToolStatus.SUCCESS
        assert "[Mock]" in result.answer or "change" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_mock_change_vqa(self, change_vqa_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool, MockChangeModel, MockSemanticReasoner
        tool = BiTemporalChangeSpecialistTool(
            change_model=MockChangeModel(),
            semantic_reasoner=MockSemanticReasoner(),
        )
        result = await tool.execute(change_vqa_request)
        assert result.status == ToolStatus.SUCCESS
        assert result.task == TaskType.CHANGE_VQA

    @pytest.mark.asyncio
    async def test_mock_identifies_itself(self, change_request):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool, MockChangeModel
        tool = BiTemporalChangeSpecialistTool(change_model=MockChangeModel())
        result = await tool.execute(change_request)
        assert "MockChangeModel" in result.model_info.get("change_model", "")

    @pytest.mark.asyncio
    async def test_mock_failure_invalid_image_count(self, t0_input):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        req = ToolRequest(task=TaskType.CHANGE_ANALYSIS, query="test", images=[t0_input])
        result = await tool.execute(req)
        assert result.status == ToolStatus.FAILED

    @pytest.mark.asyncio
    async def test_mock_failure_unsupported_task(self, t0_input, t1_input):
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        tool = BiTemporalChangeSpecialistTool()
        req = ToolRequest(
            task=TaskType.SINGLE_IMAGE_VQA,
            query="test",
            images=[t0_input, t1_input],
        )
        result = await tool.execute(req)
        assert result.status == ToolStatus.FAILED


# ============================================================
# Gate 4: Sample Inference Test
# ============================================================

class TestGate4SampleInference:
    """Real-model/sample inference test with synthetic public sample data."""

    @pytest.mark.asyncio
    async def test_sample_change_detection(self, temp_dir):
        """Run a full inference with mock model on synthetic sample images."""
        from specialists.temporal_change import BiTemporalChangeSpecialistTool
        from specialists.temporal_change.config import TemporalChangeConfig

        # Create sample images with visible difference
        t0 = np.zeros((128, 128, 3), dtype=np.uint8)
        t0[20:60, 20:60] = [0, 128, 0]  # Green patch

        t1 = np.zeros((128, 128, 3), dtype=np.uint8)
        t1[20:60, 20:60] = [200, 200, 200]  # Gray patch (simulates new construction)

        t0_path = temp_dir / "sample_t0.png"
        t1_path = temp_dir / "sample_t1.png"
        Image.fromarray(t0).save(t0_path)
        Image.fromarray(t1).save(t1_path)

        t0_img = ImageInput(
            image_id="sample-t0",
            path_or_uri=str(t0_path),
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            width=128, height=128, channel_count=3, dtype="uint8",
        )
        t1_img = ImageInput(
            image_id="sample-t1",
            path_or_uri=str(t1_path),
            format=ImageFormat.PNG,
            modality=ImageModality.OPTICAL,
            width=128, height=128, channel_count=3, dtype="uint8",
        )

        config = TemporalChangeConfig()
        config.artifact_output_dir = temp_dir / "artifacts"
        config.use_mock_model = True

        tool = BiTemporalChangeSpecialistTool(config=config)
        req = ToolRequest(
            task=TaskType.CHANGE_ANALYSIS,
            query="What changed between these two dates?",
            images=[t0_img, t1_img],
        )
        result = await tool.execute(req)

        assert result.status == ToolStatus.SUCCESS
        assert len(result.evidence) > 0
        assert len(result.answer) > 0
        assert len(result.execution_trace) > 0

        # Verify artifacts were generated
        assert len(result.artifacts) > 0
        for a in result.artifacts:
            assert a.type == "change_map"


# ============================================================
# Gate 5: Registry Invocation Test
# ============================================================

class TestGate5Registry:
    """Registry invocation tests."""

    def test_register_and_retrieve(self):
        from registry.registry import ToolRegistry
        from specialists.temporal_change import register_temporal_change_specialist
        reg = ToolRegistry()
        tool = register_temporal_change_specialist(reg)
        assert reg.has_tool("bitemporal_change_specialist")
        retrieved = reg.get("bitemporal_change_specialist")
        assert retrieved is tool

    def test_find_tools_for_change_analysis(self):
        from registry.registry import ToolRegistry
        from specialists.temporal_change import register_temporal_change_specialist
        reg = ToolRegistry()
        register_temporal_change_specialist(reg)
        tools = reg.find_tools_for_task(TaskType.CHANGE_ANALYSIS)
        assert len(tools) >= 1
        assert any(t.name == "bitemporal_change_specialist" for t in tools)

    def test_find_tools_for_change_vqa(self):
        from registry.registry import ToolRegistry
        from specialists.temporal_change import register_temporal_change_specialist
        reg = ToolRegistry()
        register_temporal_change_specialist(reg)
        tools = reg.find_tools_for_task(TaskType.CHANGE_VQA)
        assert len(tools) >= 1

    def test_health_check(self):
        from registry.registry import ToolRegistry
        from specialists.temporal_change import register_temporal_change_specialist
        reg = ToolRegistry()
        register_temporal_change_specialist(reg)
        health = reg.health_check_all()
        assert health.get("bitemporal_change_specialist") is True


# ============================================================
# Gate 6: Agent-Level Integration Test
# ============================================================

class TestGate6Integration:
    """Agent-level integration test via ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_execution_engine_invocation(self, change_request):
        from registry.registry import ToolRegistry
        from specialists.temporal_change import register_temporal_change_specialist
        from agent.execution_engine import ExecutionEngine
        from agent.workflow import WorkflowPlan, WorkflowStep
        from core.schemas import TaskIntent

        reg = ToolRegistry()
        register_temporal_change_specialist(reg)
        engine = ExecutionEngine(registry=reg, default_timeout_seconds=60)

        plan = WorkflowPlan(
            plan_id="test-plan-1",
            intent=TaskIntent(
                task=TaskType.CHANGE_ANALYSIS,
                confidence=1.0,
                intent_explanation="Test bi-temporal change detection",
            ),
            steps=[
                WorkflowStep(
                    step_index=0,
                    task=TaskType.CHANGE_ANALYSIS,
                    tool_name="bitemporal_change_specialist",
                    purpose="Bi-temporal change detection",
                ),
            ],
        )

        results = await engine.execute_plan(plan, change_request)
        assert len(results) == 1
        assert results[0].status == ToolStatus.SUCCESS
        assert results[0].task == TaskType.CHANGE_ANALYSIS
