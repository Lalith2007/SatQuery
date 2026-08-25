"""Canonical schemas and shared data contracts for SatQuery AI.

Division 1 owns these schemas to serve as the unified, strongly-typed
communication backbone across all specialist divisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field, field_validator


class ImageModality(str, Enum):
    """Remote-sensing image acquisition modality."""
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    """Supported remote-sensing and benchmark image formats."""
    GEOTIFF = "geotiff"
    TIFF = "tiff"
    PNG = "png"
    JPEG = "jpeg"


class ImagePairType(str, Enum):
    """Type of image input configuration."""
    SINGLE = "single"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR_CROSS_MODAL = "optical_sar_cross_modal"
    MULTI_IMAGE = "multi_image"


class TaskType(str, Enum):
    """Controlled vocabulary for remote-sensing vision-language tasks."""
    SINGLE_IMAGE_VQA = "single_image_vqa"
    SINGLE_IMAGE_CAPTION = "single_image_caption"
    SINGLE_IMAGE_GROUNDING = "single_image_grounding"
    CHANGE_ANALYSIS = "change_analysis"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"


class EvidenceType(str, Enum):
    """Standardized evidence representation types."""
    BOUNDING_BOX = "bounding_box"
    MASK = "mask"
    CHANGE_MAP = "change_map"
    HIGHLIGHTED_IMAGE = "highlighted_image"
    CROP = "crop"
    HEATMAP = "heatmap"
    TEXT_EVIDENCE = "text_evidence"


class ToolStatus(str, Enum):
    """Execution status returned by tools and the agent."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ExecutionStage(str, Enum):
    """Auditable stages in the operational execution pipeline."""
    REQUEST_RECEIVED = "REQUEST_RECEIVED"
    INPUT_VALIDATED = "INPUT_VALIDATED"
    TASK_RESOLVED = "TASK_RESOLVED"
    TOOL_SELECTED = "TOOL_SELECTED"
    MODEL_INITIALIZED = "MODEL_INITIALIZED"
    INFERENCE_EXECUTED = "INFERENCE_EXECUTED"
    EVIDENCE_GENERATED = "EVIDENCE_GENERATED"
    RESULT_AGGREGATED = "RESULT_AGGREGATED"
    RESULT_RETURNED = "RESULT_RETURNED"
    ERROR_ENCOUNTERED = "ERROR_ENCOUNTERED"


# ---------------------------------------------------------------------------
# Core Schema Models
# ---------------------------------------------------------------------------

class GeoSpatialMetadata(BaseModel):
    """Geospatial metadata associated with a remote-sensing raster.
    
    Fields are Optional to avoid fabricating missing CRS, bounds, or timestamps.
    """
    crs: Optional[str] = Field(default=None, description="Coordinate reference system (e.g. EPSG:4326)")
    geo_bounds: Optional[List[float]] = Field(default=None, description="Bounding coordinates [minx, miny, maxx, maxy]")
    resolution: Optional[List[float]] = Field(default=None, description="Pixel resolution [x_res, y_res]")
    nodata_value: Optional[float] = Field(default=None, description="Raster nodata pixel value")
    acquisition_timestamp: Optional[datetime] = Field(default=None, description="UTC acquisition timestamp")
    sensor: Optional[str] = Field(default=None, description="Sensor name (e.g. Sentinel-2, Sentinel-1, Landsat-8)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional verified metadata")


class ImageInput(BaseModel):
    """Canonical representation of an input raster image."""
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the image")
    path_or_uri: str = Field(description="File system path or URI to the raster")
    format: ImageFormat = Field(description="Image format (strictly GeoTIFF, TIFF, PNG, JPEG)")
    modality: ImageModality = Field(default=ImageModality.UNKNOWN, description="Acquisition modality")
    width: Optional[int] = Field(default=None, ge=1, description="Image width in pixels")
    height: Optional[int] = Field(default=None, ge=1, description="Image height in pixels")
    channel_count: Optional[int] = Field(default=None, ge=1, description="Number of spectral/polarimetric channels")
    dtype: Optional[str] = Field(default=None, description="Data type of the raster (e.g. uint8, uint16, float32)")
    geospatial: Optional[GeoSpatialMetadata] = Field(default=None, description="Optional georeferencing metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary unverified metadata")


class TaskIntent(BaseModel):
    """Structured intent resolved from a natural-language query."""
    task: TaskType = Field(description="Resolved remote-sensing task identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in intent resolution")
    intent_explanation: str = Field(default="", description="Operational rationale for task resolution (no private CoT)")
    target_features: List[str] = Field(default_factory=list, description="Extracted targets (e.g., runways, water, urban expansion)")
    extracted_parameters: Dict[str, Any] = Field(default_factory=dict, description="Key parameters extracted from the query")


class Evidence(BaseModel):
    """First-class evidence object grounding the model's answer."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique evidence ID")
    type: EvidenceType = Field(description="Type of evidence")
    label: str = Field(description="Descriptive label (e.g. 'Detected water body', 'Deforestation polygon')")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score if available")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured spatial coordinates, bbox [ymin, xmin, ymax, xmax], or mask refs")
    image_id: Optional[str] = Field(default=None, description="Associated image_id if specific to an input")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata for presentation rendering")


class Artifact(BaseModel):
    """Reference to a generated output artifact (map, mask, visualization, report)."""
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique artifact identifier")
    name: str = Field(description="Human-readable name")
    type: str = Field(description="Artifact type descriptor (e.g. 'change_map', 'segmented_mask', 'annotated_image')")
    uri_or_path: str = Field(description="File system path or accessible URI")
    description: str = Field(default="", description="Description of the artifact contents")
    mime_type: Optional[str] = Field(default=None, description="MIME type if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class ExecutionTraceEntry(BaseModel):
    """Single operational audit log entry for system execution."""
    stage: ExecutionStage = Field(description="Execution pipeline stage")
    component: str = Field(description="Component name responsible for the stage")
    status: str = Field(description="Status of the stage: STARTED, COMPLETED, FAILED, SKIPPED")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp of the event")
    duration_ms: Optional[float] = Field(default=None, ge=0.0, description="Duration of stage in milliseconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Operational audit details (no private CoT)")


class ToolRequest(BaseModel):
    """Canonical request payload passed to specialist tools."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request tracing ID")
    task: TaskType = Field(description="Normalized task type")
    query: str = Field(description="Original or refined natural language query")
    images: List[ImageInput] = Field(description="List of validated canonical input images")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request-level metadata")
    config: Dict[str, Any] = Field(default_factory=dict, description="Permitted runtime configuration")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context/outputs passed from prior workflow steps")


class ToolResult(BaseModel):
    """Canonical result object returned by specialist tools."""
    request_id: str = Field(description="Matching request correlation ID")
    task: TaskType = Field(description="Task executed")
    status: ToolStatus = Field(description="Execution status")
    answer: str = Field(description="Evidence-grounded natural language answer")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Calibrated confidence if meaningful, else None")
    evidence: List[Evidence] = Field(default_factory=list, description="Grounding evidence items")
    artifacts: List[Artifact] = Field(default_factory=list, description="Generated file artifacts")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model metadata (name, version, architecture)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Inference parameters used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible result metadata")
    execution_trace: List[ExecutionTraceEntry] = Field(default_factory=list, description="Component operational trace entries")


class ToolMetadata(BaseModel):
    """Metadata describing a registered specialist tool."""
    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human-readable capability description")
    version: str = Field(default="1.0.0", description="Semantic version string")
    supported_tasks: List[TaskType] = Field(description="Tasks supported by this tool")
    required_modalities: List[ImageModality] = Field(default_factory=list, description="Modalities expected by this tool")
    min_images: int = Field(default=1, ge=1, description="Minimum images required")
    max_images: int = Field(default=1, ge=1, description="Maximum images permitted")
    author_or_division: str = Field(default="Division 1", description="Owning division / developer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible tool attributes")


class TaskPlanStep(BaseModel):
    """Single operational step within a structured TaskPlan."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique step ID")
    step_index: int = Field(ge=0, description="0-indexed sequence position")
    task: TaskType = Field(description="Task to execute in this step")
    tool_name: str = Field(description="Name of specialist tool to invoke")
    purpose: str = Field(default="", description="Operational purpose of this workflow step")
    status: str = Field(default="planned", description="Step status: planned, running, completed, failed, skipped")
    input_references: List[str] = Field(default_factory=list, description="IDs of image inputs utilized")
    dependencies: List[str] = Field(default_factory=list, description="Step IDs that must complete prior to this step")
    pass_context_from_previous: bool = Field(default=True, description="Whether prior step outputs are passed to context")


class TaskPlan(BaseModel):
    """Canonical structured workflow plan for agent execution and observability."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique plan ID")
    goal: str = Field(description="Operational objective of this workflow plan")
    steps: List[TaskPlanStep] = Field(min_length=1, description="Ordered sequence of execution steps")
    is_multi_step: bool = Field(default=False, description="True if plan chains multiple specialist invocations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary plan metadata")


class AgentDecision(BaseModel):
    """Compact summary of the agent's routing decision and operational explanation."""
    task: TaskType = Field(description="Resolved remote-sensing task")
    task_display_name: str = Field(description="Human-readable task title")
    image_count: int = Field(ge=1, description="Number of validated input images")
    detected_modalities: List[ImageModality] = Field(description="Observed image modalities")
    selected_specialist: str = Field(description="Primary tool selected from registry")
    workflow_summary: str = Field(description="Summary of planned execution flow")
    confidence: Optional[float] = Field(default=None, description="Confidence in intent resolution")
    why_this_tool: str = Field(description="Operational rationale for specialist selection (no private CoT)")


class SatQueryErrorDetail(BaseModel):
    """Machine-readable and safe structured error detail."""
    error_code: str = Field(description="Centralized error taxonomy code")
    message: str = Field(description="Human-readable, safe error message")
    field: Optional[str] = Field(default=None, description="Field name causing validation error")
    request_id: Optional[str] = Field(default=None, description="Correlation request ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="Safe context metadata")


class QueryRequest(BaseModel):
    """Client API request payload."""
    query: str = Field(min_length=1, description="User's natural language query")
    images: List[ImageInput] = Field(min_length=1, description="List of image inputs")
    task_hint: Optional[TaskType] = Field(default=None, description="Optional manual override for task type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optional runtime parameters")


class QueryResponse(BaseModel):
    """Client API response payload."""
    request_id: str = Field(description="Correlation request identifier")
    query: str = Field(description="Original user query")
    resolved_task: TaskType = Field(description="Task resolved by agent")
    status: ToolStatus = Field(description="Overall execution status")
    answer: str = Field(description="Synthesized evidence-grounded response")
    confidence: Optional[float] = Field(default=None, description="Synthesized confidence if meaningful")
    evidence: List[Evidence] = Field(default_factory=list, description="Aggregated evidence list")
    artifacts: List[Artifact] = Field(default_factory=list, description="Aggregated artifacts list")
    execution_trace: List[ExecutionTraceEntry] = Field(default_factory=list, description="Operational audit trail")
    task_intent: Optional[TaskIntent] = Field(default=None, description="Resolved structured intent")
    task_plan: Optional[TaskPlan] = Field(default=None, description="Canonical workflow execution plan")
    agent_decision: Optional[AgentDecision] = Field(default=None, description="Agent decision summary card")
    selected_tools: List[str] = Field(default_factory=list, description="List of specialist tool names invoked")
    errors: List[SatQueryErrorDetail] = Field(default_factory=list, description="Errors encountered if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
