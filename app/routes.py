"""API Route handlers for SatQuery AI.

Provides versioned REST endpoints for query submission, raster upload,
tool inspection, task discovery, artifact access, interactive presentation UI,
multi-format report generation, and modular benchmark evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from core.config import settings
from core.errors import ErrorCode, SatQueryException
from core.logging import get_logger
from core.schemas import (
    Artifact,
    Evidence,
    ImageFormat,
    ImageInput,
    ImageModality,
    QueryRequest,
    QueryResponse,
    SatQueryErrorDetail,
    TaskType,
    ToolMetadata,
    ToolStatus,
)
from registry.registry import ToolRegistry, default_registry
from specialists.mock import (
    AlternateMockSingleImageVQATool,
    MockSingleImageVQATool,
)
from validation.validator import ALLOWED_EXTENSIONS, RasterInspector
from agent.controller import AgentController
from app.ui import DEMO_HTML
from presentation.confidence import ConfidencePresenter
from presentation.evidence_renderer import ArtifactRegistry, EvidenceRenderer
from presentation.trace_presenter import TracePresenter
from reports.generator import ReportGenerator
from evaluation.runner import BenchmarkRunner

logger = get_logger("api_routes")
router = APIRouter()

# Controller instance backed by default registry
controller = AgentController(registry=default_registry)


class ToolSwapRequest(BaseModel):
    """Payload for demo-only tool swapping demonstration."""
    target_tool: str = Field(default="single_image_vqa_mock", description="Tool name to swap")
    use_alternate: bool = Field(default=True, description="True to swap in alternate mock, False to restore standard mock")


class ReportGenerationRequest(BaseModel):
    """Payload for report generation."""
    response: QueryResponse = Field(description="Query response object to generate report from")
    format: str = Field(default="html", description="Report format: 'html', 'markdown', or 'json'")


class BenchmarkRunRequest(BaseModel):
    """Payload for running benchmark evaluation."""
    benchmark: str = Field(description="Target benchmark: 'vrsbench', 'rsvqa', 'cdvqa', or 'isro_sac'")
    predictions: List[Dict[str, Any]] = Field(description="Model prediction items")
    ground_truths: List[Dict[str, Any]] = Field(description="Ground truth reference items")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional evaluation configuration")


class EvidenceRenderRequest(BaseModel):
    """Payload for manual evidence rendering."""
    evidence_list: List[Evidence] = Field(description="List of evidence items to render")
    image_paths: List[str] = Field(description="Paths to source raster images")
    task_hint: Optional[str] = Field(default=None, description="Task context hint")


@router.get("/", response_class=HTMLResponse, summary="SatQuery Interactive Presentation UI")
@router.get("/demo", response_class=HTMLResponse, summary="SatQuery Interactive Presentation UI")
async def get_demo_dashboard():
    """Serves the rich, interactive agentic orchestration and evidence dashboard."""
    return HTMLResponse(content=DEMO_HTML)


@router.get("/health", summary="System Health & Status")
async def get_health():
    """Returns application health, version, registered tools, and status."""
    tool_health = default_registry.health_check_all()
    all_healthy = all(tool_health.values()) if tool_health else True

    return {
        "status": "healthy" if all_healthy else "degraded",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "registered_tools_count": len(default_registry.list_tools()),
        "tools_health": tool_health,
    }


@router.get("/api/v1/tasks", summary="List Supported Remote Sensing Tasks")
async def list_tasks():
    """Returns the controlled vocabulary of supported vision-language tasks."""
    return {
        "tasks": [
            {
                "task": TaskType.SINGLE_IMAGE_VQA.value,
                "description": "Visual Question Answering on a single optical/SAR remote sensing image.",
                "required_images": 1,
                "supported_modalities": ["optical", "multispectral", "sar"],
            },
            {
                "task": TaskType.SINGLE_IMAGE_CAPTION.value,
                "description": "Comprehensive scene description and automated caption generation.",
                "required_images": 1,
                "supported_modalities": ["optical", "multispectral", "sar"],
            },
            {
                "task": TaskType.SINGLE_IMAGE_GROUNDING.value,
                "description": "Spatial feature localization and bounding box extraction.",
                "required_images": 1,
                "supported_modalities": ["optical", "multispectral", "sar"],
            },
            {
                "task": TaskType.CHANGE_ANALYSIS.value,
                "description": "Bi-temporal change detection and difference map generation.",
                "required_images": 2,
                "supported_modalities": ["optical", "multispectral", "sar"],
            },
            {
                "task": TaskType.CHANGE_VQA.value,
                "description": "Natural language question answering regarding bi-temporal changes.",
                "required_images": 2,
                "supported_modalities": ["optical", "multispectral", "sar"],
            },
            {
                "task": TaskType.OPTICAL_SAR_ANALYSIS.value,
                "description": "Joint cross-modal analysis and fusion of co-registered Optical and SAR imagery.",
                "required_images": 2,
                "supported_modalities": ["optical", "sar"],
            },
        ]
    }


@router.get("/api/v1/tools", summary="List Registered Specialist Tools")
@router.get("/api/v1/registry", summary="List Registered Specialist Tools (Alias)")
async def list_registered_tools() -> List[ToolMetadata]:
    """Returns metadata for all specialist tools currently registered in the system."""
    return default_registry.list_tools()


@router.post("/api/v1/tools/swap", summary="Swap Specialist Tool Implementation (Demo/Dev Only)")
async def swap_specialist_tool(req: ToolSwapRequest):
    """Demonstrates runtime specialist tool swapping without agent modification (Development/Demo only)."""
    if not settings.enable_dev_tool_swap:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tool swapping endpoint is disabled in this environment configuration.",
        )

    if req.use_alternate:
        alternate_tool = AlternateMockSingleImageVQATool()
        old_tool = default_registry.swap_tool("single_image_vqa_mock", alternate_tool)
        return {
            "status": "swapped",
            "message": f"Successfully swapped specialist: 'single_image_vqa_mock' -> '{alternate_tool.name}' (version {alternate_tool.version}).",
            "active_tool": alternate_tool.name,
            "version": alternate_tool.version,
        }
    else:
        standard_tool = MockSingleImageVQATool()
        old_tool = default_registry.swap_tool("alternate_single_image_vqa_mock", standard_tool)
        return {
            "status": "restored",
            "message": f"Successfully restored standard specialist: 'alternate_single_image_vqa_mock' -> '{standard_tool.name}' (version {standard_tool.version}).",
            "active_tool": standard_tool.name,
            "version": standard_tool.version,
        }


@router.post("/api/v1/query", response_model=QueryResponse, summary="Submit Structured Vision-Language Query")
async def submit_query(request: QueryRequest) -> QueryResponse:
    """Submit a natural-language query with existing canonical raster image references."""
    req_id = str(uuid.uuid4())
    try:
        response = await controller.process_query(request, request_id=req_id)

        # Register any artifacts returned directly by specialists
        for art in response.artifacts:
            ArtifactRegistry.register(art.artifact_id, art.uri_or_path, name=art.name)

        # Division 5 Enhancement: Render spatial evidence into actual visual artifacts if present
        if response.evidence and request.images:
            try:
                rendered_results = EvidenceRenderer.render_all_evidence(
                    evidence_list=response.evidence,
                    image_inputs=request.images,
                    task_hint=response.resolved_task.value,
                )
                for r in rendered_results:
                    if r.artifact:
                        ArtifactRegistry.register(r.artifact.artifact_id, r.artifact.uri_or_path, name=r.artifact.name)
                        if not any(a.artifact_id == r.artifact.artifact_id for a in response.artifacts):
                            response.artifacts.append(r.artifact)
            except Exception as ev_err:
                logger.warning(f"Evidence rendering notice: {ev_err}")

        return response
    except Exception as err:
        logger.exception(f"Unhandled error in POST /api/v1/query: {err}")
        return QueryResponse(
            request_id=req_id,
            query=request.query,
            resolved_task=request.task_hint or TaskType.SINGLE_IMAGE_VQA,
            status=ToolStatus.FAILED,
            answer="Internal error while processing query.",
            errors=[
                SatQueryErrorDetail(
                    error_code="INTERNAL_SERVER_ERROR",
                    message=str(err),
                    request_id=req_id,
                )
            ],
        )


@router.post("/api/v1/query/multipart", response_model=QueryResponse, summary="Upload Images and Submit Query")
async def submit_query_multipart(
    query: str = Form(..., description="User natural language query"),
    task_hint: Optional[str] = Form(None, description="Optional TaskType hint override"),
    modalities: Optional[str] = Form(None, description="Optional JSON list of modalities e.g. ['optical', 'sar']"),
    files: List[UploadFile] = File(..., description="Uploaded GeoTIFF/TIFF or PNG/JPEG raster files"),
) -> QueryResponse:
    """Upload one or more raster images and submit a natural-language query in a single multipart request."""
    req_id = str(uuid.uuid4())
    upload_dir = settings.artifact_storage_path / "uploads" / req_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    parsed_modalities = []
    if modalities:
        try:
            parsed_modalities = json.loads(modalities)
        except Exception:
            pass

    resolved_task_hint = None
    if task_hint:
        try:
            resolved_task_hint = TaskType(task_hint.lower())
        except ValueError:
            pass

    image_inputs: List[ImageInput] = []
    try:
        for idx, file in enumerate(files):
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Uploaded file '{file.filename}' has unsupported format '{file_ext}'. Primary formats: .tif, .tiff; benchmark formats: .png, .jpg, .jpeg",
                )

            dest_path = upload_dir / f"{idx}_{file.filename}"
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Inspect saved file
            w, h, channels, dtype, geo = RasterInspector.inspect_file(dest_path)

            # Determine modality
            assigned_modality = ImageModality.UNKNOWN
            if idx < len(parsed_modalities):
                try:
                    assigned_modality = ImageModality(parsed_modalities[idx].lower())
                except ValueError:
                    pass

            image_input = ImageInput(
                path_or_uri=str(dest_path),
                format=ALLOWED_EXTENSIONS[file_ext],
                modality=assigned_modality,
                width=w,
                height=h,
                channel_count=channels,
                dtype=dtype,
                geospatial=geo,
                metadata={"original_filename": file.filename},
            )
            image_inputs.append(image_input)

        query_req = QueryRequest(
            query=query,
            images=image_inputs,
            task_hint=resolved_task_hint,
        )

        response = await controller.process_query(query_req, request_id=req_id)

        # Register any artifacts returned directly by specialists
        for art in response.artifacts:
            ArtifactRegistry.register(art.artifact_id, art.uri_or_path, name=art.name)

        # Division 5 Enhancement: Render spatial evidence into actual visual artifacts
        if response.evidence:
            try:
                rendered_results = EvidenceRenderer.render_all_evidence(
                    evidence_list=response.evidence,
                    image_inputs=image_inputs,
                    task_hint=response.resolved_task.value,
                )
                for r in rendered_results:
                    if r.artifact:
                        ArtifactRegistry.register(r.artifact.artifact_id, r.artifact.uri_or_path, name=r.artifact.name)
                        if not any(a.artifact_id == r.artifact.artifact_id for a in response.artifacts):
                            response.artifacts.append(r.artifact)
            except Exception as ev_err:
                logger.warning(f"Evidence rendering notice: {ev_err}")

        return response

    except HTTPException:
        raise
    except SatQueryException as sq_err:
        return QueryResponse(
            request_id=req_id,
            query=query,
            resolved_task=resolved_task_hint or TaskType.SINGLE_IMAGE_VQA,
            status=ToolStatus.FAILED,
            answer=f"Error: {sq_err.message}",
            errors=[sq_err.to_error_detail(request_id_override=req_id)],
        )
    except Exception as err:
        logger.exception(f"Unhandled error in multipart submission: {err}")
        return QueryResponse(
            request_id=req_id,
            query=query,
            resolved_task=resolved_task_hint or TaskType.SINGLE_IMAGE_VQA,
            status=ToolStatus.FAILED,
            answer="Internal error while processing uploaded images.",
            errors=[
                SatQueryErrorDetail(
                    error_code="INTERNAL_SERVER_ERROR",
                    message=str(err),
                    request_id=req_id,
                )
            ],
        )


@router.get("/api/v1/artifacts/{artifact_id}", summary="Retrieve Generated Artifact")
async def get_artifact(artifact_id: str):
    """Retrieve an artifact file (change map, segmented mask, annotated visual, report) by ID or filename safely."""
    # 1. Path safety: reject path traversal sequences
    if ".." in artifact_id or "\\" in artifact_id or "/" in artifact_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact identifier.",
        )

    storage_root = settings.artifact_storage_path.resolve()
    demo_root = Path("demo_assets").resolve()
    allowed_roots = [storage_root, demo_root]

    target_path: Optional[Path] = None

    # Step A: Check in-memory ArtifactRegistry
    registered_path = ArtifactRegistry.get_path(artifact_id)
    if registered_path:
        p = Path(registered_path).resolve()
        if p.is_file():
            target_path = p

    # Step B: Direct search in storage_root by full UUID or substring
    if not target_path and storage_root.exists():
        matched = list(storage_root.rglob(f"*{artifact_id}*"))
        if matched and matched[0].is_file():
            target_path = matched[0]

    # Step C: Short UUID prefix match (for backward compatibility with 8-character prefixes)
    if not target_path and storage_root.exists() and len(artifact_id) >= 8:
        short_id = artifact_id[:8]
        matched = list(storage_root.rglob(f"*{short_id}*"))
        if matched and matched[0].is_file():
            target_path = matched[0]

    # Step D: Exact filename match in storage_root
    if not target_path and storage_root.exists():
        matched = list(storage_root.rglob(artifact_id))
        if matched and matched[0].is_file():
            target_path = matched[0]

    # Step E: Search in demo_assets directory
    if not target_path and demo_root.exists():
        matched = list(demo_root.rglob(f"*{artifact_id}*"))
        if not matched:
            matched = list(demo_root.rglob(artifact_id))
        if matched and matched[0].is_file():
            target_path = matched[0]

    # If still not found, return 404
    if not target_path or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact with identifier '{artifact_id}' not found.",
        )

    # Security check: Ensure target_path is strictly within allowed roots
    resolved_path = target_path.resolve()
    is_safe = any(
        resolved_path == root or root in resolved_path.parents
        for root in allowed_roots
    )
    if not is_safe:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to requested artifact path is forbidden.",
        )

    # Determine accurate media_type
    ext = resolved_path.suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
        ".html": "text/html",
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    return FileResponse(path=str(resolved_path), filename=resolved_path.name, media_type=media_type)


# ---------------------------------------------------------------------------
# Division 5 Endpoints: Report Generation & Benchmark Evaluation
# ---------------------------------------------------------------------------

@router.post("/api/v1/reports/generate", summary="Generate Downloadable Intelligence Report")
async def generate_report(req: ReportGenerationRequest):
    """Generate and persist a downloadable intelligence report in HTML, Markdown, or JSON."""
    fmt = req.format.lower().strip()
    if fmt == "html":
        report = ReportGenerator.generate_html_report(req.response)
    elif fmt in {"md", "markdown"}:
        report = ReportGenerator.generate_markdown_report(req.response)
    elif fmt == "json":
        report = ReportGenerator.generate_json_report(req.response)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report format '{req.format}'. Supported: 'html', 'markdown', 'json'.",
        )

    return {
        "status": "success",
        "report_id": report.report_id,
        "format": report.format_type,
        "download_url": f"/api/v1/reports/{report.report_id}",
        "file_name": report.artifact.name,
        "artifact": report.artifact.model_dump(),
        "content_preview": report.content[:500] if len(report.content) > 500 else report.content,
    }


@router.get("/api/v1/reports/{report_id}", summary="Download Generated Report File")
async def download_report(report_id: str):
    """Download a generated report file directly by ID."""
    report_dir = ReportGenerator.get_report_storage_dir()
    matched = list(report_dir.glob(f"*{report_id}*"))

    if not matched or not matched[0].is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with identifier '{report_id}' not found.",
        )

    file_path = matched[0]
    media_type = "text/html" if file_path.suffix == ".html" else "application/json" if file_path.suffix == ".json" else "text/markdown"
    return FileResponse(path=str(file_path), filename=file_path.name, media_type=media_type)


@router.get("/api/v1/evaluation/benchmarks", summary="List Supported Evaluation Benchmarks")
async def list_evaluation_benchmarks():
    """List available benchmark evaluation suites and supported metrics."""
    return {
        "supported_benchmarks": [
            {
                "id": "vrsbench",
                "name": "VRSBench (VQA & Grounding)",
                "description": "Visual question answering and bounding-box spatial grounding on high-resolution optical imagery.",
                "metrics": ["vqa_accuracy", "vqa_token_f1", "grounding_miou", "grounding_p_at_05", "grounding_p_at_75"],
            },
            {
                "id": "rsvqa",
                "name": "RSVQA (LR & HR)",
                "description": "Remote-sensing VQA across low and high resolution datasets covering presence, count, and comparison questions.",
                "metrics": ["overall_accuracy", "presence_accuracy", "comparison_accuracy", "count_rmse"],
            },
            {
                "id": "cdvqa",
                "name": "CDVQA (Change Detection VQA)",
                "description": "Bi-temporal change detection VQA, change description quality, and changed feature localization.",
                "metrics": ["binary_change_accuracy", "change_description_bleu1", "change_description_bleu4", "change_description_rouge_l"],
            },
            {
                "id": "isro_sac",
                "name": "ISRO/SAC Generic Benchmark",
                "description": "Generic multi-sensor evaluation for Cartosat-2S optical and RISAT SAR test sets without hardcoded references.",
                "metrics": ["isro_sac_vqa_accuracy", "isro_sac_token_f1", "isro_sac_grounding_miou", "isro_sac_grounding_p50"],
            },
        ]
    }


@router.post("/api/v1/evaluation/run", summary="Execute Benchmark Evaluation Suite")
async def run_benchmark_evaluation(req: BenchmarkRunRequest):
    """Execute reproducible benchmark evaluation and return structured metrics and scoreboard."""
    try:
        result = BenchmarkRunner.run_evaluation(
            benchmark_name=req.benchmark,
            predictions=req.predictions,
            ground_truths=req.ground_truths,
            config=req.config,
        )
        return {
            "status": "success",
            "benchmark": result.benchmark_name,
            "samples_evaluated": result.total_samples,
            "aggregate_normalized_score": result.aggregate_normalized_score,
            "aggregate_raw_score": result.aggregate_raw_score,
            "metrics": {k: v.model_dump() for k, v in result.metrics.items()},
            "per_category_scores": result.per_category_scores,
            "scoreboard_markdown": BenchmarkRunner.format_scoreboard_markdown(result),
        }
    except Exception as e:
        logger.exception(f"Benchmark evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation failed on benchmark '{req.benchmark}': {str(e)}",
        )
