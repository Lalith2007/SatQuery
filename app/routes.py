"""API Route handlers for SatQuery AI.

Provides versioned REST endpoints for query submission, raster upload,
tool inspection, task discovery, artifact access, and interactive presentation UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import List, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from core.config import settings
from core.errors import ErrorCode, SatQueryException
from core.logging import get_logger
from core.schemas import (
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

logger = get_logger("api_routes")
router = APIRouter()

# Controller instance backed by default registry
controller = AgentController(registry=default_registry)


class ToolSwapRequest(BaseModel):
    """Payload for demo-only tool swapping demonstration."""
    target_tool: str = Field(default="single_image_vqa_mock", description="Tool name to swap")
    use_alternate: bool = Field(default=True, description="True to swap in alternate mock, False to restore standard mock")


@router.get("/", response_class=HTMLResponse, summary="SatQuery Interactive Presentation UI")
@router.get("/demo", response_class=HTMLResponse, summary="SatQuery Interactive Presentation UI")
async def get_demo_dashboard():
    """Serves the rich, interactive agentic orchestration dashboard."""
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

    # Strictly allow swapping only between approved registered mock tools (no arbitrary class loading)
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
    """Retrieve an artifact file (change map, segmented mask, annotated visual) by ID."""
    storage_root = settings.artifact_storage_path
    
    # Search for matching artifact file
    matched = list(storage_root.rglob(f"*{artifact_id}*"))
    if not matched:
        matched = list(storage_root.rglob(artifact_id))

    if not matched or not matched[0].is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact with identifier '{artifact_id}' not found.",
        )

    file_path = matched[0]
    return FileResponse(path=str(file_path), filename=file_path.name)
