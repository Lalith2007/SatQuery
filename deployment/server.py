"""ZeroGPU-compatible Production Gradio Server for SatQuery AI.

Serves:
1. Gradio Server architecture (gradio.Server) for Hugging Face Spaces ZeroGPU
2. Complete REST API (query, upload, tasks, tools, artifacts, reports, benchmarks)
3. Interactive Modern React Frontend (SPA routing from frontend/dist)
4. ZeroGPU lifecycle guarded execution via deployment.runtime and @app.api
5. Authoritative /health and /version endpoints
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import torch

from gradio import Server
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.errors import SatQueryException
from core.logging import get_logger, setup_logging
from deployment.config import deploy_settings
from deployment.runtime import runtime
from registry.registry import default_registry
from app.routes import router as api_router
from app.ui import DEMO_HTML

PROJECT_ROOT = Path(__file__).resolve().parent.parent

setup_logging(level=settings.log_level)
logger = get_logger("deployment_server")


_specialists_initialized = False


def initialize_deployment_specialists(force: bool = False) -> None:
    """Initialize real specialists and register into default_registry."""
    global _specialists_initialized
    if _specialists_initialized and not force:
        return

    logger.info("=" * 75)
    logger.info(f"STARTING SATQUERY AI GRADIO HOSTING SERVER [{deploy_settings.deployment_identifier}]")
    logger.info(f"Target Platform: {deploy_settings.environment} | Port: {deploy_settings.port}")
    logger.info("=" * 75)

    settings.ensure_storage_dirs()

    # 1. Register Division 2 (Single-Image Qwen Specialist)
    try:
        from specialists.single_image.specialist import SingleImageRSSpecialistTool
        qwen_dir = deploy_settings.qwen_checkpoint_dir
        qwen_target = str(qwen_dir) if qwen_dir.exists() else "Qwen/Qwen2.5-VL-3B-Instruct"
        div2_tool = SingleImageRSSpecialistTool(base_model_id=qwen_target)
        default_registry.register(div2_tool, overwrite=True)
        logger.info(f"[Registered] Division 2 Specialist: '{div2_tool.name}' (checkpoint: {qwen_target})")
    except Exception as e:
        logger.error(f"Could not initialize Division 2 Specialist: {e}")

    # 2. Register Division 3 (Temporal Change TinyCD Specialist)
    try:
        from specialists.temporal_change import register_temporal_change_specialist
        div3_tool = register_temporal_change_specialist(default_registry)
        if hasattr(div3_tool, "_change_model") and hasattr(div3_tool._change_model, "initialize"):
            div3_tool._change_model.initialize()
        ckpt_p = getattr(div3_tool._config, "model_checkpoint_path", "ChangeDetector-TinyCD.pth")
        logger.info(f"[Registered] Division 3 Specialist: '{div3_tool.name}' (checkpoint: {ckpt_p})")
    except Exception as e:
        logger.error(f"Could not initialize Division 3 Specialist: {e}")

    # 3. Register Division 4 (Optical-SAR CMAF Specialist)
    try:
        from specialists.optical_sar.service import OpticalSarSpecialist
        div4_tool = OpticalSarSpecialist()
        default_registry.register(div4_tool, overwrite=True)
        logger.info(f"[Registered] Division 4 Specialist: '{div4_tool.name}' (version: {div4_tool.version})")
    except Exception as e:
        logger.error(f"Could not initialize Division 4 Specialist: {e}")

    # 4. Initialize demo assets
    try:
        from app.demo_assets import ensure_demo_assets
        ensure_demo_assets(PROJECT_ROOT / "demo_assets")
        logger.info("Demo satellite rasters initialized successfully.")
    except Exception as e:
        logger.warning(f"Could not ensure demo rasters: {e}")

    _specialists_initialized = True
    logger.info("All specialist registrations complete.")


@asynccontextmanager
async def deployment_lifespan(app: Any):
    """Lifespan manager initializing real specialists and verifying checkpoints."""
    initialize_deployment_specialists()
    yield
    logger.info("Shutting down SatQuery AI hosting server...")


# Automatically ensure specialists are initialized when server module is loaded
initialize_deployment_specialists()


# Create the application as a Gradio Server instance
app = Server(
    title="SatQuery AI — ZeroGPU Hosted API",
    description="Production vision-language assistant for remote-sensing imagery.",
    version=settings.app_version,
    lifespan=deployment_lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gradio Server Queued API Endpoint for GPU-dependent Vision-Language Inference
@app.api(
    name="predict",
    description="SatQuery AI ZeroGPU Queued Vision-Language Inference",
    concurrency_limit=1,
)
@runtime.gpu_decorator(duration=60)
def gradio_predict(
    query: str,
    task_hint: str = "single_image_vqa",
    image_paths_json: str = "[]",
) -> dict:
    """Execute queued GPU vision-language inference under ZeroGPU quota guard."""
    from core.schemas import QueryRequest, ImageInput, ImageFormat, TaskType
    from agent.controller import AgentController

    images: List[ImageInput] = []
    if image_paths_json:
        try:
            if isinstance(image_paths_json, str) and image_paths_json.strip().startswith("["):
                parsed = json.loads(image_paths_json)
            else:
                parsed = [p.strip() for p in str(image_paths_json).split(",") if p.strip()]

            for idx, p in enumerate(parsed):
                path_str = str(p).strip()
                if path_str:
                    ext = Path(path_str).suffix.lower()
                    fmt = ImageFormat.PNG
                    if ext in (".tif", ".tiff"):
                        fmt = ImageFormat.TIFF
                    elif ext in (".jpg", ".jpeg"):
                        fmt = ImageFormat.JPEG
                    images.append(ImageInput(
                        image_id=f"input_{idx}",
                        path_or_uri=path_str,
                        format=fmt,
                    ))
        except Exception as err:
            logger.warning(f"Could not parse image_paths_json '{image_paths_json}': {err}")

    # Resolve task type
    resolved_task = TaskType.SINGLE_IMAGE_VQA
    for t in TaskType:
        if t.value == task_hint:
            resolved_task = t
            break

    q_req = QueryRequest(
        query=query,
        images=images,
        task_hint=resolved_task,
    )

    controller = AgentController()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            res = loop.run_until_complete(controller.process_query(q_req))
        else:
            res = loop.run_until_complete(controller.process_query(q_req))
    except RuntimeError:
        res = asyncio.run(controller.process_query(q_req))

    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res.dict()


# Include core REST API routes
app.include_router(api_router)


# Mount static assets from frontend/dist if built
dist_dir = deploy_settings.frontend_dist_dir
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    logger.info(f"Mounted React static assets from: {assets_dir}")

# Mount demo assets and artifacts storage if present
demo_assets_dir = PROJECT_ROOT / "demo_assets"
if demo_assets_dir.exists():
    app.mount("/demo_assets", StaticFiles(directory=str(demo_assets_dir)), name="demo_assets")

artifacts_dir = PROJECT_ROOT / "artifacts_storage"
if artifacts_dir.exists():
    app.mount("/artifacts_storage", StaticFiles(directory=str(artifacts_dir)), name="artifacts_storage")


@app.get("/", summary="SatQuery React Frontend Entrypoint")
async def root_frontend():
    """Serve the compiled React single-page application, falling back to embedded UI if unbuilt."""
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(path=str(index_file), media_type="text/html")
    return HTMLResponse(content=DEMO_HTML)


from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def spa_404_handler(request: Request, exc: StarletteHTTPException):
    """Fallback handler returning React index.html for client-side routing on unmapped GET routes."""
    if exc.status_code == 404 and request.method == "GET":
        path = request.url.path
        if not path.startswith(("/api/", "/gradio_api/", "/assets/", "/demo_assets/", "/artifacts_storage/", "/queue/", "/health", "/version")):
            index_file = dist_dir / "index.html"
            if index_file.exists():
                return FileResponse(path=str(index_file), media_type="text/html")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(SatQueryException)
async def satquery_exception_handler(request: Request, exc: SatQueryException):
    return JSONResponse(
        status_code=503 if exc.error_code.value == "SERVICE_UNAVAILABLE" else 400,
        content=exc.to_dict() if hasattr(exc, "to_dict") else {"error": str(exc)},
    )


def launch_server_if_needed(prevent_thread_lock: bool = True):
    """Safely configure and launch the Gradio Server queue and API routes."""
    if not getattr(app, "_is_launched", False):
        try:
            app.launch(prevent_thread_lock=prevent_thread_lock, quiet=True)
            setattr(app, "_is_launched", True)
        except Exception as e:
            logger.warning(f"Gradio launch notice: {e}")


if __name__ == "__main__":
    app.launch(
        server_name=deploy_settings.host,
        server_port=deploy_settings.port,
        prevent_thread_lock=False,
    )
