"""ZeroGPU-compatible Production FastAPI Server for SatQuery AI.

Serves:
1. Complete REST API (query, upload, tasks, tools, artifacts, reports, benchmarks)
2. Interactive Modern React Frontend (SPA routing from frontend/dist)
3. ZeroGPU lifecycle guarded execution via deployment.runtime
4. Authoritative /health and /version endpoints
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, Dict
import torch

from fastapi import FastAPI, Request
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


@asynccontextmanager
async def deployment_lifespan(app: FastAPI):
    """Lifespan manager initializing real specialists and verifying checkpoints."""
    logger.info("=" * 75)
    logger.info(f"STARTING SATQUERY AI HOSTING SERVER [{deploy_settings.deployment_identifier}]")
    logger.info(f"Target Platform: {deploy_settings.environment} | Port: {deploy_settings.port}")
    logger.info("=" * 75)

    settings.ensure_storage_dirs()

    # 1. Register Division 2 (Single-Image Qwen Specialist)
    try:
        from specialists.single_image.specialist import SingleImageRSSpecialistTool
        # Point to verified merged_full checkpoint if available
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

    logger.info("All specialist registrations complete.")
    yield
    logger.info("Shutting down SatQuery AI hosting server...")


app = FastAPI(
    title="SatQuery AI — ZeroGPU Hosted API",
    description="Production vision-language assistant for remote-sensing imagery.",
    version=settings.app_version,
    lifespan=deployment_lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include core REST API routes
app.include_router(api_router)


# Mount static assets from frontend/dist if built
dist_dir = deploy_settings.frontend_dist_dir
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    logger.info(f"Mounted React static assets from: {assets_dir}")


@app.get("/", summary="SatQuery React Frontend Entrypoint")
async def root_frontend():
    """Serve the compiled React single-page application, falling back to embedded UI if unbuilt."""
    index_file = dist_dir / "index.html"
    if index_file.exists():
        return FileResponse(path=str(index_file), media_type="text/html")
    return HTMLResponse(content=DEMO_HTML)


@app.exception_handler(SatQueryException)
async def satquery_exception_handler(request: Request, exc: SatQueryException):
    return JSONResponse(
        status_code=503 if exc.error_code.value == "SERVICE_UNAVAILABLE" else 400,
        content=exc.to_dict() if hasattr(exc, "to_dict") else {"error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "deployment.server:app",
        host=deploy_settings.host,
        port=deploy_settings.port,
        reload=False,
    )
