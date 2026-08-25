"""FastAPI Application Entry Point for SatQuery AI.

Division 1: Agent Core + Backend + Orchestration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.errors import SatQueryException
from core.logging import get_logger, setup_logging
from registry.registry import default_registry
from specialists.mock import register_default_mocks
from app.routes import router

# Initialize system logging
setup_logging(level=settings.log_level)
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version} in [{settings.env}] mode")
    settings.ensure_storage_dirs()

    # Pre-populate registry with mock specialists if enabled
    if settings.use_mock_specialists:
        logger.info("Registering default high-fidelity mock specialists for Divisions 2, 3, and 4...")
        register_default_mocks(default_registry)
        logger.info(f"Registered {len(default_registry.list_tools())} tools into default registry.")

    # Register real Division 2 specialist (SingleImageRSSpecialistTool)
    try:
        from specialists.single_image.specialist import SingleImageRSSpecialistTool
        div2_tool = SingleImageRSSpecialistTool()
        default_registry.register(div2_tool, overwrite=True)
        logger.info(f"Registered real Division 2 specialist: '{div2_tool.name}' (v{div2_tool.version})")
    except Exception as e:
        logger.warning(f"Could not register real Division 2 specialist: {e}")

    # Ensure demo assets exist on disk for judging presets
    try:
        from app.demo_assets import ensure_demo_assets
        ensure_demo_assets("demo_assets")
        logger.info("Initialized demo remote-sensing rasters for presentation presets.")
    except Exception as e:
        logger.warning(f"Could not generate demo rasters: {e}")

    yield

    logger.info(f"Shutting down {settings.app_name}...")


app = FastAPI(
    title="SatQuery AI API",
    description=(
        "Agentic vision-language assistant for remote-sensing imagery. "
        "Provides agent orchestration, natural-language task routing, "
        "evidence extraction, and specialist tool execution."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

# Configure CORS for Web/GUI (Division 5 presentation layer)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SatQueryException)
async def satquery_exception_handler(request: Request, exc: SatQueryException):
    """Handle custom SatQuery domain exceptions with structured error responses."""
    logger.warning(f"Handled SatQueryException: {exc.error_code.value} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.to_error_detail().model_dump()},
    )


# Include API routes
app.include_router(router)
