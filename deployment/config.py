"""Deployment configuration for SatQuery AI Free GPU Hosting (ZeroGPU / Cloud).

Centralizes all hosting-specific configuration, environment variables,
ZeroGPU quota management, and model directory resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DeploymentConfig(BaseModel):
    """Configuration for SatQuery AI ZeroGPU deployment layer."""

    deployment_name: str = Field(default="SatQuery AI — ZeroGPU Production")
    deployment_identifier: str = Field(default="baseline-2026-09-18")
    environment: str = Field(
        default_factory=lambda: os.getenv("SATQUERY_ENV", os.getenv("ENVIRONMENT", "zerogpu" if os.getenv("SPACE_ID") else "development"))
    )
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "7860")))

    # ZeroGPU configuration
    zerogpu_duration_seconds: int = Field(
        default_factory=lambda: int(os.getenv("ZEROGPU_DURATION", "60"))
    )
    concurrency_limit: int = Field(
        default_factory=lambda: int(os.getenv("CONCURRENCY_LIMIT", "1"))
    )

    # Checkpoint paths
    qwen_checkpoint_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("QWEN_CHECKPOINT_DIR", str(PROJECT_ROOT / "merged_full"))
        )
    )
    tinycd_checkpoint_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "TINYCD_CHECKPOINT_PATH",
                str(PROJECT_ROOT / "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"),
            )
        )
    )
    cmaf_checkpoint_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "CMAF_CHECKPOINT_PATH",
                str(PROJECT_ROOT / "specialists/optical_sar/checkpoints/cmaf_landcover_best.pth"),
            )
        )
    )

    # Hugging Face Model Repository
    hf_model_repo_id: str = Field(default="Lalith47/SatQuery-Models")
    hf_model_revision: str = Field(default="stage1-baseline")

    # Static assets
    frontend_dist_dir: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "frontend/dist"
    )

    def verify_paths(self) -> Dict[str, bool]:
        """Verify existence of critical deployment assets."""
        return {
            "qwen_checkpoint": self.qwen_checkpoint_dir.exists(),
            "tinycd_checkpoint": self.tinycd_checkpoint_path.exists(),
            "cmaf_checkpoint": self.cmaf_checkpoint_path.exists(),
            "frontend_dist": self.frontend_dist_dir.exists(),
        }


deploy_settings = DeploymentConfig()
