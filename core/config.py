"""System-wide configuration management for SatQuery AI.

Supports environment variables, .env files, and safe defaults.
Never hard-codes sensitive keys, credentials, or machine paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SatQuerySettings(BaseSettings):
    """Central settings model for the SatQuery AI application."""
    model_config = SettingsConfigDict(
        env_prefix="SATQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="SatQuery AI", description="Application display name")
    app_version: str = Field(default="0.1.0", description="Application version")
    env: str = Field(default="development", description="Environment mode: development, testing, production")
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")

    # Specialist Configuration
    use_mock_specialists: bool = Field(
        default=True,
        description="Whether to register default high-fidelity mock specialists when real modules are absent",
    )
    tool_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum execution timeout per specialist tool in seconds",
    )

    # Artifact Storage
    artifact_storage_path: Path = Field(
        default=Path("artifacts_storage"),
        description="Local directory path to store generated masks, maps, and visual artifacts",
    )
    max_upload_size_mb: int = Field(
        default=100,
        description="Maximum allowed raster file upload size in megabytes",
    )

    # Security & Networking
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins for the API",
    )

    # Optional Language Model Integration for Intent Resolution
    llm_provider: Optional[str] = Field(default=None, description="Optional LLM provider (e.g. gemini, openai)")
    llm_api_key: Optional[str] = Field(default=None, description="Safe API key reference from env")
    llm_model: Optional[str] = Field(default=None, description="LLM model identifier")

    def ensure_storage_dirs(self) -> None:
        """Create necessary storage directories if they do not exist."""
        self.artifact_storage_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = SatQuerySettings()
