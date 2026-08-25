"""Centralized error taxonomy and structured exception hierarchy for SatQuery AI.

Provides machine-readable error codes, safe human-readable messages,
and automatic conversion to structured API error models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from core.schemas import SatQueryErrorDetail


class ErrorCode(str, Enum):
    """Controlled vocabulary of system-wide error codes."""
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    INVALID_IMAGE_COUNT = "INVALID_IMAGE_COUNT"
    INCOMPATIBLE_PAIR = "INCOMPATIBLE_PAIR"
    MISSING_METADATA = "MISSING_METADATA"
    TASK_NOT_SUPPORTED = "TASK_NOT_SUPPORTED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    TIMEOUT = "TIMEOUT"
    OUTPUT_VALIDATION_ERROR = "OUTPUT_VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class SatQueryException(Exception):
    """Base exception for all SatQuery AI domain errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INVALID_INPUT,
        field: Optional[str] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.field = field
        self.request_id = request_id
        self.details = details or {}
        self.status_code = status_code

    def to_error_detail(self, request_id_override: Optional[str] = None) -> SatQueryErrorDetail:
        """Convert exception to safe, machine-readable API error detail."""
        return SatQueryErrorDetail(
            error_code=self.error_code.value,
            message=self.message,
            field=self.field,
            request_id=request_id_override or self.request_id,
            details=self.details,
        )


class InputValidationError(SatQueryException):
    """Raised when request payload or raster input fails validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.INVALID_INPUT,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code=error_code,
            field=field,
            details=details,
            status_code=422,
        )


class UnsupportedFormatError(InputValidationError):
    """Raised when an unapproved raster/image format is supplied."""

    def __init__(self, message: str, field: Optional[str] = "format", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field=field,
            error_code=ErrorCode.UNSUPPORTED_FORMAT,
            details=details,
        )


class UnsupportedModalityError(InputValidationError):
    """Raised when an unsupported or unrecognized sensor modality is provided."""

    def __init__(self, message: str, field: Optional[str] = "modality", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field=field,
            error_code=ErrorCode.UNSUPPORTED_MODALITY,
            details=details,
        )


class InvalidImageCountError(InputValidationError):
    """Raised when the number of images is incompatible with the resolved task."""

    def __init__(self, message: str, field: Optional[str] = "images", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field=field,
            error_code=ErrorCode.INVALID_IMAGE_COUNT,
            details=details,
        )


class IncompatiblePairError(InputValidationError):
    """Raised when a bi-temporal or cross-modal pair fails compatibility checks."""

    def __init__(self, message: str, field: Optional[str] = "images", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field=field,
            error_code=ErrorCode.INCOMPATIBLE_PAIR,
            details=details,
        )


class MissingMetadataError(InputValidationError):
    """Raised when a task strictly requires missing geospatial or temporal metadata."""

    def __init__(self, message: str, field: Optional[str] = "metadata", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field=field,
            error_code=ErrorCode.MISSING_METADATA,
            details=details,
        )


class TaskNotSupportedError(SatQueryException):
    """Raised when a query cannot be resolved to any supported task."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.TASK_NOT_SUPPORTED,
            details=details,
            status_code=400,
        )


class ToolNotFoundError(SatQueryException):
    """Raised when no registered specialist tool satisfies the requested task."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.TOOL_NOT_FOUND,
            details=details,
            status_code=404,
        )


class ModelLoadError(SatQueryException):
    """Raised when a specialist fails to load its underlying ML checkpoint."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_LOAD_ERROR,
            details=details,
            status_code=500,
        )


class InferenceError(SatQueryException):
    """Raised when a tool encounters an unrecoverable failure during inference."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.INFERENCE_ERROR,
            details=details,
            status_code=500,
        )


class ToolTimeoutError(SatQueryException):
    """Raised when tool execution exceeds the configured timeout budget."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.TIMEOUT,
            details=details,
            status_code=504,
        )


class OutputValidationError(SatQueryException):
    """Raised when a specialist tool returns a malformed result object."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.OUTPUT_VALIDATION_ERROR,
            details=details,
            status_code=502,
        )


class ConfigurationError(SatQueryException):
    """Raised when application configuration or environment settings are invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
            status_code=500,
        )
