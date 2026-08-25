"""Structured error taxonomy for Division 3: Bi-Temporal Change Intelligence.

Uses the existing core.errors hierarchy. Does not create competing error classes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.errors import (
    ErrorCode,
    IncompatiblePairError,
    InferenceError,
    InputValidationError,
    InvalidImageCountError,
    ModelLoadError,
)


# --- Validation errors (extend existing taxonomy) ---

class ImageReadError(InputValidationError):
    """Raised when one or both images in a temporal pair cannot be read."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            field="images",
            error_code=ErrorCode.INVALID_INPUT,
            details=details,
        )


class IncompatibleDimensionsError(IncompatiblePairError):
    """Raised when T0/T1 dimensions are incompatible and cannot be safely aligned."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


class GeospatialMismatchError(IncompatiblePairError):
    """Raised when T0/T1 geospatial metadata indicates spatial incompatibility."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


class TemporalValidationError(IncompatiblePairError):
    """Raised when T0/T1 timestamps indicate invalid temporal relationship."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


# --- Inference errors ---

class ChangeModelError(InferenceError):
    """Raised when the change detection model fails during inference."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


class SemanticReasoningError(InferenceError):
    """Raised when the semantic reasoning stage fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


class ChangeModelLoadError(ModelLoadError):
    """Raised when the change detection model cannot be loaded."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)


class ArtifactGenerationError(InferenceError):
    """Raised when evidence artifact generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, details=details)
