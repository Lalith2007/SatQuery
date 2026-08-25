"""Unit tests for centralized error taxonomy and exceptions."""

from core.errors import (
    ErrorCode,
    IncompatiblePairError,
    InferenceError,
    InputValidationError,
    InvalidImageCountError,
    SatQueryException,
    ToolNotFoundError,
    ToolTimeoutError,
    UnsupportedFormatError,
)
from core.schemas import SatQueryErrorDetail


def test_error_code_enum():
    assert ErrorCode.INVALID_INPUT == "INVALID_INPUT"
    assert ErrorCode.UNSUPPORTED_FORMAT == "UNSUPPORTED_FORMAT"
    assert ErrorCode.INCOMPATIBLE_PAIR == "INCOMPATIBLE_PAIR"
    assert ErrorCode.TOOL_NOT_FOUND == "TOOL_NOT_FOUND"


def test_exception_conversion_to_error_detail():
    exc = UnsupportedFormatError(
        message="Format .webp is not supported.",
        details={"attempted_format": ".webp"},
    )
    detail = exc.to_error_detail(request_id_override="req-999")
    assert isinstance(detail, SatQueryErrorDetail)
    assert detail.error_code == ErrorCode.UNSUPPORTED_FORMAT.value
    assert detail.message == "Format .webp is not supported."
    assert detail.request_id == "req-999"
    assert detail.details["attempted_format"] == ".webp"


def test_exception_status_codes():
    assert InputValidationError("Validation failed").status_code == 422
    assert ToolNotFoundError("Not found").status_code == 404
    assert ToolTimeoutError("Timeout").status_code == 504
    assert InferenceError("Inference failed").status_code == 500
