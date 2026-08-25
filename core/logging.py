"""Structured, context-aware logging infrastructure for SatQuery AI.

Supports request_id correlation across asynchronous execution stages
without exposing raw binary data, secrets, or private chain-of-thought.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from typing import Any, Dict, Optional

# Context variable for request correlation ID
_request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """Bind a request_id to the current async execution context."""
    _request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """Retrieve the current request_id from async context."""
    return _request_id_ctx.get()


class StructuredFormatter(logging.Formatter):
    """JSON-like structured log formatter for machine auditability."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = get_request_id() or getattr(record, "request_id", "N/A")
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "request_id": req_id,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            # Sanitize secrets or huge payloads
            safe_data = {
                k: v for k, v in record.extra_data.items()
                if "key" not in k.lower() and "secret" not in k.lower() and "token" not in k.lower()
            }
            log_entry["details"] = safe_data
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure system-wide logging."""
    root_logger = logging.getLogger("satquery")
    root_logger.setLevel(level.upper())
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())
    
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        standard_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] [%(filename)s:%(lineno)d] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(standard_formatter)
        
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger under the 'satquery' hierarchy."""
    if not name.startswith("satquery"):
        name = f"satquery.{name}"
    return logging.getLogger(name)
