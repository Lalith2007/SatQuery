"""Minimal and stable specialist tool interface for SatQuery AI.

All specialist modules (Divisions 2, 3, 4) must inherit from BaseSpecialistTool.
The Agent interacts with specialists strictly through this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set

from pydantic import BaseModel, Field

from core.schemas import TaskType, ToolMetadata, ToolRequest, ToolResult


class ValidationResult(BaseModel):
    """Result of pre-execution tool request validation."""
    is_valid: bool = Field(description="True if the request meets tool requirements")
    errors: List[str] = Field(default_factory=list, description="Validation failure messages if any")


class BaseSpecialistTool(ABC):
    """Abstract Base Class for all remote-sensing specialist tools.
    
    A specialist wraps a vision-language model, change detector, or cross-modal
    engine and exposes this standardized, model-agnostic contract.
    """

    def __init__(
        self,
        name: str,
        description: str,
        supported_tasks: Set[TaskType],
        version: str = "1.0.0",
        metadata: ToolMetadata | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._supported_tasks = supported_tasks
        self._version = version
        self._metadata = metadata or ToolMetadata(
            name=name,
            description=description,
            version=version,
            supported_tasks=list(supported_tasks),
        )

    @property
    def name(self) -> str:
        """Unique identifier of the specialist tool."""
        return self._name

    @property
    def description(self) -> str:
        """Description of the tool capability."""
        return self._description

    @property
    def supported_tasks(self) -> Set[TaskType]:
        """Set of TaskTypes supported by this tool."""
        return self._supported_tasks

    @property
    def version(self) -> str:
        """Semantic version of the tool."""
        return self._version

    @property
    def metadata(self) -> ToolMetadata:
        """Detailed capability metadata for registry cataloging."""
        return self._metadata

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        """Validate whether the incoming request is compatible with this tool.
        
        Subclasses can override to add domain-specific parameter or modality checks.
        """
        if request.task not in self.supported_tasks:
            return ValidationResult(
                is_valid=False,
                errors=[f"Task '{request.task.value}' is not supported by tool '{self.name}'."],
            )
        return ValidationResult(is_valid=True)

    @abstractmethod
    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute inference on the canonical request and return a standardized ToolResult.
        
        Must be implemented by each specialist tool.
        """
        pass

    def health_check(self) -> bool:
        """Return True if the underlying model/weights are loaded and ready."""
        return True
