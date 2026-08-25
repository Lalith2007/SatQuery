"""Configurable mock tool for testing failure boundaries, errors, and timeouts."""

from __future__ import annotations

import asyncio
from typing import Set

from core.errors import InferenceError, ModelLoadError, ToolTimeoutError
from core.interfaces import BaseSpecialistTool
from core.schemas import (
    ExecutionStage,
    ExecutionTraceEntry,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class MockFailingTool(BaseSpecialistTool):
    """Tool configured to fail with specific errors or timeouts for testing error boundaries."""

    def __init__(
        self,
        name: str = "failing_mock_tool",
        failure_mode: str = "inference_error",  # "inference_error", "model_load_error", "timeout", "partial_success"
        delay_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            name=name,
            description="Mock tool designed to test failure recovery and error taxonomy.",
            supported_tasks={TaskType.SINGLE_IMAGE_VQA, TaskType.CHANGE_ANALYSIS},
            version="1.0.0",
            metadata=ToolMetadata(
                name=name,
                description="Mock tool designed to test failure recovery and error taxonomy.",
                supported_tasks=[TaskType.SINGLE_IMAGE_VQA, TaskType.CHANGE_ANALYSIS],
                required_modalities=[ImageModality.OPTICAL, ImageModality.SAR],
                min_images=1,
                max_images=2,
                author_or_division="Testing Infrastructure",
            ),
        )
        self.failure_mode = failure_mode
        self.delay_seconds = delay_seconds

    async def execute(self, request: ToolRequest) -> ToolResult:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.failure_mode == "inference_error":
            raise InferenceError(f"Simulated unrecoverable inference error in tool '{self.name}'.")
        elif self.failure_mode == "model_load_error":
            raise ModelLoadError(f"Simulated checkpoint missing in tool '{self.name}'.")
        elif self.failure_mode == "partial_success":
            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.PARTIAL_SUCCESS,
                answer="Inference completed with degraded confidence due to high cloud shadow occlusions.",
                confidence=0.45,
                evidence=[],
                artifacts=[],
                model_info={"name": "FailingMock-Degraded"},
                execution_trace=[
                    ExecutionTraceEntry(
                        stage=ExecutionStage.INFERENCE_EXECUTED,
                        component=self.name,
                        status="PARTIAL_SUCCESS",
                        details={"warning": "degraded_signal"},
                    )
                ],
            )
        elif self.failure_mode == "timeout":
            raise ToolTimeoutError(f"Tool '{self.name}' timed out during execution.")

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.FAILED,
            answer="Tool execution failed.",
            confidence=0.0,
        )

    def health_check(self) -> bool:
        if self.failure_mode == "model_load_error":
            return False
        return True
