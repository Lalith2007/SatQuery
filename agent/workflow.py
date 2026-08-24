"""Workflow planning abstraction for SatQuery AI.

Supports both single-tool workflows and sequential multi-tool pipelines
(e.g., change detection -> region-specific visual question answering).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from core.interfaces import BaseSpecialistTool
from core.schemas import ImageInput, TaskIntent, TaskType


class WorkflowStep(BaseModel):
    """Single executable step in a workflow plan."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = Field(ge=0)
    task: TaskType = Field(description="Task to execute in this step")
    tool_name: str = Field(description="Name of specialist tool to invoke")
    description: str = Field(default="", description="Operational description of step")
    pass_context_from_previous: bool = Field(default=True, description="Whether to inject prior step outputs into context")


class WorkflowPlan(BaseModel):
    """Complete workflow execution plan."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: TaskIntent = Field(description="Resolved user intent")
    steps: List[WorkflowStep] = Field(min_length=1, description="Sequential steps to execute")
    is_multi_step: bool = Field(default=False, description="True if plan involves multiple specialist invocations")


class WorkflowPlanner:
    """Constructs single-step or sequential multi-step workflow plans."""

    @classmethod
    def create_plan(
        cls,
        intent: TaskIntent,
        selected_tool: BaseSpecialistTool,
        secondary_tool: Optional[BaseSpecialistTool] = None,
        is_composite_query: bool = False,
    ) -> WorkflowPlan:
        """Create a workflow plan based on intent and selected specialist tools."""
        # Single-step execution is the standard default
        if not is_composite_query or secondary_tool is None:
            step = WorkflowStep(
                step_index=0,
                task=intent.task,
                tool_name=selected_tool.name,
                description=f"Primary specialist inference via '{selected_tool.name}'",
            )
            return WorkflowPlan(
                intent=intent,
                steps=[step],
                is_multi_step=False,
            )

        # Multi-step sequential execution
        step1 = WorkflowStep(
            step_index=0,
            task=intent.task,
            tool_name=selected_tool.name,
            description=f"Step 1: Primary analysis via '{selected_tool.name}'",
            pass_context_from_previous=False,
        )
        step2 = WorkflowStep(
            step_index=1,
            task=secondary_tool.supported_tasks.copy().pop(),
            tool_name=secondary_tool.name,
            description=f"Step 2: Follow-up contextual interpretation via '{secondary_tool.name}'",
            pass_context_from_previous=True,
        )

        return WorkflowPlan(
            intent=intent,
            steps=[step1, step2],
            is_multi_step=True,
        )
