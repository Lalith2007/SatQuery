"""Workflow planning abstraction for SatQuery AI.

TaskPlan serves as the canonical structured workflow representation.
WorkflowPlan for internal execution is derived directly from TaskPlan to ensure
zero divergence between execution and presentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from core.interfaces import BaseSpecialistTool
from core.schemas import ImageInput, TaskIntent, TaskPlan, TaskPlanStep, TaskType


class WorkflowStep(BaseModel):
    """Single executable step in a workflow plan."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = Field(ge=0)
    task: TaskType = Field(description="Task to execute in this step")
    tool_name: str = Field(description="Name of specialist tool to invoke")
    purpose: str = Field(default="", description="Operational description of step")
    pass_context_from_previous: bool = Field(default=True, description="Whether to inject prior step outputs into context")


class WorkflowPlan(BaseModel):
    """Internal execution plan derived directly from canonical TaskPlan."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: TaskIntent = Field(description="Resolved user intent")
    steps: List[WorkflowStep] = Field(min_length=1, description="Sequential steps to execute")
    is_multi_step: bool = Field(default=False, description="True if plan involves multiple specialist invocations")
    canonical_task_plan: Optional[TaskPlan] = Field(default=None, description="Reference to canonical TaskPlan")


class WorkflowPlanner:
    """Constructs canonical TaskPlans and derives executable WorkflowPlans."""

    @classmethod
    def create_task_plan(
        cls,
        intent: TaskIntent,
        selected_tool: BaseSpecialistTool,
        secondary_tool: Optional[BaseSpecialistTool] = None,
        is_composite_query: bool = False,
        images: Optional[List[ImageInput]] = None,
    ) -> TaskPlan:
        """Create the canonical TaskPlan for a request."""
        img_refs = [img.image_id for img in images] if images else []

        if not is_composite_query or secondary_tool is None:
            # Single-step canonical plan
            step = TaskPlanStep(
                step_index=0,
                task=intent.task,
                tool_name=selected_tool.name,
                purpose=f"Execute {intent.task.value} inference via specialist '{selected_tool.name}'.",
                status="planned",
                input_references=img_refs,
                dependencies=[],
                pass_context_from_previous=False,
            )
            return TaskPlan(
                goal=f"Execute {intent.task.value.replace('_', ' ').title()} on provided input imagery.",
                steps=[step],
                is_multi_step=False,
            )

        # Multi-step composite plan
        step1_id = str(uuid.uuid4())
        step1 = TaskPlanStep(
            step_id=step1_id,
            step_index=0,
            task=intent.task,
            tool_name=selected_tool.name,
            purpose=f"Stage 1: Detect and localize surface changes across temporal acquisitions.",
            status="planned",
            input_references=img_refs,
            dependencies=[],
            pass_context_from_previous=False,
        )
        step2_task = next(iter(secondary_tool.supported_tasks))
        step2 = TaskPlanStep(
            step_index=1,
            task=step2_task,
            tool_name=secondary_tool.name,
            purpose=f"Stage 2: Contextual evaluation and fine-grained characterization of localized change regions.",
            status="planned",
            input_references=img_refs,
            dependencies=[step1_id],
            pass_context_from_previous=True,
        )

        return TaskPlan(
            goal=f"Multi-step sequential analysis: {intent.task.value} -> {step2_task.value}",
            steps=[step1, step2],
            is_multi_step=True,
        )

    @classmethod
    def derive_workflow_plan(cls, task_plan: TaskPlan, intent: TaskIntent) -> WorkflowPlan:
        """Derive internal WorkflowPlan directly from canonical TaskPlan."""
        workflow_steps = [
            WorkflowStep(
                step_id=step.step_id,
                step_index=step.step_index,
                task=step.task,
                tool_name=step.tool_name,
                description=step.purpose,
                pass_context_from_previous=step.pass_context_from_previous,
            )
            for step in task_plan.steps
        ]
        return WorkflowPlan(
            plan_id=task_plan.plan_id,
            intent=intent,
            steps=workflow_steps,
            is_multi_step=task_plan.is_multi_step,
            canonical_task_plan=task_plan,
        )

    @classmethod
    def create_plan(
        cls,
        intent: TaskIntent,
        selected_tool: BaseSpecialistTool,
        secondary_tool: Optional[BaseSpecialistTool] = None,
        is_composite_query: bool = False,
        images: Optional[List[ImageInput]] = None,
    ) -> WorkflowPlan:
        """Legacy-compatible convenience method: creates TaskPlan and derives WorkflowPlan."""
        task_plan = cls.create_task_plan(
            intent=intent,
            selected_tool=selected_tool,
            secondary_tool=secondary_tool,
            is_composite_query=is_composite_query,
            images=images,
        )
        return cls.derive_workflow_plan(task_plan, intent)
