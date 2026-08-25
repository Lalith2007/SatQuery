"""Deterministic Task Router for SatQuery AI.

Selects approved tools from the ToolRegistry matching the resolved TaskIntent
and verified image configurations without allowing unrestricted LLM execution.
"""

from __future__ import annotations

from typing import List, Optional

from core.errors import ToolNotFoundError
from core.interfaces import BaseSpecialistTool
from core.logging import get_logger
from core.schemas import ImageInput, TaskIntent, TaskType
from registry.registry import ToolRegistry, default_registry

logger = get_logger("router")


class TaskRouter:
    """Determines and selects the appropriate specialist tool(s) for execution."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or default_registry

    def select_tool(
        self,
        intent: TaskIntent,
        images: List[ImageInput],
    ) -> BaseSpecialistTool:
        """Select a single primary specialist tool for the resolved TaskIntent."""
        candidate_tools = self.registry.find_tools_for_task(intent.task, images=images)

        if not candidate_tools:
            # Check if any tool exists for the task regardless of image filter
            all_for_task = self.registry.find_tools_for_task(intent.task)
            if not all_for_task:
                raise ToolNotFoundError(
                    f"No registered specialist tool found for task '{intent.task.value}'. "
                    f"Available tools: {[t.name for t in self.registry.list_tools()]}"
                )
            else:
                # Tools exist for the task, but image configuration constraints failed
                raise ToolNotFoundError(
                    f"Registered tool(s) for task '{intent.task.value}' do not support "
                    f"{len(images)} images."
                )

        selected = candidate_tools[0]
        logger.info(f"Selected tool '{selected.name}' (v{selected.version}) for task '{intent.task.value}'")
        return selected

    def select_tools_for_workflow(
        self,
        tasks: List[TaskType],
        images: List[ImageInput],
    ) -> List[BaseSpecialistTool]:
        """Select a sequence of specialist tools for a multi-step workflow."""
        selected_tools = []
        for task in tasks:
            candidate_tools = self.registry.find_tools_for_task(task, images=images)
            if not candidate_tools:
                raise ToolNotFoundError(f"Workflow step requires tool for '{task.value}', but none found in registry.")
            selected_tools.append(candidate_tools[0])
        return selected_tools
