"""Extensible Tool and Model Registry for SatQuery AI.

Provides a decoupled registry where specialist tools from Divisions 2, 3, and 4
can be dynamically registered, inspected, health-checked, and resolved by the Agent.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from core.errors import ToolNotFoundError
from core.interfaces import BaseSpecialistTool
from core.logging import get_logger
from core.schemas import ImageInput, TaskType, ToolMetadata

logger = get_logger("registry")


class ToolRegistry:
    """Thread-safe registry for specialist tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseSpecialistTool] = {}
        self._lock = threading.RLock()

    def register(self, tool: BaseSpecialistTool, overwrite: bool = True) -> None:
        """Register a specialist tool."""
        if not isinstance(tool, BaseSpecialistTool):
            raise TypeError(f"Tool must inherit from BaseSpecialistTool, got {type(tool)}")

        with self._lock:
            if tool.name in self._tools and not overwrite:
                raise ValueError(f"Tool '{tool.name}' is already registered.")
            self._tools[tool.name] = tool
            logger.info(f"Registered tool: '{tool.name}' (version {tool.version}) for tasks: {[t.value for t in tool.supported_tasks]}")

    def unregister(self, name: str) -> Optional[BaseSpecialistTool]:
        """Unregister a specialist tool by name."""
        with self._lock:
            tool = self._tools.pop(name, None)
            if tool:
                logger.info(f"Unregistered tool: '{name}'")
            return tool

    def get(self, name: str) -> BaseSpecialistTool:
        """Retrieve a registered tool by its unique name."""
        with self._lock:
            if name not in self._tools:
                raise ToolNotFoundError(f"No specialist tool registered under name '{name}'")
            return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        with self._lock:
            return name in self._tools

    def find_tools_for_task(self, task: TaskType, images: Optional[List[ImageInput]] = None) -> List[BaseSpecialistTool]:
        """Find all registered tools that support the given task and optional image inputs."""
        with self._lock:
            matching = [
                tool for tool in self._tools.values()
                if task in tool.supported_tasks
            ]
            
            # If images are provided, further filter by modality and cardinality constraints
            if images is not None:
                filtered = []
                img_count = len(images)
                for tool in matching:
                    meta = tool.metadata
                    if meta.min_images <= img_count <= meta.max_images:
                        filtered.append(tool)
                return filtered if filtered else matching
            return matching

    def list_tools(self) -> List[ToolMetadata]:
        """List metadata for all currently registered tools."""
        with self._lock:
            return [tool.metadata for tool in self._tools.values()]

    def health_check_all(self) -> Dict[str, bool]:
        """Perform health checks on all registered tools."""
        with self._lock:
            results = {}
            for name, tool in self._tools.items():
                try:
                    results[name] = tool.health_check()
                except Exception as e:
                    logger.error(f"Health check failed for tool '{name}': {e}")
                    results[name] = False
            return results

    def clear(self) -> None:
        """Clear all registered tools (primarily for test resets)."""
        with self._lock:
            self._tools.clear()


# Global default registry instance
default_registry = ToolRegistry()
