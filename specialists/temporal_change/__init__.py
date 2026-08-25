"""Division 3: Bi-Temporal Change Intelligence.

Owner: Dheeraj
Module: specialists/temporal_change/
Shared Contract: core.interfaces.BaseSpecialistTool
"""

from __future__ import annotations

from typing import Optional

from registry.registry import ToolRegistry, default_registry
from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool
from specialists.temporal_change.interfaces import ChangeModel, SemanticReasoner
from specialists.temporal_change.config import TemporalChangeConfig
from specialists.temporal_change.model_adapter import MockChangeModel, ChangeFormerAdapter
from specialists.temporal_change.semantic_reasoning import (
    MockSemanticReasoner,
    SpatialMetricSynthesizer,
)


def register_temporal_change_specialist(
    registry: Optional[ToolRegistry] = None,
    change_model: Optional[ChangeModel] = None,
    semantic_reasoner: Optional[SemanticReasoner] = None,
    config: Optional[TemporalChangeConfig] = None,
) -> BiTemporalChangeSpecialistTool:
    """Register the Division 3 BiTemporalChangeSpecialistTool into the registry.

    Usage:
        from specialists.temporal_change import register_temporal_change_specialist
        tool = register_temporal_change_specialist(default_registry)
    """
    reg = registry or default_registry
    tool = BiTemporalChangeSpecialistTool(
        change_model=change_model,
        semantic_reasoner=semantic_reasoner,
        config=config,
    )
    reg.register(tool, overwrite=True)
    return tool


__all__ = [
    "BiTemporalChangeSpecialistTool",
    "register_temporal_change_specialist",
    "ChangeModel",
    "SemanticReasoner",
    "TemporalChangeConfig",
    "MockChangeModel",
    "ChangeFormerAdapter",
    "MockSemanticReasoner",
    "SpatialMetricSynthesizer",
]
