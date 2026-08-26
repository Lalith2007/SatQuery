"""Division 3: Bi-Temporal Change Intelligence.

Owner: Dheeraj
Module: specialists/temporal_change/
Shared Contract: core.interfaces.BaseSpecialistTool
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from registry.registry import ToolRegistry, default_registry
    from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool
    from specialists.temporal_change.interfaces import ChangeModel, SemanticReasoner
    from specialists.temporal_change.config import TemporalChangeConfig
    from specialists.temporal_change.model_adapter import MockChangeModel, ChangeFormerAdapter, TinyCDAdapter
    from specialists.temporal_change.semantic_reasoning import (
        MockSemanticReasoner,
        SpatialMetricSynthesizer,
    )
except ImportError:
    # Graceful fallback when running in minimal ML/training environments
    ToolRegistry = Any  # type: ignore
    default_registry = None  # type: ignore
    BiTemporalChangeSpecialistTool = Any  # type: ignore
    ChangeModel = Any  # type: ignore
    SemanticReasoner = Any  # type: ignore
    TemporalChangeConfig = Any  # type: ignore
    MockChangeModel = Any  # type: ignore
    ChangeFormerAdapter = Any  # type: ignore
    TinyCDAdapter = Any  # type: ignore
    MockSemanticReasoner = Any  # type: ignore
    SpatialMetricSynthesizer = Any  # type: ignore


def register_temporal_change_specialist(
    registry: Optional[Any] = None,
    change_model: Optional[Any] = None,
    semantic_reasoner: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Any:
    """Register the Division 3 BiTemporalChangeSpecialistTool into the registry.

    Usage:
        from specialists.temporal_change import register_temporal_change_specialist
        tool = register_temporal_change_specialist(default_registry)
    """
    import os
    from registry.registry import default_registry as def_reg
    from specialists.temporal_change.specialist import BiTemporalChangeSpecialistTool as SpecTool

    if change_model is None:
        ckpt = "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth"
        if os.path.isfile(ckpt):
            try:
                from specialists.temporal_change.model_adapter import TinyCDAdapter
                change_model = TinyCDAdapter(checkpoint_path=ckpt)
            except Exception:
                change_model = None

    reg = registry or def_reg
    tool = SpecTool(
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
    "TinyCDAdapter",
    "MockSemanticReasoner",
    "SpatialMetricSynthesizer",
]
