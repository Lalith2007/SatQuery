# Division 3: Bi-Temporal Change Intelligence

**Owner:** Dheeraj  
**Module:** `specialists/temporal_change/`  
**Status:** Implemented and tested (44 tests, all passing)

## Overview

This module provides bi-temporal change detection, spatial localization, and change-based visual question answering for remote sensing imagery. It accepts exactly two co-registered images (T0 and T1) and a natural-language query, producing a structured `ToolResult` with change maps, bounding boxes, and textual analysis.

## Architecture

```
T0 + T1 + Query
    → 6-Point Validation
    → Controlled Geospatial Alignment
    → Pluggable ChangeModel
    → Postprocessing / Localization
    → Query Intent Classification
    → Pluggable SemanticReasoner
    → Evidence Packaging
    → Canonical ToolResult
```

## Module Structure

```
specialists/temporal_change/
├── __init__.py            # Package init + registration helper
├── config.py              # Environment-configurable settings (SATQUERY_TC_*)
├── errors.py              # Error taxonomy extending core.errors
├── evidence.py            # Evidence and artifact generation
├── interfaces.py          # ChangeModel, SemanticReasoner ABCs
├── model_adapter.py       # MockChangeModel, ChangeFormerAdapter
├── postprocessing.py      # Thresholding, morphology, connected components
├── preprocessing.py       # Image loading, normalization, model-input resize
├── semantic_reasoning.py  # MockSemanticReasoner, SpatialMetricSynthesizer
├── specialist.py          # BiTemporalChangeSpecialistTool (main entry)
├── utils.py               # Geospatial metadata utilities
└── validation.py          # 6-point pair validation
```

## Quick Start

```python
from registry.registry import ToolRegistry
from specialists.temporal_change import register_temporal_change_specialist

# Register into any ToolRegistry
registry = ToolRegistry()
tool = register_temporal_change_specialist(registry)

# Or use pluggable backends
from specialists.temporal_change import (
    BiTemporalChangeSpecialistTool,
    ChangeFormerAdapter,
    SpatialMetricSynthesizer,
)

tool = BiTemporalChangeSpecialistTool(
    change_model=ChangeFormerAdapter(checkpoint_path="path/to/weights.pth"),
    semantic_reasoner=SpatialMetricSynthesizer(),
)
```

## Configuration

All settings are environment-configurable with `SATQUERY_TC_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SATQUERY_TC_DEVICE` | `auto` | Compute device (auto, cpu, cuda, mps) |
| `SATQUERY_TC_CHANGE_THRESHOLD` | `0.5` | Binary change threshold |
| `SATQUERY_TC_MIN_REGION_AREA` | `100` | Minimum changed region area (pixels) |
| `SATQUERY_TC_USE_MOCK` | `true` | Use mock model for testing |
| `SATQUERY_TC_ALLOW_REPROJECTION` | `false` | Enable controlled geospatial resampling |
| `SATQUERY_TC_INPUT_SIZE` | `256` | Model input square dimension |

## Testing

```bash
uv run pytest tests/test_temporal_change_specialist.py -v
```

44 tests across 6 gates: Unit, Contract, Mock-Service, Sample Inference, Registry, Agent Integration.

## Integration Notes

- Fully isolated from Division 2 (`specialists/single_image/`)
- Inherits from `core.interfaces.BaseSpecialistTool`
- Consumes `core.schemas.ToolRequest`, returns `core.schemas.ToolResult`
- Uses existing `core.errors` hierarchy without creating competing error classes
- Registers via `registry.registry.ToolRegistry`
