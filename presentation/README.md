# Division 5: Evidence, Evaluation & Presentation Layer

**Developer / Owner**: Laksh (`lucifer2007india@gmail.com`)  
**Git Branch**: `feature/laksh-evidence-evaluation`  
**Shared Contracts Consumed**: `core.schemas.QueryResponse`, `core.schemas.ToolResult`, `core.schemas.Evidence`, `core.schemas.Artifact`, `core.schemas.ExecutionTraceEntry`

---

## 1. Executive Summary & Purpose

Division 5 is the presentation, evidence-grounding, evaluation, and reporting subsystem of **SatQuery AI**. It sits immediately downstream of the specialist inference and agent orchestration backbone.

Its core responsibilities are:
1. **Evidence Handling & Spatial Rendering**: Ingesting standardized `Evidence` items and rendering high-visibility bounding boxes (`[ymin, xmin, ymax, xmax]`), segmentation/change masks, bi-temporal side-by-side change maps, optical-SAR cross-modal blends, localized ROI crops, and attention heatmaps saved as persistent artifacts.
2. **Honest Confidence Calibration & Presentation**: Preserving raw specialist confidence scores, formatting them into percentages and decimals, classifying them into qualitative tiers (`HIGH`, `MODERATE`, `LOW`, `UNAVAILABLE`), with a strict non-fabrication guarantee (`None` is explicitly displayed as `"Unavailable"`).
3. **Auditable Operational Execution Trace Presentation**: Formatting stage-by-stage operational execution events, calculating stage latencies and percentage contributions, and enforcing strict privacy safeguards against private chain-of-thought exposure.
4. **Interactive Web Presentation Dashboard**: Delivering an interactive, responsive web interface directly from FastAPI with 1-click judging presets, SVG/Canvas bounding box overlays, before/after change comparison sliders, and real-time execution observability.
5. **Multi-Format Report Generation**: Automated generation of standalone interactive HTML reports, GitHub-flavored Markdown reports, and machine-readable JSON audit dumps.
6. **Modular Benchmark Evaluation Suite**: Reproducible evaluation framework for **VRSBench**, **RSVQA**, **CDVQA**, and generic **ISRO/SAC** multi-sensor test sets with score normalization.

---

## 2. Architectural Pipeline

```text
Specialist Models (Div 2, 3, 4) 
               │
               ▼
   Standardized ToolResult
               │
               ▼
   Division 1 Agent Aggregator
               │
               ▼
   Standardized QueryResponse
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │            DIVISION 5 (Laksh — Owner)                  │
   │                                                        │
   │  ┌───────────────────────┬──────────────────────────┐  │
   │  ▼                       ▼                          ▼  │
   │ Evidence Renderer   Confidence Presenter   Trace Presenter
   │ (BBoxes, Masks,     (Calibrated Tiers,     (Operational │
   │  Change Maps, Crops) Non-Fabrication)       Audit Trail)│
   │  └───────────────────────┼──────────────────────────┘  │
   │                          ▼                             │
   │                 Reports Generator                      │
   │            (Interactive HTML, MD, JSON)                │
   │                          │                             │
   │                          ▼                             │
   │            Interactive Presentation UI                 │
   │            (Canvas Overlays, Slider)                   │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                      User / Judging Panel
```

---

## 3. Directory Layout & Owned Files

```text
SatQuery/
├── presentation/
│   ├── __init__.py               # Package exports
│   ├── evidence_renderer.py      # Spatial evidence visualizer (BBoxes, Masks, Storyboards)
│   ├── confidence.py             # Calibrated confidence formatting & qualitative tiers
│   ├── trace_presenter.py        # Operational execution trace auditor & latency tracker
│   ├── ui_components.py          # Reusable frontend HTML/SVG/Canvas component generators
│   └── README.md                 # Presentation documentation
├── reports/
│   ├── __init__.py               # Package exports
│   ├── generator.py              # Multi-format report builder (JSON, Markdown, HTML)
│   ├── templates.py              # HTML/Markdown report templates
│   └── README.md                 # Report generation documentation
├── evaluation/
│   ├── __init__.py               # Package exports
│   ├── base.py                   # Abstract evaluator interface & MetricResult schemas
│   ├── normalizer.py             # Normalization strategies (scale_100, min_max, invert_error)
│   ├── runner.py                 # CLI & programmatic benchmark execution runner
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── vrsbench.py           # VRSBench VQA & Visual Grounding (mIoU, P@0.5)
│   │   ├── rsvqa.py              # RSVQA Presence, Comparison, Count (RMSE)
│   │   ├── cdvqa.py              # CDVQA Change VQA & BLEU/ROUGE description
│   │   └── isro_sac.py           # ISRO/SAC Generic Evaluator (Cartosat-2S + RISAT SAR)
│   └── README.md                 # Benchmark documentation & CLI guide
├── app/
│   ├── ui.py                     # Presentation dashboard template (DEMO_HTML)
│   └── routes.py                 # API endpoints (/api/v1/reports/*, /api/v1/evaluation/*)
└── tests/
    ├── test_evidence_renderer.py
    ├── test_confidence_presentation.py
    ├── test_trace_presenter.py
    ├── test_report_generation.py
    ├── test_evaluation_benchmarks.py
    ├── test_division5_contracts.py
    ├── test_division5_failures.py
    └── test_division5_integration.py
```

---

## 4. Key Interfaces & Schemas

### A. Evidence Object Ingestion (`core.schemas.Evidence`)
Division 5 consumes standardized evidence items:
```json
{
  "id": "e1-456",
  "type": "bounding_box",
  "label": "Airport Runway",
  "confidence": 0.95,
  "data": {
    "bbox": [0.082, 0.399, 0.942, 0.624],
    "format": "[ymin, xmin, ymax, xmax]",
    "coordinate_system": "normalized_image_coordinates (0.0 - 1.0)"
  },
  "image_id": "img-001"
}
```

### B. Calibrated Confidence (`presentation.confidence.ConfidenceDisplay`)
- `HIGH`: $\ge 0.85$ (Green `#10b981`, strong model certainty)
- `MODERATE`: $0.65 \le c < 0.85$ (Amber `#f59e0b`, verification recommended)
- `LOW`: $< 0.65$ (Red `#ef4444`, significant uncertainty)
- `UNAVAILABLE`: `None` (Gray `#9ca3af`, strictly un-fabricated)

### C. Operational Trace (`presentation.trace_presenter.ExecutionTraceSummary`)
- Stages: `INPUT_VALIDATED`, `TASK_RESOLVED`, `TOOL_SELECTED`, `INFERENCE_EXECUTED`, `EVIDENCE_GENERATED`, `RESULT_AGGREGATED`.
- Privacy Guard: Automatically filters out private chain-of-thought keys (`chain_of_thought`, `thought`, `cot`, `reasoning`).

---

## 5. Running Tests

```bash
# Run all Division 5 unit and integration test suites
pytest tests/test_evidence_renderer.py \
       tests/test_confidence_presentation.py \
       tests/test_trace_presenter.py \
       tests/test_report_generation.py \
       tests/test_evaluation_benchmarks.py \
       tests/test_division5_contracts.py \
       tests/test_division5_failures.py \
       tests/test_division5_integration.py -v

# Run entire repository test suite (115 passing tests)
pytest tests/ -v
```
