# SatQuery AI — Division 5 Handoff & Integration Manual

**Author / Developer**: Laksh (`lucifer2007india@gmail.com`)  
**Division**: Division 5 (Evidence, Evaluation & Presentation)  
**Git Branch**: `feature/laksh-evidence-evaluation`  
**Target Recipient**: Lalith Praveen (Division 1 Lead / Core Backend & Orchestration) & Team

---

## 1. Overview & Architectural Role

Division 5 provides the presentation, visual evidence-grounding, evaluation, and intelligence reporting layer for SatQuery AI. It consumes canonical output schemas (`core.schemas.QueryResponse`, `core.schemas.ToolResult`, `core.schemas.Evidence`, `core.schemas.Artifact`, `core.schemas.ExecutionTraceEntry`) and produces interactive visualizations, calibrated confidence metrics, auditable operational traces, multi-format downloadable reports, and modular benchmark evaluations.

Division 5 **strictly respects architectural boundaries**:
- Does **not** mutate Division 1 core schemas or router internals.
- Does **not** modify Division 2-4 model implementations (`specialists/single_image`, etc.).
- Does **not** create a competing backend or secondary agent.
- All 115 tests in the repository pass with zero errors.

---

## 2. Directory Layout & Owned Files

```text
SatQuery/
├── presentation/
│   ├── __init__.py               # Package exports
│   ├── evidence_renderer.py      # Spatial evidence visualizer (BBoxes, Masks, Change Maps, Cross-Modal Fusion)
│   ├── confidence.py             # Calibrated confidence formatting & qualitative tier mapping
│   ├── trace_presenter.py        # Operational execution trace auditor & latency tracker
│   ├── ui_components.py          # Frontend HTML/SVG/Canvas component generators
│   └── README.md                 # Presentation documentation
├── reports/
│   ├── __init__.py               # Package exports
│   ├── generator.py              # Multi-format report builder (JSON, Markdown, HTML)
│   ├── templates.py              # HTML/Markdown report presentation templates
│   └── README.md                 # Report generation documentation
├── evaluation/
│   ├── __init__.py               # Package exports
│   ├── base.py                   # Abstract Base Benchmark Evaluator & MetricResult schemas
│   ├── normalizer.py             # Configurable metric normalizer & weighted aggregator
│   ├── runner.py                 # CLI & programmatic benchmark runner (python -m evaluation.runner)
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── vrsbench.py           # VRSBench VQA & Visual Grounding (mIoU, Precision@0.5)
│   │   ├── rsvqa.py              # RSVQA Presence, Comparison, Count (RMSE/MAE)
│   │   ├── cdvqa.py              # CDVQA Change VQA & BLEU/ROUGE description quality
│   │   └── isro_sac.py           # ISRO/SAC Generic Evaluator (Cartosat-2S & RISAT SAR)
│   └── README.md                 # Evaluation documentation & benchmark formulas
├── app/
│   ├── ui.py                     # Interactive presentation dashboard template (DEMO_HTML)
│   └── routes.py                 # FastAPI endpoints (/api/v1/reports/*, /api/v1/evaluation/*)
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

## 3. Exposed Interfaces & Usage for Lalith

### A. Automatic Evidence Rendering
When `POST /api/v1/query` or `POST /api/v1/query/multipart` is invoked, Division 5 automatically inspects `response.evidence` and renders spatial bounding boxes, change maps, or fusion blends into visual artifacts saved to `artifacts_storage/evidence/`.

To call programmatically:
```python
from presentation.evidence_renderer import EvidenceRenderer

rendered_results = EvidenceRenderer.render_all_evidence(
    evidence_list=query_response.evidence,
    image_inputs=request.images,
    task_hint=query_response.resolved_task.value,
)
for r in rendered_results:
    if r.artifact:
        query_response.artifacts.append(r.artifact)
```

### B. Calibrated Confidence Presentation
```python
from presentation.confidence import ConfidencePresenter

conf_display = ConfidencePresenter.format_confidence(query_response.confidence)
# Returns structured object:
# - raw_score: 0.92
# - formatted_percentage: "92.0%"
# - tier: ConfidenceTier.HIGH ("HIGH", "MODERATE", "LOW", "UNAVAILABLE")
# - color_hex: "#10b981"
# - badge_style: CSS dictionary for UI badges
```

### C. Auditable Execution Trace Presenter
```python
from presentation.trace_presenter import TracePresenter

trace_summary = TracePresenter.format_trace(query_response.execution_trace)
# Strips private chain-of-thought, computes stage durations & percentages, detects failures
markdown_timeline = TracePresenter.render_markdown_timeline(trace_summary)
```

### D. Multi-Format Report Generator
```python
from reports.generator import ReportGenerator

# Generates and persists reports in artifacts_storage/reports/
html_report = ReportGenerator.generate_html_report(query_response, save_to_disk=True)
md_report = ReportGenerator.generate_markdown_report(query_response, save_to_disk=True)
json_report = ReportGenerator.generate_json_report(query_response, save_to_disk=True)

# html_report.artifact is a standard core.schemas.Artifact object
```

### E. Modular Benchmark Evaluator
```python
from evaluation.runner import BenchmarkRunner

result = BenchmarkRunner.run_evaluation(
    benchmark_name="vrsbench",  # "vrsbench" | "rsvqa" | "cdvqa" | "isro_sac"
    predictions=predictions_list,
    ground_truths=ground_truth_list,
    output_dir="evaluation_output",
)
print(f"Aggregate Score: {result.aggregate_normalized_score:.1f} / 100.0")
```

---

## 4. REST API Endpoints Added

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/` or `/demo` | `GET` | Interactive presentation dashboard with live canvas overlays & evaluation lab |
| `/api/v1/reports/generate` | `POST` | Generates HTML, Markdown, or JSON report for a query response |
| `/api/v1/reports/{report_id}` | `GET` | Downloads generated report file directly |
| `/api/v1/evaluation/benchmarks`| `GET` | Lists supported benchmark evaluation suites and metrics |
| `/api/v1/evaluation/run` | `POST` | Executes reproducible benchmark evaluation and returns scoreboard |

---

## 5. Verification & Test Suite Summary

- **Total Test Cases**: **115 passing tests (100% pass rate)**
- **Test Categories**:
  - Division 1 Core & API tests: 55 passed
  - Division 2 Specialist & LoRA tests: 19 passed
  - Division 5 Presentation, Evidence, Trace, Reports & Evaluation tests: 41 passed
- **Execution Command**:
  ```bash
  pytest tests/ -v
  ```

---

## 6. Known Limitations & Operational Assumptions

1. **Raster Resolution**: Spatial bounding box visualization depends on input image dimensions. Normalized coordinates `[ymin, xmin, ymax, xmax]` are scaled directly to raster pixel height and width.
2. **Confidence Calibration**: Division 5 does not invent or alter confidence values. If specialist models do not supply a confidence score, it is strictly presented as `"Unavailable"`.
3. **ISRO/SAC Hidden Datasets**: Because official Cartosat-2S and RISAT test annotations are private, `evaluation.benchmarks.isro_sac.ISROSACGenericEvaluator` is designed generically to compute exact match, token overlap F1, and mIoU once predictions and references are provided at evaluation time.
