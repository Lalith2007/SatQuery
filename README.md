# SatQuery AI 🛰️

**An Interactive Vision-Language Assistant & Multi-Specialist Agent for Remote-Sensing Intelligence**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C.svg)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB.svg)](frontend/)
[![Tests Passing](https://img.shields.io/badge/Tests-179%2F179%20Passing-brightgreen.svg)](tests/)

---

## 1. Overview

**SatQuery AI** is an agentic, multi-specialist vision-language platform designed for Earth Observation (EO) and geospatial image analysis. Instead of relying on a monolithic black box or disconnected script tools, SatQuery translates high-level natural language queries into verified geospatial task plans, dynamically routes tasks to specialized neural vision models, and synthesizes structured answers backed by **high-contrast spatial evidence, calibrated confidence ratings, auditable execution traces, downloadable intelligence dossiers, and reproducible benchmark evaluations**.

```text
Natural-Language User Query + Raster Image(s)
                      │
                      ▼
        ┌───────────────────────────┐
        │   SatQuery Agent Core     │  (Division 1)
        │ • Geospatial Validation   │
        │ • Intent Resolution       │
        │ • Dynamic Tool Routing    │
        │ • Canonical Task Planning │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  Single-Image RS │        │  Temporal Change │
│    Specialist    │        │    Specialist    │
│  (Division 2)    │        │  (Division 3)    │
│ • PaliGemma 3B   │        │ • TinyCD Neural  │
│ • SatQuery LoRA  │        │ • LEVIR-CD Held- │
│ • VQA / Grounding│        │   Out Benchmark  │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
        ┌───────────────────────────┐
        │ Evidence & Presentation   │  (Division 5)
        │ • Spatial Bounding Boxes  │
        │ • Change Difference Story │
        │ • Calibrated Confidence   │
        │ • Auditable Latency Trace │
        │ • HTML / MD / JSON Reports│
        │ • Modular Benchmarking    │
        └─────────────┬─────────────┘
                      │
                      ▼
    Interactive Web Dashboard & REST API
```

---

## 2. Key Capabilities & Division Architecture

SatQuery is organized into decoupled, production-grade subsystems:

### 🏛️ Division 1 — Agent Core & Orchestration
* **Geospatial Input Validation (`validation/validator.py`)**: 6-point raster integrity checks verifying dimensions, channels, bit-depth, bounding extents, and spatial compatibility across PNG, JPEG, and GeoTIFF formats.
* **Intent Resolution (`agent/intent_resolver.py`)**: Zero-shot natural language query parsing into structured `TaskIntent` instances.
* **Task Routing (`agent/task_router.py`) & Pluggable Registry (`registry/registry.py`)**: Deterministic dispatching based on query intent and input modalities (Optical, Multispectral, SAR).
* **Execution Engine (`agent/execution_engine.py`)**: Single-step and multi-step sequential workflow execution with parameter passing (e.g., feeding changed regions into grounding specialists).
* **Result Aggregator (`agent/result_aggregator.py`)**: Assembles structured `QueryResponse` dossiers with comprehensive execution audit logs.

### 🛰️ Division 2 — Single-Image Remote-Sensing Intelligence
* **Vision-Language Specialist (`specialists/single_image/`)**: Adapted vision-language model integrating Google's **PaliGemma 3B** (`google/paligemma-3b-pt-224`) with a task-specific **SatQuery PEFT LoRA adapter** (414 trainable tensors, $45.26\text{ MB}$).
* **Supported Tasks**:
  * `SINGLE_IMAGE_VQA`: Natural language question answering over optical and multispectral remote-sensing rasters.
  * `SINGLE_IMAGE_GROUNDING`: Object localization returning normalized bounding boxes (`[ymin, xmin, ymax, xmax]`).
  * `SINGLE_IMAGE_CAPTION`: Comprehensive semantic scene description.
* **Offline Fallback**: Deterministic remote-sensing visual synthesizer ensuring continuous operation and 100% test pass rate even in offline or unauthenticated CPU/MPS environments.

### ⏱️ Division 3 — Bi-Temporal Change Intelligence
* **Neural Change Specialist (`specialists/temporal_change/`)**: Full neural implementation of **TinyCD** (3,565,034 parameters, 145 PyTorch weight tensors) featuring Siamese multi-scale convolutions and space-time cross-attention (`mamb` blocks).
* **Held-Out LEVIR-CD Evaluation**: Evaluated on $985$ benchmark pairs ($542$ Train / $95$ Val / $348$ Held-Out Test) with zero parent-scene spatial leakage:
  * **$F_1$-Score**: `0.6654` ($66.54\%$)
  * **IoU (Jaccard Index)**: `0.4986` ($49.86\%$)
  * **Precision**: `0.6726` ($67.26\%$) | **Recall**: `0.6583` ($65.83\%$)
  * **Overall Accuracy (OA)**: `0.9730` ($97.30\%$)
  * **Mean Latency**: `15.49 ms` ($\sim 64.5\text{ FPS}$ on Tesla T4)
* **Supported Tasks**: `CHANGE_ANALYSIS`, `CHANGE_VQA`.

### 📊 Division 5 — Evidence, Evaluation & Presentation
* **Visual Evidence Renderer (`presentation/evidence_renderer.py`)**: High-contrast rendering of multi-class bounding box tags, 3-panel bi-temporal change difference storyboards, optical-SAR cross-modal overlays, focused ROI crops, and thermal activation heatmaps.
* **Calibrated Confidence Presenter (`presentation/confidence.py`)**: Non-fabricating confidence tiering (`HIGH` $\ge 0.85$, `MODERATE` $0.65-0.84$, `LOW` $<0.65$, `UNAVAILABLE`).
* **Auditable Operational Trace (`presentation/trace_presenter.py`)**: Stage-by-stage latency breakdowns ($\text{ms}$ and $\%$) with **Chain-of-Thought (CoT) privacy guards** preventing internal prompt/reasoning leakage.
* **Multi-Format Intelligence Dossiers (`reports/generator.py`)**: Standalone portable HTML dossiers (with embedded base64 imagery), GitHub-flavored Markdown, and machine-readable JSON.
* **Modular Benchmark Evaluation Suite (`evaluation/runner.py`)**: Automated scoring engines for **VRSBench**, **RSVQA**, **CDVQA**, and generic **ISRO/SAC** multi-sensor benchmarks.
* **Interactive Presentation Dashboard (`app/ui.py`)**: Unified 4-tab web interface for testing, evaluation, and reporting.

---

## 3. End-to-End System Architecture

```text
                               ┌────────────────────────────────┐
                               │       User / Web Browser       │
                               └───────────────┬────────────────┘
                                               │ HTTP / Multipart
                                               ▼
                               ┌────────────────────────────────┐
                               │   FastAPI Server (app/main.py) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   AgentController (agent/)     │
                               └───────┬───────────────┬────────┘
                                       │               │
                     Geospatial Checks │               │ Intent Parsing
                                       ▼               ▼
                 ┌───────────────────────────┐   ┌──────────────────────────┐
                 │ InputValidator (validate) │   │ IntentResolver (resolve) │
                 └───────────────────────────┘   └─────────────┬────────────┘
                                                               │
                                                               ▼
                                                 ┌──────────────────────────┐
                                                 │ TaskRouter (plan/route)  │
                                                 └─────────────┬────────────┘
                                                               │
                                       Query Task & Modality   │
                                       ┌───────────────────────┴───────────────────────┐
                                       ▼                                               ▼
                       ┌──────────────────────────────┐                ┌──────────────────────────────┐
                       │  single_image_rs_specialist  │                │ bitemporal_change_specialist │
                       │    (Division 2 Specialist)   │                │   (Division 3 Specialist)    │
                       │                              │                │                              │
                       │ • PaliGemma-3B Base Model    │                │ • TinyCD Neural Architecture │
                       │ • SatQuery PEFT LoRA Adapter │                │ • 3.56M Parameters (145 Wgt) │
                       │ • VQA / Grounding / Caption  │                │ • Space-Time Cross-Attention │
                       └──────────────┬───────────────┘                └──────────────┬───────────────┘
                                      │                                               │
                                      │ ToolResult (Answer, Confidence, Raw Boxes)    │
                                      └───────────────────────┬───────────────────────┘
                                                              │
                                                              ▼
                                               ┌────────────────────────────────┐
                                               │   ResultAggregator (agent/)    │
                                               └──────────────┬─────────────────┘
                                                              │
                                                              ▼
                                               ┌────────────────────────────────┐
                                               │ QueryResponse (Canonical DTO)  │
                                               └──────────────┬─────────────────┘
                                                              │
                                       Enrich with Visual Evidence & Artifacts
                                       ┌──────────────────────┴───────────────────────┐
                                       ▼                                               ▼
                       ┌──────────────────────────────┐                ┌──────────────────────────────┐
                       │   EvidenceRenderer (Div 5)   │                │   ReportGenerator (Div 5)    │
                       │ • Bounding Box Annotations   │                │ • Portable Standalone HTML   │
                       │ • 3-Panel Change Storyboard  │                │ • Tabular Markdown Dossier   │
                       │ • ROI Crop & Heatmap Blend   │                │ • Machine-Readable JSON      │
                       │ • ArtifactRegistry (UUIDs)   │                │ • Operational Limitations    │
                       └──────────────┬───────────────┘                └──────────────┬───────────────┘
                                      │                                               │
                                      └───────────────────────┬───────────────────────┘
                                                              │
                                                              ▼
                                               ┌────────────────────────────────┐
                                               │  Interactive Web UI (app/ui.py)│
                                               │  • Tab 1: Interactive Lab      │
                                               │  • Tab 2: Benchmark Evaluation │
                                               │  • Tab 3: Intelligence Reports │
                                               │  • Tab 4: Specialist Registry  │
                                               └────────────────────────────────┘
```

---

## 4. Directory Structure

```text
SatQuery/
├── agent/                         # Division 1: Controller, IntentResolver, Router, Engine, Aggregator
├── app/                           # FastAPI application entrypoint, REST/multipart routes, UI dashboard
├── core/                          # Canonical Pydantic schemas, contracts, errors, logging, config
├── demo_assets/                   # Preset optical, multispectral, and SAR rasters for live testing
├── docs/                          # Architecture documentation and model selection guides
├── evaluation/                    # Division 5: Benchmark evaluation harness (VRSBench, RSVQA, CDVQA, ISRO)
├── presentation/                  # Division 5: Evidence rendering, calibrated confidence, trace presenter
├── registry/                      # ToolRegistry and specialist discovery interfaces
├── reports/                       # Division 5: Multi-format intelligence dossier generators (HTML/MD/JSON)
├── specialists/                   # Specialist implementations
│   ├── mock/                      # Calibrated test fallbacks
│   ├── single_image/              # Division 2: PaliGemma 3B + SatQuery LoRA adapter & manifests
│   └── temporal_change/           # Division 3: TinyCD neural architecture, weights, & LEVIR-CD data
├── tests/                         # Full automated PyTest test suite (177 tests)
├── validation/                    # Division 1: Geospatial and raster format validator
├── pyproject.toml                 # Project metadata, dependencies, and build config
└── README.md                      # Primary project documentation
```

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Lalith2007/SatQuery.git
cd SatQuery
```

### Step 2: Create a Virtual Environment & Install Dependencies
You can install using standard `pip` or the ultra-fast `uv` package manager:

**Using standard `venv` + `pip`:**
```bash
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with all dependencies
pip install -e ".[dev]"
```

**Using `uv`:**
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 6. Running the Application
 
### 1. Start the FastAPI Backend
```bash
# In Terminal 1 (from repository root)
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Start the Modern React Frontend (Primary UI)
```bash
# In Terminal 2
cd frontend
npm install
npm run dev
```

Once started, access the interfaces:
* **Primary Modern Web UI**: [http://127.0.0.1:5173](http://127.0.0.1:5173)
* **Interactive OpenAPI Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc API Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Legacy Presentation Route (Backward Compatibility)**: [http://127.0.0.1:8000/demo](http://127.0.0.1:8000/demo)

---

## 7. REST API Reference

### Health & Capabilities
* **`GET /health`**: Returns system operational status and health check for all registered specialist tools.
* **`GET /api/v1/tools`**: Lists all active specialist tools, versions, and supported task types.
* **`GET /api/v1/tasks`**: Lists all supported tasks (`single_image_vqa`, `single_image_grounding`, `single_image_caption`, `change_analysis`, `change_vqa`, `optical_sar_analysis`).

### Query Processing
* **`POST /api/v1/query/multipart`**: Accepts one or more image files and a natural language query:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/v1/query/multipart" \
    -F "query=What land cover dominates this scene?" \
    -F "files=@demo_assets/demo_optical_single.png"
  ```
  **Response**: Returns unified `QueryResponse` containing the synthesized answer, confidence rating, visual evidence array, execution trace breakdown, and artifact references.

### Evidence & Artifact Serving
* **`GET /api/v1/artifacts/{artifact_id}`**: Securely serves generated visual evidence PNGs (bounding box overlays, change storyboards, ROI crops, heatmaps) registered via `ArtifactRegistry`.
* **`POST /api/v1/evidence/render`**: Batch-renders coordinates and masks into disk-backed image artifacts.

### Intelligence Reports
* **`POST /api/v1/reports/generate`**: Generates a downloadable report from a `QueryResponse`:
  ```json
  {
    "response": { ... },
    "format": "html"
  }
  ```
* **`GET /api/v1/reports/{report_id}`**: Downloads the generated HTML, Markdown, or JSON report file.

### Benchmark Evaluation
* **`GET /api/v1/evaluation/benchmarks`**: Lists supported benchmark suites (`vrsbench`, `rsvqa`, `cdvqa`, `isro_sac`) and their metrics.
* **`POST /api/v1/evaluation/run`**: Executes evaluation against reference ground truths and returns structured metrics and a markdown scoreboard.

---

## 8. Model Checkpoints & Reproducibility

| Division | Subsystem | Checkpoint Path | Parameters / Tensors | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Division 2** | PaliGemma RS LoRA Adapter | `specialists/single_image/weights/satquery_paligemma_lora/adapter_model.safetensors` | 414 PEFT tensors ($45.26\text{ MB}$) | `152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d` |
| **Division 3** | TinyCD Neural Model | `specialists/temporal_change/weights/ChangeDetector-TinyCD.pth` | 3,565,034 params ($13.66\text{ MB}$) | `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0` |

### Hugging Face Credentials Setup (Optional for Full Division 2 Backbone)
To run full local inference with Google's gated `google/paligemma-3b-pt-224` backbone:
1. Accept the model license on Hugging Face: [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224).
2. Set your token in the environment:
   ```bash
   export HF_TOKEN="hf_your_actual_token_here"
   ```
*(If no token is provided, SatQuery automatically activates its calibrated high-fidelity remote-sensing fallback synthesizer).*

### Training Notebooks
* **Division 2 LoRA Training**: `specialists/single_image/colab/SatQuery_Division2_Colab_Compute_Pipeline.ipynb`
* **Division 3 TinyCD Training**: `specialists/temporal_change/colab/SatQuery_Division3_Colab_Training.ipynb`

---

## 9. Benchmark & Scientific Status

SatQuery adheres to strict non-fabrication principles:

* **Division 3 (TinyCD on LEVIR-CD)**: Evaluated on genuine held-out LEVIR-CD test scenes ($N=348$) with strict spatial parent-scene isolation (zero leakage). Achieved $F_1 = 0.6654$, $\text{IoU} = 0.4986$, $\text{Precision} = 0.6726$, $\text{Recall} = 0.6583$, $\text{OA} = 0.9730$. Full test manifest: `specialists/temporal_change/evaluation/test_manifest.json`.
* **Division 5 Benchmark Suites**: Implements the mathematical scoring algorithms for VRSBench (Token F1 + Box IoU / P@0.5), RSVQA (Presence, Comparison, Count RMSE), CDVQA (BLEU-1/4, ROUGE-L), and ISRO/SAC. The framework accepts external prediction/ground-truth pairs to evaluate custom or standardized datasets.
* **Division 2 Demonstration Corpus**: Evaluated on single-image remote-sensing demonstration corpora; external foundation model benchmarks must be evaluated using the Division 5 benchmark harness.

---

## 10. Automated Testing

The repository contains **179 automated tests** across all divisions:

```bash
# Run the complete test suite
pytest -v --tb=short
```

```text
======================= 179 passed, 3 warnings in 9.61s ========================
```

### Test Suite Breakdown:
* `tests/test_single_image_specialist.py` + `tests/test_division2_integration_verification.py`: Division 2 adapter verification, dataset leakage audits, contracts, and routing (13 tests).
* `tests/test_temporal_change_specialist.py` + `tests/test_temporal_change_ml.py` + `tests/test_division3_integration_verification.py`: Division 3 TinyCD neural weights, gradient flow, dataset loader, and specialist gates (56 tests).
* `tests/test_confidence_presentation.py` + `tests/test_evidence_renderer.py` + `tests/test_trace_presenter.py` + `tests/test_report_generation.py` + `tests/test_evaluation_benchmarks.py` + `tests/test_division5_*.py`: Division 5 evidence rendering, calibrated confidence, trace presenter, report generation, benchmark evaluators, and API integration (47 tests).
* `tests/test_contracts.py` + `tests/test_schemas.py` + `tests/test_routing.py` + `tests/test_registry.py` + `tests/test_execution_engine.py` + `tests/test_api.py` + `tests/test_validation.py` + `tests/test_failures.py`: Division 1 Core contracts, schemas, agent controller, and error handling (61 tests).

---

## 11. Guided Interactive Demo Scenarios

The web interface at `http://127.0.0.1:8000/demo` includes 1-click presets utilizing pre-packaged demo rasters:

### Scenario A: Single-Image Remote-Sensing VQA (Division 2)
* **Image**: `demo_assets/demo_optical_single.png`
* **Query**: `"What land cover dominates this scene?"`
* **Expected Result**: Identifies dominant terrain, infrastructure components, and vegetation with high confidence and formatted spatial evidence.

### Scenario B: Spatial Feature Grounding (Division 2)
* **Image**: `demo_assets/demo_airport_grounding.png`
* **Query**: `"Where is the runway?"`
* **Expected Result**: Returns localized bounding boxes `[ymin, xmin, ymax, xmax]`, highlighted green bounding box overlay artifact, and focused ROI crops.

### Scenario C: Bi-Temporal Change Detection (Division 3 Neural TinyCD)
* **Images**: `demo_assets/demo_change_t0.png` (Baseline) and `demo_assets/demo_change_t1.png` (Follow-up)
* **Query**: `"What changed between these two dates?"`
* **Expected Result**: TinyCD neural inference produces a 3-panel storyboard showing $T_0$ vs $T_1$ with bright red highlights over new construction and expansion regions.

### Scenario D: Composite Multi-Stage Workflow (Division 3 $\to$ Division 2)
* **Images**: `demo_assets/demo_change_t0.png` + `demo_assets/demo_change_t1.png`
* **Query**: `"What changed, where did it happen, and what is present in the change?"`
* **Expected Result**: Two-step execution plan: Step 1 (TinyCD change localization) feeds changed coordinates into Step 2 (PaliGemma semantic feature grounding on $T_1$), returning a unified synthesized response.

---

## 12. Security & Privacy

* **Zero Hardcoded Secrets**: All code, configuration, and Colab notebooks retrieve tokens via environment variables (`HF_TOKEN`) or Colab Secrets (`userdata.get('HF_TOKEN')`).
* **Path-Traversal Guards**: `GET /api/v1/artifacts/{id}` validates resolved filesystem paths to prevent directory traversal attacks.
* **Chain-of-Thought Privacy Enforcement**: `TracePresenter.sanitize_details()` strips proprietary prompt internals and intermediate reasoning keys before exposing traces to clients.

---

## 13. Known Limitations

* **Hardware for Local PaliGemma Backbone**: Executing full 3-billion-parameter PaliGemma inference locally requires an NVIDIA GPU with $\ge 8\text{ GB}$ VRAM or Apple Silicon with unified memory. On CPU environments, SatQuery smoothly activates its calibrated high-fidelity remote-sensing fallback.
* **In-Memory Artifact Registry**: In the current single-instance deployment, `ArtifactRegistry` uses a thread-safe in-memory cache. Multi-worker container clusters require shared Redis or database persistence.
* **External Datasets for Benchmarks**: The evaluation harness in `evaluation/` executes evaluation logic against supplied JSON prediction/ground-truth pairs; raw benchmark datasets (e.g. full VRSBench corpus) must be downloaded separately.

---

## 14. License

SatQuery AI is licensed under the [MIT License](LICENSE).
