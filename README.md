# SatQuery AI

**An Agentic Vision-Language Assistant for Remote-Sensing Imagery**

---

## 🛰️ Architecture Overview

SatQuery AI is designed around a decoupled, modular, agentic architecture. Division 1 provides the core backend, orchestration engine, canonical data contracts, and tool registry that coordinates specialist intelligence models.

```
USER / WEB UI (Division 5 - Manoj / Presentation)
       │
       ▼
 FastAPI BACKEND (Division 1 - Lalith)
       │
       ▼
 AGENT CONTROLLER (Division 1 - Lalith)
 ┌─────┴────────────────────────────────────────────────┐
 │                                                      │
 ▼                                                      ▼
Geospatial Input Validator                  Intent Resolver
(GeoTIFF/TIFF/PNG/JPEG Validation)           (NL Query ➔ TaskIntent)
       │                                                │
       └────────────────────────┬───────────────────────┘
                                ▼
                        CANONICAL TASK PLAN
                   (TaskPlan & TaskPlanStep)
                                │
                                ▼
                       DETERMINISTIC ROUTER
                                │
                        TOOL REGISTRY
 ┌──────────────────────────────┼──────────────────────────────┐
 │                              │                              │
 ▼                              ▼                              ▼
Single-Image (Div 2)         Bi-Temporal (Div 3)            Optical-SAR (Div 4)
- VQA                        - Change Analysis              - Cross-Modal Fusion
- Caption                    - Change VQA                   - All-Weather Detection
- Grounding                  - Change Maps
 └──────────────────────────────┬──────────────────────────────┘
                                ▼
                     EXECUTION ENGINE (Sequential)
                                │
                                ▼
                        RESULT AGGREGATOR
                                │
                                ▼
                    Standardized QueryResponse
  (Answer + Evidence + Agent Decision Card + TaskPlan + Trace + Artifacts)
```

---

## 🔄 Architecture: Before vs Enhanced

| Stage | Baseline Concept | SatQuery AI Division 1 Architecture |
| :--- | :--- | :--- |
| **Input Ingestion** | Generic image loading | **Geospatial & Remote-Sensing Aware**: Inspects GeoTIFF tags, multi-band rasters, verified bounds without fabricating missing metadata. |
| **Language Understanding** | Raw prompt forwarded to blackbox | **Decoupled Intent Resolution**: Resolves natural language into structured `TaskIntent` (`task`, `confidence`, `intent_explanation`, `target_features`). |
| **Workflow Planning** | Single static tool call | **Explicit Canonical TaskPlan**: `TaskPlan` with structured operational steps, goals, dependencies, and single/multi-tool orchestration. |
| **Tool Selection** | Hard-coded if/else | **Pluggable Tool Registry**: Deterministic routing matching intent & image modalities against registered `BaseSpecialistTool` interfaces. |
| **Observability** | Opaque blackbox | **Live Agent Decision Card & Audit Trace**: Shows "Why This Tool?" operational explanation, detected modalities, and millisecond execution timings per stage. |
| **Modularity & Swapping** | Monolithic coupling | **Pluggable & Demonstrable**: Mock specialists can be swapped with new/production specialists in `ToolRegistry` with zero agent redesign. |

---

## 👥 Team & Division Structure

- **Division 1 (Lalith Praveen - Lead)**: Agent Core, Backend, Orchestration, Validation, Registry, Schemas, Error Taxonomy, Integration Framework.
- **Division 2 (Sruthi)**: Single-Image Remote-Sensing Intelligence (VQA, Captioning, Grounding across Optical, Multispectral, and SAR).
- **Division 3 (Dheeraj)**: Bi-Temporal Change Intelligence (Change Detection, Change VQA, Difference Maps).
- **Division 4 (Laksh)**: Optical-SAR Cross-Modal Intelligence (Joint Analysis, All-Weather Target Detection).
- **Division 5 (Manoj)**: Evidence Presentation, Web/GUI Interface, Benchmark Evaluation, Reports.
- **Tanvi**: Model study and Presentation support.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Lalith2007/SatQuery.git
cd SatQuery

# Switch to the feature branch
git checkout feature/lalith-agent

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Run the Test Suite
```bash
pytest -v
```

### 3. Launch the API Server & Interactive Presentation UI
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000/](http://localhost:8000/)** in your browser to access the live interactive judging dashboard, or **[http://localhost:8000/docs](http://localhost:8000/docs)** for the OpenAPI Swagger interface.

---

## 🎯 How to Demonstrate the Agentic Architecture (Judge Scenarios)

The live presentation interface at `http://localhost:8000/` contains **1-Click Judge Presets** that load real on-disk remote sensing rasters and execute the actual pipeline:

### 🛰️ Demo A: Single-Image Remote-Sensing VQA
- **Query**: `"What is the dominant land cover and infrastructure in this scene?"`
- **Input**: 1 Optical raster (`demo_optical_single.png`)
- **Observed Behavior**:
  - `IntentResolver` resolves task to `single_image_vqa`.
  - `TaskRouter` selects `single_image_vqa_mock`.
  - **Agent Decision Card** explains: *"Query resolved as visual question answering on single remote sensing image."*
  - **Output**: Detailed answer with 4 aircraft bounding boxes and 90% confidence.

### 🎯 Demo B: Spatial Feature Grounding & Localization
- **Query**: `"Where is the airport runway and apron located?"`
- **Input**: 1 Optical raster (`demo_airport_grounding.png`)
- **Observed Behavior**:
  - `IntentResolver` resolves task to `single_image_grounding`.
  - `TaskRouter` selects `single_image_grounding_mock`.
  - **Output**: Bounding box coordinates `[0.40, 0.10, 0.60, 0.90]` and attention heatmap evidence.

### ⏳ Demo C: Bi-Temporal Change Analysis
- **Query**: `"What changed between these two acquisition dates?"`
- **Input**: 2 Temporal acquisitions T0 (2021) and T1 (2023).
- **Observed Behavior**:
  - Validates temporal pair without enforcing artificial dimension constraints.
  - `TaskRouter` selects `bi_temporal_change_mock`.
  - **Output**: 22.4% urban growth detection, change bounding boxes, and a generated `change_map` artifact link.

### ⚡ Demo D: Optical-SAR Cross-Modal Joint Analysis
- **Query**: `"Use optical and SAR images together to identify structures beneath clouds."`
- **Input**: Optical RGB image + SAR C-band raster (`demo_sar_cross.tif`).
- **Observed Behavior**:
  - Validator recognizes heterogeneous modalities (`Optical` + `SAR`).
  - `TaskRouter` routes to `optical_sar_cross_modal_mock`.
  - **Output**: High-dielectric metallic structure detection through cloud occlusion with false-color composite artifact.

### 🔄 Demo E: Multi-Tool Sequential Workflow
- **Query**: `"What changed, where did it happen, and was the new region built-up?"`
- **Input**: 2 Temporal images.
- **Observed Behavior**:
  - `IntentResolver` identifies a multi-step composite query.
  - `WorkflowPlanner` creates a 2-step `TaskPlan`:
    - **Step 1**: `bi_temporal_change_mock` (detect & localize change clusters).
    - **Step 2**: `single_image_grounding_mock` (contextual characterization on detected clusters).
  - `ExecutionEngine` runs steps sequentially, passing context from Step 1 into Step 2.
  - `ResultAggregator` synthesizes composite answers and aggregates evidence from both tools.

### ⚡ Demonstrating Tool / Model Swapping
- Click the **"⚡ Swap Specialist Mock (Demo)"** button in the dashboard header.
- Swaps `single_image_vqa_mock` with `alternate_single_image_vqa_mock` (`v2.0.0-demo-mock`) in `ToolRegistry` via `POST /api/v1/tools/swap`.
- Re-executing Demo A immediately invokes the newly swapped specialist, proving the plug-and-play contract.

---

## 📡 API Endpoints

- `GET /`: Interactive web presentation dashboard.
- `GET /health`: Health status, system version, and registered specialist tool health.
- `GET /api/v1/tools`: List registered specialist tools with capability metadata.
- `GET /api/v1/tasks`: List supported remote-sensing vision-language tasks.
- `POST /api/v1/query`: Submit a structured JSON query with image references.
- `POST /api/v1/query/multipart`: Upload GeoTIFF/TIFF or PNG/JPEG rasters and execute query.
- `POST /api/v1/tools/swap`: Development/demo endpoint to swap approved specialist tools at runtime.
- `GET /api/v1/artifacts/{artifact_id}`: Retrieve generated masks, change maps, and visual artifacts.

---

## 📖 Developer Integration Guide

For detailed integration guides and contract specifications for Divisions 2, 3, 4, and 5, consult [INTEGRATION.md](file:///Users/lalith/Desktop/SatQuery/INTEGRATION.md).
