# SatQuery AI

**An Agentic Vision-Language Assistant for Remote-Sensing Imagery**

---

## 🛰️ Architecture Overview

SatQuery AI is designed around a decoupled, modular architecture. Division 1 provides the core backend, orchestration engine, canonical data contracts, and tool registry that seamlessly coordinates specialist intelligence models.

```
USER / FRONTEND (Division 5 - Manoj)
       │
       ▼
 FastAPI BACKEND (Division 1 - Lalith)
       │
       ▼
 AGENT CONTROLLER (Division 1 - Lalith)
 ┌─────┴────────────────────────────┐
 │                                  │
 ▼                                  ▼
Geospatial Input Validator    Intent Resolver & Deterministic Router
       │                                  │
       └──────────────────┬───────────────┘
                          ▼
                  EXECUTION ENGINE
                          │
                  TOOL REGISTRY
 ┌────────────────────────┼────────────────────────┐
 │                        │                        │
 ▼                        ▼                        ▼
Single-Image (Div 2)   Bi-Temporal (Div 3)     Optical-SAR (Div 4)
- VQA                  - Change Analysis       - Cross-Modal Fusion
- Caption              - Change VQA            - All-Weather Detection
- Grounding            - Change Maps
 └────────────────────────┬────────────────────────┘
                          ▼
                  RESULT AGGREGATOR
                          │
                          ▼
              Standardized QueryResponse
          (Answer + Evidence + Trace + Artifacts)
```

---

## 👥 Team & Division Structure

- **Division 1 (Lalith)**: Agent Core, Backend, Orchestration, Validation, Registry, Schemas, Error Taxonomy, Integration Framework.
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

### 3. Launch the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for the interactive Swagger API documentation.

---

## 📡 API Endpoints

- `GET /health`: Health status, system version, and specialist tool health checks.
- `GET /api/v1/tools`: List all registered specialist tools and capability metadata.
- `GET /api/v1/tasks`: List supported remote-sensing vision-language tasks.
- `POST /api/v1/query`: Submit a structured JSON query with image references.
- `POST /api/v1/query/multipart`: Upload GeoTIFF/TIFF or PNG/JPEG rasters and execute query.
- `GET /api/v1/artifacts/{artifact_id}`: Retrieve generated masks, change maps, and visualization artifacts.

---

## 📖 Developer Integration Guide

For detailed instructions on how to implement specialist tools and integrate with the registry, consult [INTEGRATION.md](file:///Users/lalith/Desktop/SatQuery/INTEGRATION.md).
