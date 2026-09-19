---
title: SatQuery AI — Agentic Multimodal Satellite Intelligence
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SatQuery AI — Free GPU Hosting (ZeroGPU)

SatQuery AI is an agentic vision-language system engineered for remote-sensing earth observation, bi-temporal change detection, and multi-sensor (Optical + SAR) cross-modal fusion.

## Verified Scientific Baseline (`baseline-2026-09-18`)

| Division | Capability | Architecture | Checkpoint | Parameters / Size |
| :--- | :--- | :--- | :--- | :--- |
| **Division 2** | VQA, Grounding, Captioning | Qwen2.5-VL-3B-Instruct | `merged_full` (standalone) | **3,754,622,976 params** (6.99 GB) |
| **Division 3** | Bi-Temporal Change Detection | TinyCD | `ChangeDetector-TinyCD.pth` | **3,565,034 params** (14.35 MB) |
| **Division 4** | Optical-SAR Cross-Modal Fusion | CMAF | `cmaf_landcover_best.pth` | **439.8 MB** |

## System Architecture

```
Browser
   ↓
React Frontend (Vite + TypeScript + Tailwind)
   ↓
FastAPI Backend Gateway
   ↓
AgentController
   ↓
TaskRouter
   ↓
SpecialistRegistry
   ├── Qwen2.5-VL Specialist (Single-Image VQA, Referring Expression Grounding, Captioning)
   ├── TinyCD Specialist (Bi-Temporal Change Localization & 3-Image Change VQA)
   └── CMAF Specialist (Optical + SAR Cross-Attention Fusion Neck)
   ↓
ResultAggregator
   ↓
Interactive Evidence / Visual Overlays / Formal Reports
```

## Running Locally

```bash
# 1. Install dependencies
uv sync # or pip install -e .

# 2. Build React frontend
cd frontend
npm install
npm run build
cd ..

# 3. Launch deployment server
python -m deployment.server
```

Open http://localhost:7860 to access the production application.
