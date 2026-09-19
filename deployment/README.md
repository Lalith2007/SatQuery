---
title: SatQuery AI — Agentic Multimodal Satellite Intelligence
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# SatQuery AI — Hugging Face Spaces Free GPU Hosting (ZeroGPU)

SatQuery AI is an agentic vision-language system engineered for remote-sensing earth observation, bi-temporal change detection, and multi-sensor (Optical + SAR) cross-modal fusion.

## Verified Scientific Baseline (`baseline-2026-09-18`)

| Division | Capability | Architecture | Checkpoint / Revision | Parameters / Size |
| :--- | :--- | :--- | :--- | :--- |
| **Division 2** | VQA, Grounding, Captioning | Qwen2.5-VL-3B-Instruct | `Lalith2007/SatQuery-Models` (`stage1-baseline`) | **3,754,622,976 params** (6.99 GB) |
| **Division 3** | Bi-Temporal Change Detection | TinyCD | `ChangeDetector-TinyCD.pth` (frozen) | **3,565,034 params** (14.35 MB) |
| **Division 4** | Optical-SAR Cross-Modal Fusion | CMAF | `cmaf_landcover_best.pth` (frozen) | **439.8 MB** |

## ZeroGPU Production Architecture

- **ZeroGPU SDK:** Built on Hugging Face Spaces **Gradio SDK** in Server mode (`gradio.Server`), running `@spaces.GPU(duration=60)` queued inference.
- **Hardware Allocation:** Dynamic multi-tenant allocation providing up to **48 GB VRAM** dynamically per queued inference step.
- **Quota Model:** Free accounts receive **5 GPU minutes/day** of dynamic execution time.
- **Decoupled Model Hub:** The 7 GB vision-language model weights reside in the dedicated model repository `Lalith2007/SatQuery-Models` under immutable revision `stage1-baseline`. The Space repository remains lightweight without binary weights in Git history.
- **Custom React Frontend:** Serves the existing React + TypeScript + Vite Single-Page Application (`frontend/dist/`) at `/` with static assets at `/assets`, backed by Gradio Server and FastAPI routes.
- **Frozen Baseline Integrity:** Specialists TinyCD and CMAF remain strictly frozen with verified SHA-256 hashes.

```
Browser (React Frontend SPA at /)
   ↓
Gradio Server (gradio.Server + @spaces.GPU queued API)
   ↓
FastAPI Routes (/health, /version, /api/v1/...) + Gradio Queue (/gradio_api/...)
   ↓
AgentController & TaskRouter
   ├── Division 2: Qwen2.5-VL Specialist (Single-Image VQA, Grounding, Caption, 3-Image Change VQA)
   ├── Division 3: TinyCD Specialist (Bi-Temporal Change Localization, FROZEN)
   └── Division 4: CMAF Specialist (Optical + SAR Cross-Attention Fusion Neck, FROZEN)
```

## Branch & Release Model

- **Production Branch:** `main` (auto-synced to `Lalith2007/SatQuery-Space`)
- **Deployment Branch:** `deployment/zerogpu-baseline`
- **Baseline Freeze Tag:** `baseline-2026-09-18`
- **Model Revision:** `stage1-baseline`

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build React frontend
cd frontend && npm install && npm run build && cd ..

# 3. Launch Gradio production server
python app.py
```

Access http://localhost:7860 to interact with the production application.

