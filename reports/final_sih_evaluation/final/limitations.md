# SATQUERY AI — BENCHMARK EVALUATION BOUNDARIES & INTEGRITY NOTES

**Problem Statement:** SIH26167 — Agentic Multimodal AI for Spaceborne and Aerial Intelligence  
**Evaluation Scope:** Operational Boundaries, Resolution Limits & Model Constraints  
**Evaluation Date:** 2026-09-17  

---

## 1. Single-Image VLM Resolution & Grounding Dynamics
- **RSVQA-LR Spatial Resolution**: RSVQA-LR consists of Sentinel-2 low-resolution imagery (10m GSD). The VLM achieves 39.15% overall accuracy on these low-resolution rasters. Revalidation of complete reference answer mapping is pending.
- **VRSBench Fine-Grained Grounding**: The frozen Qwen2.5-VL-3B achieves a raw auxiliary Box IoU of 0.0484 on 500 VRSBench evaluation samples without task-specific fine-tuning. Official benchmark evaluation metrics (Acc@0.5, Acc@0.7) require the full official evaluator package.
- **VRSBench Captioning**: The evaluation run completed inference on 500 samples, but the final captioning score was not output to the execution log. Enforcing strict non-fabrication, this metric is marked `NOT AVAILABLE — METRIC NOT GENERATED`.

---

## 2. Frozen Specialist Scope & Decision Thresholds
- **TinyCD (LEVIR-CD)**: Evaluated at fixed threshold 0.50. Extremely robust global performance (F1: 79.31%, IoU: 65.71%, OA: 97.99%). Worst-case scene (`test_82`, F1: 27.46%) exhibits diffuse ground-disturbance boundaries rather than distinct building footprint construction.
- **CMAF (WHU-OPT-SAR)**: Dual-encoder attention fusion achieves 71.71% OA and 74.18% Weighted F1 across 4,950 tiles (308.6M valid pixels). Challenges persist in separating visually indistinguishable farmland vs village fringes without higher-order cadastral boundary vectors.
