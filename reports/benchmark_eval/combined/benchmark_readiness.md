# SatQuery AI — Master Benchmark Readiness & Provenance Summary

**Audit Scope:** Public VLM Benchmarks (RSVQA, VRSBench, CDVQA) & Private Mission Target (ISRO/SAC)  
**Date:** 2026-09-11 16:14:46 UTC  
**Policy:** **STRICT NON-FABRICATION COMPLIANCE**  

---

## Master Authoritative Scoreboard

| Benchmark | Task | Model | Split | Sample Count | Actual Inference | Score | Status |
|:---|:---|:---|:---|:---:|:---:|:---|:---:|
| **LEVIR-CD** | Bi-Temporal Change Detection | TinyCD (`ChangeDetector-TinyCD.pth`) | Official held-out test | 128 scenes ($8.39\text{M}$ px) | **YES** | **F1: 79.31%, IoU: 65.71%, OA: 97.99%** | **COMPLETE** |
| **WHU-OPT-SAR** | Cross-Modal Land-Cover Classification | CMAF (`cmaf_landcover_best.pth`) | Official held-out test | 4,950 tiles (15 scenes) | **YES** | **OA: 71.71%, mIoU: 35.08%, Macro F1: 46.62%, Weighted F1: 74.18%** | **COMPLETE** |
| **RSVQA-LR** | Remote Sensing VQA | Qwen2.5-VL-3B-Instruct (Vision-Language Adapter) | Validation split | 2,000 records (100 images) | **NO** (Pending checkpoint merge) | **NOT AVAILABLE** | **PENDING QWEN TRAINING MERGE** |
| **VRSBench** | Remote Sensing VQA + Visual Grounding | Qwen2.5-VL-3B-Instruct (Vision-Language Adapter) | Test representative partition | 18 records (smoke: 1, eval: 18) | **NO** (Pending checkpoint merge) | **NOT AVAILABLE** | **PENDING QWEN TRAINING MERGE** |
| **CDVQA** | Change Detection Visual Question Answering | TinyCD + Qwen2.5-VL-3B-Instruct | Official LEVIR-CD test pairs | 128 scenes | Pipeline verified / VLM inference pending | **NOT AVAILABLE** | **CDVQA: PIPELINE VERIFIED — QWEN SEMANTIC EVALUATION PENDING** |
| **ISRO/SAC** | Joint Optical-SAR Multi-Sensor Intelligence | SatQuery AI Multi-Specialist Architecture | Restricted private evaluation | 0 bytes (Awaiting official data) | **NO** | **NOT AVAILABLE** | **AWAITING OFFICIAL EVALUATION DATA** |

---

## Instructions for Post-Training Evaluation

Once the Qwen2.5-VL fine-tuning job on Google Colab concludes:
1. Export the merged LoRA checkpoint directory (or HuggingFace adapter format).
2. Execute the post-training evaluation hook:
   ```bash
   ./.venv/bin/python evaluation/post_training_qwen_eval.py --checkpoint-dir /path/to/merged_qwen_weights
   ```
