# Model Selection Report — Division 3: Bi-Temporal Change Intelligence

**Author:** Dheeraj (Division 3)  
**Date:** August 2026  
**Status:** APPROVED — Model selections documented below

---

## 1. Problem Statement

Division 3 requires two pluggable backends:

1. **Change Detection Model** — Produces binary/probabilistic change maps from T0/T1 image pairs.
2. **Semantic Reasoner** — Interprets change detection outputs with respect to a user's natural-language query.

Both must be pluggable through the `ChangeModel` and `SemanticReasoner` interfaces defined in `specialists/temporal_change/interfaces.py`.

---

## 2. Candidate Change Detection Models

| Model | Params | Size | F1 (LEVIR-CD) | GPU VRAM | CPU Speed | License | Integration |
|-------|--------|------|---------------|----------|-----------|---------|-------------|
| **ChangeFormer** | ~41M | ~160 MB | 90.40% | 4–8 GB | ~250ms | Non-Commercial | Low-Medium |
| **BIT** | ~11.5M | ~45 MB | 89.31% | <2 GB | <100ms | Non-Commercial | Very Low |
| **TinyCD** | ~0.31M | ~1.2 MB | 91.31% | <1 GB | ~35ms | Non-Commercial | Very Low |
| **Open-CD** | Varies | Varies | 91.80% (ChangerEx) | 4–12 GB | Config-dependent | **Apache 2.0** | High |

### 2.1 ChangeFormer (wgcban/ChangeFormer)

- **Paper:** Bandara & Patel, IGARSS 2022 (arXiv:2201.01293)
- **Architecture:** Siamese hierarchical Transformer (SegFormer MiT) + MLP decoder
- **Checkpoints:** LEVIR-CD, DSIFN-CD, WHU-CD (~160 MB)
- **Input:** 2×(3×256×256) RGB, ImageNet-normalized
- **Output:** Binary change probability map (256×256)
- **Dependencies:** PyTorch, einops, timm, torchvision
- **Windows:** Pure PyTorch, no custom CUDA extensions — Windows-compatible
- **Assessment:** Strong baseline, well-documented. Moderate integration effort.

### 2.2 BIT (justchenhao/BIT_CD)

- **Paper:** Chen & Shi, IEEE TGRS 2022 (arXiv:2103.00208)
- **Architecture:** ResNet-18 + Spatial-Temporal Transformer
- **Checkpoints:** LEVIR-CD, WHU-CD, CDD, DSIFN-CD (~45 MB, 11.5M params)
- **Input/Output:** Same as ChangeFormer (256×256 RGB pairs → binary mask)
- **Efficiency:** ~8.5 GFLOPs, <2 GB VRAM, <100ms CPU
- **Windows:** Fully compatible (pure PyTorch)
- **Assessment:** Extremely clean, modular code. Easiest to integrate.

### 2.3 TinyCD (AndreaCodegoni/Tiny_model_4_CD)

- **Paper:** Codegoni et al., NCAA 2023 (arXiv:2207.13159)
- **Architecture:** Lightweight Siamese U-Net + MAMB attention
- **Checkpoints:** LEVIR-CD, WHU-CD (~1.2 MB, only 316K params)
- **Efficiency:** ~1.3 GFLOPs, <1 GB VRAM, ~35ms on CPU
- **F1 Score:** 91.31% on LEVIR-CD (highest among evaluated candidates)
- **Assessment:** Best accuracy/efficiency trade-off. Ultra-compact. Ideal for production.

### 2.4 Open-CD (likyoo/open-cd)

- **License:** Apache 2.0 (permissive)
- **Coverage:** 15+ methods (ChangeFormer, BIT, TinyCD, Changer, etc.)
- **Assessment:** Excellent for benchmarking but introduces heavy OpenMMLab dependency. Not suitable for minimal microservice integration.

---

## 3. Candidate Semantic Reasoning Approaches

| Approach | Type | Compute | Open-Ended? | Land-Cover Claims? |
|----------|------|---------|-------------|-------------------|
| **SpatialMetricSynthesizer** | Rule-based statistics | Negligible | No | No (honest) |
| **CDVQA** | Closed-vocabulary VQA | ~4-6 GB VRAM | No | Yes (limited vocabulary) |
| **Change-Agent** | LLM + MCI visual model | 16-24 GB VRAM | Yes | Yes (full) |
| **RSICCformer** | Change captioning | ~4-8 GB VRAM | Partial | Yes (sentence generation) |

### 3.1 SpatialMetricSynthesizer (Implemented — Default)

A controlled quantitative spatial statistics reasoner that:
- Honestly reports changed area percentage, cluster count, spatial distribution
- Explicitly declares limitations when queries require semantic classification
- Does NOT claim land-cover transitions from binary change detection alone
- Zero compute overhead, deterministic, fully testable

### 3.2 CDVQA (YZHJessica/CDVQA)

- **Paper:** Yuan et al., IEEE TGRS 2022 (arXiv:2112.06343)
- **Architecture:** Dual ResNet + CEM + BiLSTM/BERT fusion
- **Dataset:** SECOND benchmark, 122K+ QA pairs
- **Question types:** Change presence, change type, object counting, comparison
- **Answer vocabulary:** Closed (few hundred tokens)
- **Assessment:** Good for structured QA but rigid vocabulary. Best as optional plugin.

### 3.3 Change-Agent / RS-VLMs

- **Compute:** Requires 16-24 GB VRAM (or 8-12 GB quantized)
- **Assessment:** Full open-ended reasoning but impractical for standard deployment. Future integration target.

---

## 4. Selected Configuration

### Primary Change Detector: **Siamese Feature-Difference Network**

For the initial implementation, we use a lightweight Siamese feature-difference model (`ChangeFormerAdapter`) that:
- Uses ResNet-18 feature extraction + 1×1 conv classifier
- Runs on CPU in <200ms per pair
- Accepts pretrained ChangeFormer/TinyCD/BIT checkpoints when available
- Falls back to untrained feature extraction when no checkpoint is provided

**Upgrade path:** Drop in TinyCD weights (~1.2 MB) for production accuracy.

### Primary Semantic Reasoner: **SpatialMetricSynthesizer**

Selected for scientific honesty:
- Does not claim semantic capabilities that binary change detection cannot provide
- Produces quantitative spatial statistics (area %, region count, bounding boxes)
- Explicitly declares limitations for land-cover transition queries
- Zero additional dependencies

**Upgrade path:** Add CDVQA adapter for structured change VQA when weights are available.

---

## 5. Integration Architecture

```
ChangeModel (ABC)
├── MockChangeModel          # Testing (deterministic pixel-diff)
├── ChangeFormerAdapter      # Production (Siamese feature-diff)
│   ├── accepts TinyCD weights
│   ├── accepts ChangeFormer weights
│   └── accepts BIT weights
└── [Future: TinyCDAdapter]

SemanticReasoner (ABC)
├── MockSemanticReasoner     # Testing (labeled mock output)
├── SpatialMetricSynthesizer # Production (quantitative statistics)
└── [Future: CDVQAReasoner]
```

Both interfaces are fully pluggable through dependency injection in `BiTemporalChangeSpecialistTool.__init__()`.

---

## 6. Scientific Honesty Declaration

1. Binary change detection identifies **spatial change presence/extent only**. It cannot determine specific land-cover transitions.
2. The `SpatialMetricSynthesizer` explicitly reports this limitation when queries ask about vegetation, built-up, or water changes.
3. Confidence scores are kept separate: `change_confidence` and `semantic_confidence` are never combined into a fabricated `overall_confidence`.
4. The `overall_confidence` field is set to `None` until a defensible calibration method is implemented.
