# SatQuery AI — Division 5: Modular Benchmark Evaluation Suite (`evaluation/`)

**Owner**: Laksh  
**Module**: `evaluation/`  
**Supported Benchmarks**: `VRSBench`, `RSVQA`, `CDVQA`, `ISRO_SAC`

---

## 1. Overview & Architecture

Division 5 provides a unified, modular evaluation suite designed to evaluate model predictions against public remote-sensing benchmark datasets and generic hidden test sets.

```text
               Evaluation Runner CLI / Programmatic API
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
VRSBench Evaluator      RSVQA Evaluator          CDVQA Evaluator
(VQA + Grounding mIoU)  (LR/HR QA & Count)       (Change Detection & BLEU)
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ▼
                   ISRO / SAC Generic Evaluator
              (Cartosat-2S Optical + RISAT SAR Pairs)
                                │
                                ▼
                     Score Normalizer & Weights
                 (Scale-100, Invert-Error, Min-Max)
                                │
                                ▼
            Standardized BenchmarkEvaluationResult &
                   Markdown Scoreboard Output
```

---

## 2. Supported Benchmark Suites & Metrics

### A. VRSBench (`evaluation.benchmarks.vrsbench.VRSBenchEvaluator`)
- **Domain**: High-resolution optical Remote Sensing VQA & Text-Guided Visual Grounding.
- **Metrics**:
  - `vqa_accuracy`: Exact match string accuracy on normalized text.
  - `vqa_token_f1`: Token-level harmonic mean precision & recall for open-ended answers.
  - `grounding_miou`: Mean Intersection over Union ($IoU$) between predicted bounding boxes and ground truth boxes.
  - `grounding_p_at_05`: Precision @ $IoU \ge 0.50$.
  - `grounding_p_at_75`: Precision @ $IoU \ge 0.75$.
  - Category breakdowns: `presence_accuracy`, `count_accuracy`, `landcover_accuracy`.

### B. RSVQA (`evaluation.benchmarks.rsvqa.RSVQAEvaluator`)
- **Domain**: Multi-resolution VQA (Low Resolution Sentinel-2 & High Resolution Aerial).
- **Metrics**:
  - `overall_accuracy`: Global classification accuracy across all question types.
  - `presence_accuracy`: Binary object presence detection accuracy.
  - `comparison_accuracy`: Spatial comparison question accuracy.
  - `count_rmse`: Root Mean Squared Error on numerical counting queries.

### C. CDVQA (`evaluation.benchmarks.cdvqa.CDVQAEvaluator`)
- **Domain**: Bi-temporal Change Detection Visual Question Answering & Description.
- **Metrics**:
  - `binary_change_accuracy`: Accuracy on change existence queries (Yes/No).
  - `change_description_bleu1`: Unigram lexical precision for temporal narratives.
  - `change_description_bleu4`: 4-gram sentence-level BLEU score.
  - `change_description_rouge_l`: Longest Common Subsequence (LCS) structure recall.

### D. ISRO/SAC Generic Test Evaluator (`evaluation.benchmarks.isro_sac.ISROSACGenericEvaluator`)
- **Domain**: Multi-sensor test evaluation for Cartosat-2S (high-resolution optical) and RISAT (C-band SAR) co-registered pairs.
- **Strict Non-Fabrication Rule**: Built generically without hardcoded ground truths. Ingests whatever predictions and reference labels are supplied during evaluation runs.

---

## 3. Score Normalization Policy

Raw scores are computed and preserved without silent mutation.

The `ScoreNormalizer` (`evaluation.normalizer.ScoreNormalizer`) provides explicit normalization methods:
- `scale_100`: Converts `[0.0, 1.0]` scores to `[0.0, 100.0]`.
- `invert_error`: Converts error metrics (RMSE/MAE) into normalized accuracy-like scores via $\frac{100}{1 + \text{error}}$.
- `min_max`: Rescales arbitrary ranges to 0 - 100.

---

## 4. CLI Execution Commands

Another developer or judge can execute benchmark evaluations without reading source code:

```bash
# Run VRSBench Evaluation
python -m evaluation.runner \
  --benchmark vrsbench \
  --predictions data/vrsbench_predictions.json \
  --ground-truth data/vrsbench_ground_truth.json \
  --output-dir evaluation_output

# Run RSVQA Evaluation
python -m evaluation.runner \
  --benchmark rsvqa \
  --predictions data/rsvqa_predictions.json \
  --ground-truth data/rsvqa_ground_truth.json

# Run CDVQA Change Evaluation
python -m evaluation.runner \
  --benchmark cdvqa \
  --predictions data/cdvqa_predictions.json \
  --ground-truth data/cdvqa_ground_truth.json

# Run ISRO/SAC Generic Evaluation
python -m evaluation.runner \
  --benchmark isro_sac \
  --predictions data/isro_sac_predictions.json \
  --ground-truth data/isro_sac_ground_truth.json
```

---

## 5. REST API Integration

- `GET /api/v1/evaluation/benchmarks`: Lists available benchmarks and metrics.
- `POST /api/v1/evaluation/run`: Runs live evaluation on JSON payload:
  ```json
  {
    "benchmark": "vrsbench",
    "predictions": [{"answer": "Airport runway with aircraft.", "bbox": [0.1, 0.1, 0.5, 0.5]}],
    "ground_truths": [{"answer": "Airport runway with aircraft.", "bbox": [0.12, 0.08, 0.52, 0.51]}]
  }
  ```
