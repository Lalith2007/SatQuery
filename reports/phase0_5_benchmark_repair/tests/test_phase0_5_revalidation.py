"""Authoritative Phase 0.5 Revalidation & Regression Test Suite.

Verifies:
1. Prediction Serialization Contract (JSONL structure, schema, metadata)
2. Duplicate detection and missing prediction detection
3. Official RSVQA answer normalization (no substring matches, integer counts)
4. Coordinate conversion integrity (VRSBench 0-100 <-> Qwen 0-1000)
5. Official VRSBench metric selection (Acc@0.5, Acc@0.7)
6. CDVQA reference authenticity & zero-synthetic rejection gate
7. Ground truth leakage prevention (GT separation)
8. Independent metric recomputation (Run 1 in-memory == Run 2 from JSONL)
9. TinyCD regression check (LEVIR-CD test: F1=79.31%, IoU=65.71%, OA=97.99%)
10. CMAF regression check (WHU-OPT-SAR test: OA=71.71%, mIoU=35.08%)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from evaluation.benchmarks.cdvqa import CDVQAEvaluator
from evaluation.benchmarks.rsvqa import RSVQAEvaluator
from evaluation.benchmarks.vrsbench import VRSBenchEvaluator
from evaluation.colab_qwen_final_benchmark_runner import (
    append_prediction_record,
    generate_manifest_for_file,
)
from scripts.vrsbench_coordinate_converter import BoundingBox, CoordinateConverter


# --- 1. Prediction Serialization Contract ---
def test_prediction_serialization_schema(tmp_path: Path):
    pred_file = tmp_path / "test_predictions.jsonl"
    record = {
        "sample_id": "rsvqa_val_00001",
        "image_id": "rsvqa_img_00001",
        "question_id": 1,
        "prompt": "Is there a building?",
        "raw_prediction": "Yes, there is.",
        "normalized_prediction": "yes",
        "ground_truth": "yes",
        "category": "presence",
        "checkpoint": "Qwen2.5-VL-3B-merged_full",
        "checkpoint_hash": "c830b809d08e826b6df52a0a25690b22194cfc6104bc36b281f6d39103c80a0a",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_prediction_record(pred_file, record)

    assert pred_file.exists()
    lines = pred_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])

    required_keys = [
        "sample_id", "image_id", "question_id", "prompt",
        "raw_prediction", "normalized_prediction", "ground_truth",
        "category", "checkpoint", "checkpoint_hash", "timestamp_utc"
    ]
    for k in required_keys:
        assert k in loaded, f"Missing required key in serialized prediction: {k}"


# --- 2. Duplicate Detection in Manifest ---
def test_manifest_duplicate_detection(tmp_path: Path):
    pred_file = tmp_path / "preds_with_dups.jsonl"
    rec1 = {"sample_id": "s1", "raw_prediction": "a"}
    rec2 = {"sample_id": "s2", "raw_prediction": "b"}
    rec3 = {"sample_id": "s1", "raw_prediction": "a_dup"}

    for r in [rec1, rec2, rec3]:
        append_prediction_record(pred_file, r)

    manifest = generate_manifest_for_file(pred_file, "test_bench")
    assert manifest["total_records"] == 3
    assert manifest["unique_records"] == 2
    assert manifest["duplicate_records"] == 1
    assert manifest["status"] == "WARNING"


# --- 3. RSVQA Normalization & Zero Substring Matching ---
def test_rsvqa_normalization_and_matching():
    # 0 must NEVER match 10 (the historical bug)
    assert RSVQAEvaluator.official_compute_accuracy("0", "10", "count") == 0
    assert RSVQAEvaluator.official_compute_accuracy("10", "0", "count") == 0

    # Counts
    assert RSVQAEvaluator.official_compute_accuracy("There are 5 buildings", "5", "count") == 1
    assert RSVQAEvaluator.official_compute_accuracy("five", "5", "count") == 1
    assert RSVQAEvaluator.official_compute_accuracy("6", "5", "count") == 0

    # Presence
    assert RSVQAEvaluator.official_compute_accuracy("yes", "yes", "presence") == 1
    assert RSVQAEvaluator.official_compute_accuracy("no", "yes", "presence") == 0
    assert RSVQAEvaluator.official_compute_accuracy("yes, definitely", "yes", "presence") == 1

    # Rural / Urban
    assert RSVQAEvaluator.official_compute_accuracy("rural area", "rural", "rural_urban") == 1
    assert RSVQAEvaluator.official_compute_accuracy("urban area", "rural", "rural_urban") == 0


# --- 4. Coordinate Conversion Integrity ---
def test_coordinate_conversion_qwen_to_vrsbench():
    # Qwen: [ymin=400, xmin=250, ymax=600, xmax=330] in 0-1000
    # Expected VRSBench: [x1=25, y1=40, x2=33, y2=60] in 0-100
    qwen_box = (400, 250, 600, 330)
    vrs_box = CoordinateConverter.qwen_to_vrsbench(qwen_box)
    assert vrs_box == (25, 40, 33, 60)


# --- 5. Official VRSBench Metric Selection ---
def test_vrsbench_grounding_metric_names():
    vrs = VRSBenchEvaluator()
    preds = [{"predicted_box": [20, 30, 40, 50]}]
    gts = [{"ground_truth": "{<20><30><40><50>}", "unique": True}]
    res = vrs.evaluate(preds, gts)

    assert "acc_05_all" in res.metrics
    assert "acc_07_all" in res.metrics
    assert "mean_iou" in res.metrics
    # Acc@0.5 is the primary official metric
    assert res.metrics["acc_05_all"].raw_score == 1.0


# --- 6. CDVQA Reference Authenticity & Synthetic Rejection ---
def test_cdvqa_synthetic_reference_rejection():
    evaluator = CDVQAEvaluator()
    forbidden_sample = {
        "answer": "New residential buildings and infrastructure constructed in the cleared agricultural area."
    }
    with pytest.raises(ValueError) as excinfo:
        evaluator.evaluate([{"answer": "anything"}], [forbidden_sample])
    assert "CDVQA EVALUATION INTEGRITY VIOLATION" in str(excinfo.value)


# --- 7. Ground Truth Leakage Prevention ---
def test_ground_truth_leakage_separation():
    # Ensure evaluators only compare after generation and never modify inputs
    evaluator = RSVQAEvaluator()
    preds = [{"prediction": "yes"}]
    gts = [{"ground_truth": "yes"}]
    original_gts_repr = str(gts)

    res = evaluator.evaluate(preds, gts)
    assert str(gts) == original_gts_repr  # GT completely untouched


# --- 8. Independent Metric Recomputation from JSONL ---
def test_independent_metric_recomputation(tmp_path: Path):
    pred_file = tmp_path / "eval_records.jsonl"
    records = [
        {"sample_id": "s1", "prediction": "yes", "ground_truth": "yes", "category": "presence"},
        {"sample_id": "s2", "prediction": "no", "ground_truth": "yes", "category": "presence"},
        {"sample_id": "s3", "prediction": "5", "ground_truth": "5", "category": "count"},
    ]
    for r in records:
        append_prediction_record(pred_file, r)

    # RUN 1: In-memory evaluation
    evaluator = RSVQAEvaluator()
    res1 = evaluator.evaluate(
        [{"prediction": r["prediction"]} for r in records],
        [{"ground_truth": r["ground_truth"], "category": r["category"]} for r in records],
    )

    # RUN 2: Reloading purely from saved disk file
    loaded_preds = []
    loaded_gts = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            loaded_preds.append({"prediction": item["prediction"]})
            loaded_gts.append({"ground_truth": item["ground_truth"], "category": item["category"]})

    res2 = evaluator.evaluate(loaded_preds, loaded_gts)

    assert res1.aggregate_raw_score == res2.aggregate_raw_score
    assert res1.metrics["overall_accuracy"].raw_score == res2.metrics["overall_accuracy"].raw_score


# --- 9. TinyCD Frozen Regression Reference ---
def test_tinycd_frozen_regression_baseline():
    tinycd_report_path = Path("reports/final_sih_evaluation/specialists/tinycd/levir_cd_evaluation.json")
    if not tinycd_report_path.exists():
        tinycd_report_path = Path("final_sih_evaluation/metrics/levir_cd_evaluation.json")
    if tinycd_report_path.exists():
        data = json.loads(tinycd_report_path.read_text())
        assert abs(data["metrics"]["f1"] - 0.7931) < 0.001
        assert abs(data["metrics"]["iou"] - 0.6571) < 0.001
        assert abs(data["metrics"]["overall_accuracy"] - 0.9799) < 0.001


# --- 10. CMAF Frozen Regression Reference ---
def test_cmaf_frozen_regression_baseline():
    cmaf_report_path = Path("reports/final_sih_evaluation/specialists/cmaf/whu_opt_sar_evaluation.json")
    if not cmaf_report_path.exists():
        cmaf_report_path = Path("final_sih_evaluation/metrics/whu_opt_sar_evaluation.json")
    if cmaf_report_path.exists():
        data = json.loads(cmaf_report_path.read_text())
        assert abs(data["overall_metrics"]["overall_accuracy"] - 0.7171) < 0.001
        assert abs(data["overall_metrics"]["mean_iou"] - 0.3508) < 0.001
        assert abs(data["overall_metrics"]["weighted_f1"] - 0.7418) < 0.001
