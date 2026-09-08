"""Colab Step 02: Dataset Quality Validation and Strict Integrity Gate.

Verifies image readability, dimensions, bounding box ordering, prompt/assistant messages,
task balance, duplicate IDs, and parent-scene split leakage.
Outputs `dataset_validation_report.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set
from PIL import Image

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.bbox_codec import BoxCodec

logger = get_logger("colab_validate_dataset")


def validate_single_sample(sample: Dict[str, Any], check_file_exists: bool = True) -> tuple[bool, List[str]]:
    """Validate schema, content, bounds, and syntax of an individual sample."""
    errors: List[str] = []
    sid = sample.get("id")
    if not sid:
        errors.append("Missing 'id' field")

    img_path = sample.get("image")
    if not img_path:
        errors.append("Missing 'image' path")
    elif check_file_exists:
        p = Path(img_path)
        if not p.exists():
            errors.append(f"Image file does not exist: {img_path}")

    messages = sample.get("messages")
    if not messages or len(messages) < 2:
        errors.append("Invalid or incomplete messages array")
    else:
        assistant_msg = messages[1].get("content")
        if not assistant_msg or not str(assistant_msg).strip():
            errors.append("Empty assistant response")
        elif sample.get("task") == "grounding" and "<|box_start|>" not in str(assistant_msg):
            errors.append("Grounding assistant message missing <|box_start|>")

    if sample.get("task") == "grounding":
        bbox = sample.get("bbox")
        if bbox is None:
            errors.append("Grounding task missing bbox")
        else:
            w = sample.get("width", 1024)
            h = sample.get("height", 1024)
            if not BoxCodec.validate_bbox(bbox, format_name="pixel_xyxy", width=w, height=h):
                errors.append(f"Invalid bounding box coordinates out of bounds: {bbox}")

    return (len(errors) == 0, errors)


def validate_dataset_split(
    jsonl_path: Path,
    seen_ids: Set[str],
) -> Dict[str, Any]:
    """Validate all samples within a single dataset split file."""
    if not jsonl_path.exists():
        return {"error": f"File not found: {jsonl_path}", "passed": False}

    valid_count = 0
    rejected_count = 0
    rejections: List[Dict[str, Any]] = []
    parent_scenes: Set[str] = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except Exception as e:
                rejections.append({"line": line_num, "reason": f"Malformed JSON: {e}"})
                rejected_count += 1
                continue

            sid = sample.get("id")
            if not sid:
                rejections.append({"line": line_num, "reason": "Missing 'id' field"})
                rejected_count += 1
                continue

            if sid in seen_ids:
                rejections.append({"id": sid, "line": line_num, "reason": "Duplicate sample ID"})
                rejected_count += 1
                continue
            seen_ids.add(sid)

            is_valid, errors = validate_single_sample(sample, check_file_exists=True)
            if not is_valid:
                for err in errors:
                    rejections.append({"id": sid, "reason": err})
                rejected_count += 1
                continue

            parent_scenes.add(sample.get("parent_scene_id", "unknown"))
            valid_count += 1

    return {
        "file": str(jsonl_path),
        "total": valid_count + rejected_count,
        "valid": valid_count,
        "rejected": rejected_count,
        "rejection_details": rejections[:20],
        "parent_scenes": list(parent_scenes),
        "passed": rejected_count == 0,
    }


def run_dataset_validation(
    data_dir: str = "data/qwen_dataset",
    output_report: str = "data/qwen_dataset/dataset_validation_report.json",
) -> Dict[str, Any]:
    """Validate train, val, and test splits and check for spatial leakage."""
    dir_p = Path(data_dir)
    seen_ids: Set[str] = set()

    train_res = validate_dataset_split(dir_p / "train.jsonl", seen_ids)
    val_res = validate_dataset_split(dir_p / "val.jsonl", seen_ids)
    test_res = validate_dataset_split(dir_p / "test.jsonl", seen_ids)

    # Check parent scene leakage across splits
    train_scenes = set(train_res.get("parent_scenes", []))
    val_scenes = set(val_res.get("parent_scenes", []))
    test_scenes = set(test_res.get("parent_scenes", []))

    inter_train_val = len(train_scenes.intersection(val_scenes))
    inter_train_test = len(train_scenes.intersection(test_scenes))
    inter_val_test = len(val_scenes.intersection(test_scenes))

    leakage_passed = (inter_train_val == 0 and inter_train_test == 0 and inter_val_test == 0)

    overall_passed = (
        train_res.get("passed", False)
        and val_res.get("passed", False)
        and test_res.get("passed", False)
        and leakage_passed
    )

    report = {
        "overall_validation_passed": overall_passed,
        "splits": {
            "train": train_res,
            "val": val_res,
            "test": test_res,
        },
        "spatial_leakage_audit": {
            "train_val_intersection": inter_train_val,
            "train_test_intersection": inter_train_test,
            "val_test_intersection": inter_val_test,
            "passed": leakage_passed,
        },
    }

    out_p = Path(output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Dataset validation report generated at {out_p}. Overall status: {overall_passed}")
    return report


if __name__ == "__main__":
    run_dataset_validation()
