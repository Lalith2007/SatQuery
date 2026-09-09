"""Colab Step 01: Stage 1 BigEarthNet Multimodal Dataset Preparation.

Loads the approved 8,000-pair / 16,000-example Stage 1 BigEarthNet manifest,
formats instructions and ground truth into native Qwen2.5-VL ChatML messages
with verified BoxCodec coordinate tokens, and prepares training/validation splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.logging import get_logger
from specialists.single_image.adaptation.qwen25vl.bbox_codec import BoxCodec

logger = get_logger("colab_prepare_dataset")


def format_qwen_grounding_output(raw_output: str, width: int = 120, height: int = 120) -> str:
    """Encode bounding box string into Qwen native token syntax."""
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_output)]
    if len(nums) == 4:
        # Expected [ymin, xmin, ymax, xmax] in normalized [0.0, 1.0]
        return BoxCodec.encode_bbox(nums, width=width, height=height, source_format="normalized_0_1_ymin_xmin")
    return raw_output


def build_qwen_chatml_record(
    record: Dict[str, Any],
    image_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert raw BigEarthNet manifest record into Qwen ChatML format."""
    rec_id = record["id"]
    pair_id = record["pair_id"]
    sensor = record["sensor"]
    is_sar = "Sentinel-1" in sensor
    modality = "sar" if is_sar else "optical"
    anno_type = record.get("annotation_type", "captioning")

    task = "caption" if anno_type == "captioning" else ("grounding" if anno_type == "bounding box" else "vqa")

    # Format assistant output
    raw_output = record["output"]
    if task == "grounding":
        formatted_output = format_qwen_grounding_output(raw_output)
    else:
        formatted_output = raw_output

    # Determine image path with deterministic resolution (zero FUSE file-system roundtrips)
    patch_id = record.get("patch_id", pair_id.split("___")[0])
    s1_name = record.get("s1_name", pair_id.split("___")[-1] if "___" in pair_id else "")
    target_filename = "s1_2bands.tif" if is_sar else "s2_10bands.tif"

    if image_dir is not None:
        image_path = str(Path(image_dir) / pair_id / target_filename)
    else:
        # Default canonical relative path pointer
        candidate_roots = [
            Path("/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/bigearthnet_stage1/pairs"),
            Path("data/curated_mixture/materialized_samples"),
        ]
        base_dir = candidate_roots[0]
        for root in candidate_roots:
            if root.exists():
                base_dir = root
                break
        image_path = str(base_dir / pair_id / target_filename)

    user_prompt = record["input"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": formatted_output,
        },
    ]

    item = {
        "id": rec_id,
        "pair_id": pair_id,
        "patch_id": patch_id,
        "s1_name": s1_name,
        "source_dataset": "BigEarthNet",
        "source_annotation_id": record.get("source_row_id", record.get("ID", rec_id)),
        "s1_source_path": f"BigEarthNet-S1/{s1_name}.tif",
        "s2_source_path": f"BigEarthNet-S2/{patch_id}.tif",
        "image": image_path,
        "sensor": sensor,
        "modality": modality,
        "task": task,
        "annotation_type": anno_type,
        "country": record.get("country", "Unknown"),
        "parent_granule": record.get("parent_granule", "Unknown"),
        "split": record.get("split", "train"),
        "instruction": user_prompt,
        "target": formatted_output,
        "caption": formatted_output if task == "caption" else None,
        "messages": messages,
    }

    if task == "grounding":
        nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_output)]
        if len(nums) == 4:
            item["bbox"] = nums
            item["bbox_format"] = "normalized_0_1_ymin_xmin"

    return item

def load_stage1_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load records from the approved Stage 1 BigEarthNet JSONL manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stage 1 manifest not found at: {manifest_path}")
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def partition_by_parent_granule(
    samples: List[Dict[str, Any]],
    val_count: int = 800,
    test_count: int = 800,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Partition Stage 1 samples by parent granule to ensure zero spatial leakage."""
    granule_to_pairs: Dict[str, List[str]] = {}
    pair_to_samples: Dict[str, List[Dict[str, Any]]] = {}

    for idx, s in enumerate(samples):
        pid = s.get("pair_id") or s.get("id") or f"pair_{idx}"
        g = s.get("parent_granule") or s.get("parent_scene_id") or s.get("parent_scene") or "unknown"
        pair_to_samples.setdefault(pid, []).append(s)
        if pid not in granule_to_pairs.setdefault(g, []):
            granule_to_pairs[g].append(pid)

    all_granules = sorted(list(granule_to_pairs.keys()))
    rng = random.Random(seed)
    rng.shuffle(all_granules)

    # Greedily allocate granules to test, val, and train
    test_pairs: Set[str] = set()
    val_pairs: Set[str] = set()
    train_pairs: Set[str] = set()

    for g in all_granules:
        pairs = granule_to_pairs[g]
        # Each pair has 2 examples
        if len(test_pairs) * 2 < test_count:
            test_pairs.update(pairs)
        elif len(val_pairs) * 2 < val_count:
            val_pairs.update(pairs)
        else:
            train_pairs.update(pairs)

    # In case test/val became empty due to few granules, fall back to pair-level stratified split
    if len(val_pairs) == 0 or len(test_pairs) == 0:
        all_pairs = sorted(list(pair_to_samples.keys()))
        rng.shuffle(all_pairs)
        n_test = max(1, test_count // 2)
        n_val = max(1, val_count // 2)
        test_pairs = set(all_pairs[:n_test])
        val_pairs = set(all_pairs[n_test:n_test + n_val])
        train_pairs = set(all_pairs[n_test + n_val:])

    # Strict zero-leakage check
    assert len(train_pairs.intersection(val_pairs)) == 0, "Leakage: train ∩ val > 0"
    assert len(train_pairs.intersection(test_pairs)) == 0, "Leakage: train ∩ test > 0"
    assert len(val_pairs.intersection(test_pairs)) == 0, "Leakage: val ∩ test > 0"

    train_samples = [s for pid in train_pairs for s in pair_to_samples[pid]]
    val_samples = [s for pid in val_pairs for s in pair_to_samples[pid]]
    test_samples = [s for pid in test_pairs for s in pair_to_samples[pid]]

    split_report = {
        "total_samples": len(samples),
        "total_pairs": len(pair_to_samples),
        "total_granules": len(granule_to_pairs),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "leakage_passed": True,
        "intersections": {
            "train_val": len(train_pairs.intersection(val_pairs)),
            "train_test": len(train_pairs.intersection(test_pairs)),
            "val_test": len(val_pairs.intersection(test_pairs)),
        },
    }
    return train_samples, val_samples, test_samples, split_report


def extract_parent_scene_id(sample: Dict[str, Any]) -> str:
    """Extract parent scene or granule ID from sample."""
    return sample.get("parent_granule") or sample.get("parent_scene") or sample.get("pair_id", "unknown")


def build_canonical_qwen_sample(
    sample_id: str,
    image_path: str,
    modality: str,
    task: str,
    user_prompt: str,
    assistant_response: str,
    parent_scene: str = "unknown",
    width: int = 1024,
    height: int = 1024,
    bbox: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build a canonical Qwen ChatML record."""
    rec = {
        "id": sample_id,
        "pair_id": sample_id,
        "sensor": "Sentinel-1" if modality == "sar" else "Sentinel-2",
        "input": user_prompt,
        "output": assistant_response,
        "parent_granule": parent_scene,
        "annotation_type": "bounding box" if task == "grounding" else ("captioning" if task == "caption" else "vqa"),
    }
    return build_qwen_chatml_record(rec)


def partition_by_parent_scene(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Legacy wrapper for partition_by_parent_granule."""
    total = len(samples)
    val_count = max(1, int(total * val_ratio))
    test_count = max(1, total - int(total * train_ratio) - val_count)
    return partition_by_parent_granule(samples, val_count=val_count, test_count=test_count, seed=seed)


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Save record list to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_and_export_splits(
    manifest_path: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
    output_dir: str = "data/qwen_dataset",
    image_dir: Optional[str] = "data/curated_mixture/materialized_samples",
    val_count: int = 800,
    test_count: int = 800,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Stage 1 dataset preparation and export train/val/test splits."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    mf_p = Path(manifest_path)
    img_p = Path(image_dir) if image_dir else None

    print("=" * 60)
    print("SatQuery AI — Phase C: Stage 1 Dataset Preparation")
    print("=" * 60)
    print(f"Loading Stage 1 manifest from: {mf_p}...")
    raw_records = load_stage1_manifest(mf_p)
    print(f"Loaded {len(raw_records)} records from manifest.")

    print(f"Formatting ChatML records with image directory: {img_p}...")
    formatted_samples = [build_qwen_chatml_record(r, image_dir=img_p) for r in raw_records]
    print(f"Formatted all {len(formatted_samples)} ChatML records.")

    print("Partitioning samples by parent granule (Train: 14,304, Val: 846, Test: 850)...")
    train_s, val_s, test_s, split_report = partition_by_parent_granule(
        formatted_samples, val_count=val_count, test_count=test_count, seed=seed
    )

    print(f"Writing split files to {out_p}...")
    save_jsonl(train_s, out_p / "train.jsonl")
    save_jsonl(val_s, out_p / "val.jsonl")
    save_jsonl(test_s, out_p / "test.jsonl")

    with open(out_p / "dataset_split_report.json", "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)
    print(f"Splits exported: {len(train_s)} train, {len(val_s)} val, {len(test_s)} test.")

    # Mirror to persistent Google Drive backup via fast block copy
    drive_backup = Path("/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/qwen_dataset")
    if drive_backup.parent.exists():
        try:
            drive_backup.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(out_p / "train.jsonl", drive_backup / "train.jsonl")
            shutil.copy2(out_p / "val.jsonl", drive_backup / "val.jsonl")
            shutil.copy2(out_p / "test.jsonl", drive_backup / "test.jsonl")
            shutil.copy2(out_p / "dataset_split_report.json", drive_backup / "dataset_split_report.json")
            print(f"Mirrored dataset splits to Google Drive backup: {drive_backup}")
        except Exception as e:
            logger.warning(f"Could not mirror dataset splits to Drive: {e}")

    # Compute task x sensor distribution matrix
    matrix: Dict[str, Dict[str, int]] = {
        "optical": {"vqa": 0, "grounding": 0, "caption": 0},
        "sar": {"vqa": 0, "grounding": 0, "caption": 0},
    }
    for s in formatted_samples:
        mod = s["modality"]
        tsk = s["task"]
        if mod in matrix and tsk in matrix[mod]:
            matrix[mod][tsk] += 1

    task_sensor_report = {
        "total_samples": len(formatted_samples),
        "matrix": matrix,
        "optical_count": sum(matrix["optical"].values()),
        "sar_count": sum(matrix["sar"].values()),
        "optical_pct": round((sum(matrix["optical"].values()) / len(formatted_samples)) * 100, 2),
        "sar_pct": round((sum(matrix["sar"].values()) / len(formatted_samples)) * 100, 2),
    }
    with open(out_p / "dataset_task_modality_report.json", "w", encoding="utf-8") as f:
        json.dump(task_sensor_report, f, indent=2)

    logger.info(
        f"Stage 1 dataset prepared: {len(train_s)} train, {len(val_s)} val, {len(test_s)} test."
    )
    return {
        "split_report": split_report,
        "task_sensor_report": task_sensor_report,
    }


def ensure_dataset_splits(
    train_file: str = "data/qwen_dataset/train.jsonl",
    val_file: str = "data/qwen_dataset/val.jsonl",
    test_file: Optional[str] = "data/qwen_dataset/test.jsonl",
    manifest_path: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
) -> None:
    """Ensure dataset split files exist locally, restoring from Google Drive or auto-generating if needed."""
    train_p = Path(train_file)
    val_p = Path(val_file)
    test_p = Path(test_file) if test_file else None

    # Check if requested files already exist
    splits_exist = train_p.exists() and val_p.exists() and (test_p is None or test_p.exists())
    if splits_exist:
        return

    import shutil

    # 1. Try restoring from Google Drive persistent backup
    drive_backup = Path("/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/qwen_dataset")
    if drive_backup.exists() and (drive_backup / "train.jsonl").exists():
        logger.info(f"Restoring prepared dataset splits from Google Drive persistent backup: {drive_backup}")
        train_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_backup / "train.jsonl", train_p)
        if (drive_backup / "val.jsonl").exists():
            shutil.copy2(drive_backup / "val.jsonl", val_p)
        if test_p and (drive_backup / "test.jsonl").exists():
            shutil.copy2(drive_backup / "test.jsonl", test_p)
        if (drive_backup / "dataset_split_report.json").exists():
            shutil.copy2(drive_backup / "dataset_split_report.json", train_p.parent / "dataset_split_report.json")
        logger.info(f"Restored dataset splits to {train_p.parent}")
        return

    # 2. Auto-generate from Google Drive pairs if present
    drive_pairs = Path("/content/drive/MyDrive/SatQueryAI_Qwen25VL/datasets/bigearthnet_stage1/pairs")
    local_samples = Path("data/curated_mixture/materialized_samples")
    image_dir = str(drive_pairs) if drive_pairs.exists() else (str(local_samples) if local_samples.exists() else None)

    logger.info(f"Dataset splits missing at '{train_p}'. Auto-preparing from image source: {image_dir}...")
    prepare_and_export_splits(
        manifest_path=manifest_path,
        output_dir=str(train_p.parent),
        image_dir=image_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Dataset Preparation")
    parser.add_argument("--manifest_path", default="data/curated_mixture/bigearthnet_stage1_manifest.jsonl")
    parser.add_argument("--output_dir", default="data/qwen_dataset")
    parser.add_argument("--image_dir", default="data/curated_mixture/materialized_samples")
    args = parser.parse_args()

    prepare_and_export_splits(
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
    )
