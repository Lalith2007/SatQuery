"""Comprehensive Verification Audit for Stage-1 BigEarthNet.txt Shard.

Verifies:
1. Unique image pairs == 8,000
2. Total examples == 16,000
3. Annotation density == exactly 2.00 annotations per pair
4. Duplicate count == 0
5. Split integrity == 100% TRAIN, 0% validation/test/bench
6. Sensor distribution == 8,000 S1 SAR / 8,000 S2 Optical
7. Task distribution == 6,636 captions / 5,228 VQA / 4,136 grounding
8. Granules == 115
9. Countries == 8
10. Grounding semantics == coarse_patch_level_referring_expression
11. Provenance completeness across all records
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_stage1_shard(
    manifest_path: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
) -> Dict[str, Any]:
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    file_size_bytes = p.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    manifest_sha = compute_sha256(p)

    seen_record_ids = set()
    pair_to_records = {}
    granules = set()
    countries = set()
    splits = Counter()
    sensors = Counter()
    task_types = Counter()
    grounding_levels = Counter()
    missing_provenance_count = 0

    total_records = 0

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            total_records += 1
            rec = json.loads(line_str)
            rec_id = rec.get("id")

            # Check duplicate record IDs
            if rec_id in seen_record_ids:
                raise ValueError(f"Duplicate record ID found: {rec_id} at line {line_num}")
            seen_record_ids.add(rec_id)

            # Check pair
            pair_id = rec.get("pair_id")
            if not pair_id:
                raise ValueError(f"Missing pair_id at line {line_num}")
            pair_to_records.setdefault(pair_id, []).append(rec)

            # Granule & Country
            granule = rec.get("parent_granule")
            if granule:
                granules.add(granule)
            else:
                missing_provenance_count += 1

            country = rec.get("country")
            if country:
                countries.add(country)
            else:
                missing_provenance_count += 1

            # Split
            split = rec.get("split")
            splits[split] += 1

            # Sensor
            sensor = rec.get("sensor")
            sensors[sensor] += 1

            # Task Type
            anno_type = rec.get("annotation_type")
            if anno_type == "captioning":
                task_types["captioning"] += 1
            elif anno_type == "bounding box":
                task_types["grounding"] += 1
                g_level = rec.get("grounding_semantic_level")
                grounding_levels[g_level] += 1
            else:
                task_types["vqa"] += 1

            # Provenance completeness check
            for req_field in ["source_row_id", "patch_id", "s1_name", "input", "output", "channels_rendered"]:
                if req_field not in rec or rec[req_field] is None:
                    missing_provenance_count += 1

    unique_pairs = len(pair_to_records)
    density_values = [len(v) for v in pair_to_records.values()]
    min_density = min(density_values)
    max_density = max(density_values)
    avg_density = round(total_records / unique_pairs, 2)

    # Audits & Assertions
    passed_pairs = (unique_pairs == 8000)
    passed_examples = (total_records == 16000)
    passed_density = (min_density == 2 and max_density == 2 and avg_density == 2.00)
    passed_duplicates = (len(seen_record_ids) == total_records)
    passed_split = (splits["train"] == 16000 and splits["validation"] == 0 and splits["test"] == 0 and splits["bench"] == 0)
    passed_sensors = (sensors["Sentinel-1 SAR"] == 8000 and sensors["Sentinel-2 MSI"] == 8000)
    passed_tasks = (task_types["captioning"] == 6636 and task_types["vqa"] == 5228 and task_types["grounding"] == 4136)
    passed_granules = (len(granules) == 115)
    passed_countries = (len(countries) == 8)
    passed_grounding_semantic = (grounding_levels["coarse_patch_level_referring_expression"] == 4136)
    passed_provenance = (missing_provenance_count == 0)

    all_passed = (
        passed_pairs and passed_examples and passed_density and
        passed_duplicates and passed_split and passed_sensors and
        passed_tasks and passed_granules and passed_countries and
        passed_grounding_semantic and passed_provenance
    )

    result = {
        "status": "PASS" if all_passed else "FAIL",
        "manifest_path": str(p),
        "manifest_sha256": manifest_sha,
        "storage_footprint_bytes": file_size_bytes,
        "storage_footprint_mb": file_size_mb,
        "metrics": {
            "unique_pairs": unique_pairs,
            "total_examples": total_records,
            "pair_annotation_density": avg_density,
            "min_annotations_per_pair": min_density,
            "max_annotations_per_pair": max_density,
            "duplicate_records_count": total_records - len(seen_record_ids),
            "granules_count": len(granules),
            "countries_count": len(countries),
            "countries_list": sorted(list(countries)),
            "sensors": dict(sensors),
            "tasks": dict(task_types),
            "splits": dict(splits),
            "train_source_pct": round((splits["train"] / total_records) * 100, 2),
            "eval_partitions_used": splits["validation"] + splits["test"] + splits["bench"],
            "grounding_semantic_level_counts": dict(grounding_levels),
            "missing_provenance_fields_count": missing_provenance_count,
        },
        "assertions": {
            "unique_pairs_is_8000": passed_pairs,
            "total_examples_is_16000": passed_examples,
            "density_is_exactly_2": passed_density,
            "zero_duplicates": passed_duplicates,
            "source_is_100pct_train": passed_split,
            "sensors_are_8000_each": passed_sensors,
            "tasks_match_locked_distribution": passed_tasks,
            "granules_count_is_115": passed_granules,
            "countries_count_is_8": passed_countries,
            "grounding_semantics_explicitly_coarse": passed_grounding_semantic,
            "provenance_complete": passed_provenance,
        },
    }

    print("=" * 60)
    print("STAGE 1 SHARD VERIFICATION AUDIT")
    print("=" * 60)
    print(f"Overall Status:       {'PASS' if all_passed else 'FAIL'}")
    print(f"Manifest File:        {manifest_path}")
    print(f"Manifest SHA-256:     {manifest_sha}")
    print(f"Storage Footprint:    {file_size_mb} MB ({file_size_bytes:,} bytes)")
    print("-" * 60)
    print(f"UNIQUE PAIRS:         {unique_pairs} (Expected: 8000)")
    print(f"EXAMPLES:             {total_records} (Expected: 16000)")
    print(f"ANNOTATION DENSITY:   {avg_density} (Expected: exactly 2.0)")
    print(f"DUPLICATES:           {result['metrics']['duplicate_records_count']} (Expected: 0)")
    print(f"TRAIN SOURCE:         {result['metrics']['train_source_pct']}% (Expected: 100%)")
    print(f"VAL/TEST/BENCH USAGE: {result['metrics']['eval_partitions_used']} (Expected: 0)")
    print(f"SAR / OPTICAL:        {sensors['Sentinel-1 SAR']} / {sensors['Sentinel-2 MSI']} (Expected: 8000 / 8000)")
    print(f"TASK COUNTS:          Captions: {task_types['captioning']} | VQA: {task_types['vqa']} | Grounding: {task_types['grounding']}")
    print(f"GRANULES:             {len(granules)} (Expected: 115)")
    print(f"COUNTRIES:            {len(countries)}: {sorted(list(countries))}")
    print("=" * 60)

    # Save verification JSON
    with open("bigearthnet_stage1_verification_audit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    verify_stage1_shard()
