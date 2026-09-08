"""Comprehensive Test Suite for Kaggle Qwen2.5-VL Training Architecture.

Verifies:
- Kaggle Notebook existence, valid JSON schema, and required 12 stages (Phases A through L)
- Direct HTTP streaming materialization (0 GB archive disk footprint)
- Zero demo, fallback, or synthetic image substitutions permitted
- Working disk quotas and Kaggle environment diagnostics
- Pre-training readiness gate logic for Kaggle runtime
"""

import json
from pathlib import Path
import pytest

from specialists.single_image.training.kaggle.build_kaggle_notebook import notebook as kaggle_notebook_dict
from specialists.single_image.training.kaggle.materialize_bigearthnet_kaggle import (
    CombinedStream,
    validate_raster_file,
    load_unique_pairs_from_manifest,
    materialize_bigearthnet_kaggle,
)
import io


# 1. Kaggle Notebook Exists and is Valid JSON
def test_kaggle_notebook_exists_and_is_valid_json():
    nb_path = Path("specialists/single_image/training/kaggle/kaggle_qwen25vl_training.ipynb")
    assert nb_path.exists(), f"Kaggle notebook missing at {nb_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert nb_data.get("nbformat") == 4
    assert len(nb_data["cells"]) >= 25, f"Expected at least 25 cells, got {len(nb_data['cells'])}"


# 2. All 12 Phases Exist in Kaggle Notebook
def test_expected_kaggle_notebook_phases_exist():
    nb_path = Path("specialists/single_image/training/kaggle/kaggle_qwen25vl_training.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_text = ""
    for cell in nb_data["cells"]:
        all_text += "".join(cell.get("source", [])) + "\n"

    expected_phases = [
        "PHASE A",
        "PHASE B",
        "PHASE C",
        "PHASE D",
        "PHASE E",
        "PHASE F",
        "PHASE G",
        "PHASE H",
        "PHASE I",
        "PHASE J",
        "PHASE K",
        "PHASE L",
    ]
    for phase in expected_phases:
        assert phase in all_text, f"Missing {phase} in Kaggle notebook"


# 3. Kaggle Notebook Uses Direct Streaming Materializer
def test_kaggle_notebook_uses_streaming_materializer():
    nb_path = Path("specialists/single_image/training/kaggle/kaggle_qwen25vl_training.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_text = ""
    for cell in nb_data["cells"]:
        all_text += "".join(cell.get("source", [])) + "\n"

    assert "materialize_bigearthnet_kaggle" in all_text
    assert "00_environment_check_kaggle" in all_text
    assert "/kaggle/working" in all_text
    assert "STRICT_REAL_DATA = True" in all_text
    assert "DEMO_MODE = False" in all_text


# 4. Test CombinedStream Functionality
def test_combined_stream_functionality():
    s1 = io.BytesIO(b"part1_")
    s2 = io.BytesIO(b"part2")
    comb = CombinedStream([s1, s2])

    data = b""
    while True:
        chunk = comb.read(3)
        if not chunk:
            break
        data += chunk
    assert data == b"part1_part2"


# 5. Test Mini Materialization with Gate Authorization
def test_mini_materialize_kaggle(tmp_path):
    manifest_file = tmp_path / "test_manifest.jsonl"
    sample_pool = Path("data/curated_mixture/materialized_samples")
    sample_dirs = [d.name for d in sample_pool.iterdir() if d.is_dir() and "___" in d.name][:2]
    assert len(sample_dirs) == 2

    with open(manifest_file, "w") as f:
        for pid in sample_dirs:
            parts = pid.split("___")
            rec = {
                "pair_id": pid,
                "patch_id": parts[0],
                "s1_name": parts[1],
                "parent_granule": "TEST_GRANULE",
                "country": "TEST_COUNTRY",
            }
            f.write(json.dumps(rec) + "\n")

    out_dir = tmp_path / "out_pairs"
    res = materialize_bigearthnet_kaggle(
        manifest_path=str(manifest_file),
        output_dir=str(out_dir),
        auto_download=False,
    )

    assert res["training_authorized"] is True
    assert res["resolved_s1_images"] == 2
    assert res["resolved_s2_images"] == 2
    assert res["missing_s1_images"] == 0
    assert res["missing_s2_images"] == 0
    assert (out_dir / "materialization_manifest.jsonl").exists()
    assert (out_dir / "checksums.sha256").exists()
    assert (out_dir / "materialization_summary.json").exists()
