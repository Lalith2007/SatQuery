"""Deterministic Materialization of at least 25 Representative BigEarthNet S1/S2 Pairs.

Uses HTTP Range requests to extract real Sentinel-1 and Sentinel-2 GeoTIFFs
matching the Stage-1 manifest.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ZIP_URL = "https://huggingface.co/datasets/ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2/resolve/main/BigEarthNet_14K.zip"


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get("Content-Length", 0))
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + size - 1}"}
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def materialize_25_pairs(
    manifest_path: str = "data/curated_mixture/bigearthnet_stage1_manifest.jsonl",
    output_dir: str = "data/curated_mixture/materialized_samples",
    target_pair_count: int = 27,
):
    print("Opening remote zip archive on Hugging Face with urllib...")
    t0 = time.perf_counter()
    rz = RemoteZipReader(ZIP_URL)
    zf = zipfile.ZipFile(rz)
    zip_names = set(zf.namelist())
    print(f"Zip directory loaded in {time.perf_counter() - t0:.2f}s ({len(zip_names)} files).")

    # Load manifest grouped by pair
    manifest_by_pair = {}
    with open(manifest_path, "r", encoding="utf-8") as mf:
        for line in mf:
            rec = json.loads(line)
            pid = rec["pair_id"]
            manifest_by_pair.setdefault(pid, []).append(rec)

    # Filter pairs available in zip archive
    available_pairs = []
    for pid, recs in manifest_by_pair.items():
        patch_id = recs[0]["patch_id"]
        s1_name = recs[0]["s1_name"]
        s2_path = f"BEN_14k/BigEarthNet-S2/train/{patch_id}.tif"
        s1_path = f"BEN_14k/BigEarthNet-S1/train/{s1_name}.tif"
        if s2_path in zip_names and s1_path in zip_names:
            available_pairs.append({
                "pair_id": pid,
                "patch_id": patch_id,
                "s1_name": s1_name,
                "country": recs[0]["country"],
                "granule": recs[0]["parent_granule"],
                "records": recs,
                "tasks": [r["annotation_type"] for r in recs],
                "s2_zip_path": s2_path,
                "s1_zip_path": s1_path,
            })

    print(f"Discovered {len(available_pairs)} candidate pairs in both manifest and zip.")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    existing_dirs = {p.name for p in out_p.iterdir() if p.is_dir() and (p / "s2_10bands.tif").exists() and (p / "s1_2bands.tif").exists()}
    print(f"Found {len(existing_dirs)} already materialized pairs on disk.")

    selected_pairs = [p for p in available_pairs if p["pair_id"] in existing_dirs]
    for p in available_pairs:
        if p["pair_id"] not in existing_dirs:
            selected_pairs.append(p)
        if len(selected_pairs) >= target_pair_count:
            break

    print(f"Targeting {len(selected_pairs)} pairs for validation suite...")

    for i, pair_info in enumerate(selected_pairs, 1):
        pid = pair_info["pair_id"]
        p_dir = out_p / pid
        s2_out = p_dir / "s2_10bands.tif"
        s1_out = p_dir / "s1_2bands.tif"
        meta_out = p_dir / "metadata.json"

        if s2_out.exists() and s1_out.exists() and meta_out.exists():
            continue

        p_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(selected_pairs)}] Downloading {pid}...")

        s2_bytes = zf.read(pair_info["s2_zip_path"])
        with open(s2_out, "wb") as f:
            f.write(s2_bytes)

        s1_bytes = zf.read(pair_info["s1_zip_path"])
        with open(s1_out, "wb") as f:
            f.write(s1_bytes)

        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(pair_info["records"], f, indent=2)

    total_ready = len([p for p in out_p.iterdir() if p.is_dir() and (p / "s2_10bands.tif").exists()])
    print(f"Materialization complete! Total valid pairs ready: {total_ready}")


if __name__ == "__main__":
    materialize_25_pairs(target_pair_count=27)
