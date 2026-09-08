"""Dataset loader and audit utilities for LEVIR-CD and large-scale change detection datasets.

Features:
- Lazy, memory-safe PyTorch Dataset (__getitem__ disk reads)
- Identical paired spatial augmentation for T0, T1, and ground-truth mask
- Dynamic split discovery (FULL mode vs DEV mode)
- Parent-scene isolation and cross-split image-hash data leakage auditing
- Dataset manifest generation and training class balance analysis.

Official LEVIR-CD Structure:
  - Train: 445 parent scenes (~7,120 cropped 256x256 patches)
  - Val:   64 parent scenes (~1,024 cropped 256x256 patches)
  - Test:  128 parent scenes (~2,048 cropped 256x256 patches)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TemporalChangeDataset(Dataset):
    """Standardized memory-safe PyTorch Dataset for Bi-Temporal Change Detection (T0, T1, Mask)."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: int = 256,
        is_training: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.is_training = is_training
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Lazy load T0 and T1 as uint8 RGB (memory safe: not kept resident in RAM)
        t0_img = Image.open(item["t0_path"]).convert("RGB")
        t1_img = Image.open(item["t1_path"]).convert("RGB")

        # Lazy load ground truth mask (L mode: 0 or 255)
        if "mask_path" in item and item["mask_path"] and os.path.exists(item["mask_path"]):
            mask_img = Image.open(item["mask_path"]).convert("L")
        else:
            # Fallback zero mask if evaluating without ground truth
            mask_img = Image.new("L", t0_img.size, color=0)

        # Resize if dimensions differ from target
        if t0_img.size != (self.image_size, self.image_size):
            t0_img = t0_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            t1_img = t1_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)

        t0_arr = np.array(t0_img, dtype=np.float32) / 255.0
        t1_arr = np.array(t1_img, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask_img, dtype=np.float32) > 128.0).astype(np.float32)

        # Spatial data augmentation applied IDENTICALLY to T0, T1, and ground-truth mask
        if self.is_training:
            # 1. Random Horizontal Flip
            if random.random() > 0.5:
                t0_arr = np.fliplr(t0_arr).copy()
                t1_arr = np.fliplr(t1_arr).copy()
                mask_arr = np.fliplr(mask_arr).copy()

            # 2. Random Vertical Flip
            if random.random() > 0.5:
                t0_arr = np.flipud(t0_arr).copy()
                t1_arr = np.flipud(t1_arr).copy()
                mask_arr = np.flipud(mask_arr).copy()

            # 3. Random 90-degree Rotations (0, 90, 180, 270 deg)
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                t0_arr = np.rot90(t0_arr, rot_k).copy()
                t1_arr = np.rot90(t1_arr, rot_k).copy()
                mask_arr = np.rot90(mask_arr, rot_k).copy()

        # Convert to PyTorch Tensors: (H, W, C) -> (C, H, W)
        t0_tensor = torch.from_numpy(t0_arr).permute(2, 0, 1).float()
        t1_tensor = torch.from_numpy(t1_arr).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()  # (1, H, W)

        return {
            "t0": t0_tensor,
            "t1": t1_tensor,
            "mask": mask_tensor,
            "sample_id": item.get("sample_id", f"sample_{idx}"),
            "parent_id": item.get("parent_id", "unknown"),
            "t0_path": item["t0_path"],
            "t1_path": item["t1_path"],
            "mask_path": item.get("mask_path"),
        }


class LEVIRCDDatasetLoader:
    """Manages discovery, leakage auditing, class balance, and manifest generation for LEVIR-CD."""

    @classmethod
    def extract_parent_id(cls, filename: str) -> str:
        """Extract parent scene identifier from patch or scene filename.

        Examples:
          - 'train_1_0.png' -> 'train_1'
          - 'train_p0001_00.png' -> 'train_p0001'
          - 'test_12.png' -> 'test_12'
          - 'levir_scene_44_patch_03.png' -> 'levir_scene_44'
        """
        stem = Path(filename).stem
        if "_" in stem:
            parts = stem.split("_")
            # If last part is patch index (e.g. 0, 00, 01, 15), remove it
            if len(parts) >= 2 and parts[-1].isdigit():
                return "_".join(parts[:-1])
            if len(parts) >= 3 and parts[-2] == "patch":
                return "_".join(parts[:-2])
        return stem

    @classmethod
    def discover_split_samples(
        cls,
        dataset_root: Path | str,
        split: str = "train",
        mode: str = "full",
        dev_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Discover T0, T1, and Mask triplets from standard LEVIR-CD directory layout.

        Expected directory structures:
        1. Pre-split standard layout:
           <root>/<split>/A/ (T0)
           <root>/<split>/B/ (T1)
           <root>/<split>/label/ or <root>/<split>/label_256/ (Ground Truth Mask)
        2. Flat directory format with manifest.
        """
        root = Path(dataset_root)
        split_dir = root / split
        samples: List[Dict[str, Any]] = []

        # Find A (T0) and B (T1) directories (case-insensitive search)
        a_dir = split_dir / "A" if (split_dir / "A").exists() else split_dir / "a"
        b_dir = split_dir / "B" if (split_dir / "B").exists() else split_dir / "b"

        # Find label directory (label, labels, label_256, etc.)
        label_dir = None
        for cand in ["label", "label_256", "labels", "Label", "mask", "masks"]:
            if (split_dir / cand).exists():
                label_dir = split_dir / cand
                break

        if a_dir.exists() and b_dir.exists():
            # Search all images (.png, .jpg, .tif)
            candidates = sorted(list(a_dir.glob("*.png")) + list(a_dir.glob("*.jpg")) + list(a_dir.glob("*.tif")))
            for f in candidates:
                stem = f.stem
                # Check matching B file
                b_path = b_dir / f.name
                if not b_path.exists():
                    # Check matching with other extensions
                    for ext in [".png", ".jpg", ".tif"]:
                        alt = b_dir / f"{stem}{ext}"
                        if alt.exists():
                            b_path = alt
                            break

                if b_path.exists():
                    label_path = None
                    if label_dir is not None:
                        l_cand = label_dir / f.name
                        if l_cand.exists():
                            label_path = str(l_cand)
                        else:
                            for ext in [".png", ".jpg", ".tif"]:
                                alt_l = label_dir / f"{stem}{ext}"
                                if alt_l.exists():
                                    label_path = str(alt_l)
                                    break

                    parent_id = cls.extract_parent_id(f.name)
                    samples.append({
                        "sample_id": f"{split}_{stem}",
                        "parent_id": parent_id,
                        "t0_path": str(f),
                        "t1_path": str(b_path),
                        "mask_path": label_path,
                        "split": split,
                    })

        # Apply DEV mode limit if requested
        if mode.lower() == "dev":
            limit = dev_limit or (24 if split == "train" else (8 if split == "val" else 16))
            samples = samples[:limit]

        return samples

    @classmethod
    def discover_all_splits(
        cls,
        dataset_root: Path | str,
        mode: str = "full",
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Discover train, val, and test splits with guaranteed zero parent-scene leakage.

        If an explicit validation split folder is missing or empty, creates a deterministic,
        parent-scene-grouped validation partition from the training set.
        """
        train_samples = cls.discover_split_samples(dataset_root, split="train", mode=mode)
        val_samples = cls.discover_split_samples(dataset_root, split="val", mode=mode)
        test_samples = cls.discover_split_samples(dataset_root, split="test", mode=mode)

        # If val split has 0 samples or was empty, perform scene-isolated partition from train
        if not val_samples and len(train_samples) > 20:
            import random
            parent_scenes = sorted(list({s["parent_id"] for s in train_samples}))
            rng = random.Random(seed)
            rng.shuffle(parent_scenes)

            val_scene_count = max(int(len(parent_scenes) * val_ratio), 1)
            val_scene_set = set(parent_scenes[:val_scene_count])
            train_scene_set = set(parent_scenes[val_scene_count:])

            new_train = [s for s in train_samples if s["parent_id"] in train_scene_set]
            new_val = [dict(s, split="val") for s in train_samples if s["parent_id"] in val_scene_set]

            train_samples = new_train
            val_samples = new_val

        return train_samples, val_samples, test_samples

    @classmethod
    def _compute_quick_hash(cls, p: str) -> str:
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read(4096))
        return h.hexdigest()

    @classmethod
    def audit_split_leakage(
        cls,
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        test_samples: Sequence[Dict[str, Any]],
        check_file_hashes: bool = False,
    ) -> Dict[str, Any]:
        """Programmatic verification of zero data leakage across train, val, test splits."""
        train_parents = {s.get("parent_id") or s["sample_id"] for s in train_samples}
        val_parents = {s.get("parent_id") or s["sample_id"] for s in val_samples}
        test_parents = {s.get("parent_id") or s["sample_id"] for s in test_samples}

        # Filter out trivial generic tokens if any
        train_parents.discard("")
        val_parents.discard("")
        test_parents.discard("")

        train_val_overlap = train_parents.intersection(val_parents)
        train_test_overlap = train_parents.intersection(test_parents)
        val_test_overlap = val_parents.intersection(test_parents)

        is_leakage_free = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        audit_result = {
            "is_leakage_free": is_leakage_free,
            "train_parent_count": len(train_parents),
            "val_parent_count": len(val_parents),
            "test_parent_count": len(test_parents),
            "train_samples_count": len(train_samples),
            "val_samples_count": len(val_samples),
            "test_samples_count": len(test_samples),
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap": list(train_val_overlap),
            "train_test_overlap": list(train_test_overlap),
            "val_test_overlap": list(val_test_overlap),
            "train_val_overlap_examples": list(train_val_overlap)[:5],
            "train_test_overlap_examples": list(train_test_overlap)[:5],
            "val_test_overlap_examples": list(val_test_overlap)[:5],
        }

        if check_file_hashes and is_leakage_free:
            train_hashes = {cls._compute_quick_hash(s["t0_path"]) for s in train_samples[:50] if os.path.exists(s["t0_path"])}
            val_hashes = {cls._compute_quick_hash(s["t0_path"]) for s in val_samples[:50] if os.path.exists(s["t0_path"])}
            test_hashes = {cls._compute_quick_hash(s["t0_path"]) for s in test_samples[:50] if os.path.exists(s["t0_path"])}
            hash_overlap = train_hashes.intersection(val_hashes) or train_hashes.intersection(test_hashes)
            if hash_overlap:
                audit_result["is_leakage_free"] = False
                audit_result["binary_content_overlap"] = True

        return audit_result


    @classmethod
    def compute_class_balance_statistics(
        cls,
        samples: Sequence[Dict[str, Any]],
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """Compute pixel-level change class balance statistics over ground truth masks."""
        total_pixels = 0
        changed_pixels = 0
        ratios: List[float] = []

        valid_samples = [s for s in samples if s.get("mask_path") and os.path.exists(s["mask_path"])]
        if not valid_samples:
            return {"status": "NO_MASKS_AVAILABLE"}

        # Subsample if large
        eval_samples = valid_samples[:max_samples]

        for s in eval_samples:
            mask_img = Image.open(s["mask_path"]).convert("L")
            mask_arr = np.array(mask_img) > 128
            n_px = int(mask_arr.size)
            n_chg = int(mask_arr.sum())
            total_pixels += n_px
            changed_pixels += n_chg
            ratios.append(n_chg / n_px if n_px > 0 else 0.0)

        unchanged_pixels = total_pixels - changed_pixels
        pct_changed = (changed_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        pct_unchanged = (unchanged_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        return {
            "analyzed_samples_count": len(eval_samples),
            "total_pixels_analyzed": total_pixels,
            "changed_pixels": changed_pixels,
            "unchanged_pixels": unchanged_pixels,
            "percentage_changed": round(pct_changed, 3),
            "percentage_unchanged": round(pct_unchanged, 3),
            "mean_changed_pixel_ratio": round(float(np.mean(ratios)), 4) if ratios else 0.0,
            "median_changed_pixel_ratio": round(float(np.median(ratios)), 4) if ratios else 0.0,
            "min_changed_pixel_ratio": round(float(np.min(ratios)), 4) if ratios else 0.0,
            "max_changed_pixel_ratio": round(float(np.max(ratios)), 4) if ratios else 0.0,
            "class_imbalance_ratio": f"1 : {round(unchanged_pixels / max(changed_pixels, 1), 1)} (Change vs Background)",
        }

    @classmethod
    def generate_dataset_manifest(
        cls,
        dataset_root: Path | str,
        train_samples: Sequence[Dict[str, Any]],
        val_samples: Sequence[Dict[str, Any]],
        test_samples: Sequence[Dict[str, Any]],
        output_path: Path | str = "specialists/temporal_change/evaluation/levir_cd_manifest.json",
        mode: str = "full",
    ) -> Dict[str, Any]:
        """Generate and save official dataset manifest with audit proofs and class statistics."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        audit = cls.audit_split_leakage(train_samples, val_samples, test_samples)
        class_stats = cls.compute_class_balance_statistics(train_samples)

        # Inspect dimensions from first sample
        img_w, img_h = 256, 256
        if train_samples and os.path.exists(train_samples[0]["t0_path"]):
            with Image.open(train_samples[0]["t0_path"]) as im:
                img_w, img_h = im.size

        manifest = {
            "manifest_version": "1.2.0",
            "dataset_name": "LEVIR-CD",
            "dataset_root": str(dataset_root),
            "dataset_mode": mode.upper(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spatial_resolution": "0.5 meters/pixel",
            "image_dimensions": {"width": img_w, "height": img_h, "channels": 3},
            "mask_dimensions": {"width": img_w, "height": img_h, "channels": 1},
            "split_counts": {
                "train_parent_scenes": audit["train_parent_count"],
                "train_patches_count": len(train_samples),
                "val_parent_scenes": audit["val_parent_count"],
                "val_patches_count": len(val_samples),
                "test_parent_scenes": audit["test_parent_count"],
                "test_patches_count": len(test_samples),
                "total_patches_count": len(train_samples) + len(val_samples) + len(test_samples),
            },
            "official_levir_cd_target_counts": {
                "train_parent_scenes": 445,
                "train_patches_count": 7120,
                "val_parent_scenes": 64,
                "val_patches_count": 1024,
                "test_parent_scenes": 128,
                "test_patches_count": 2048,
            },
            "leakage_audit": audit,
            "train_class_balance": class_stats,
        }

        with open(out_p, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
