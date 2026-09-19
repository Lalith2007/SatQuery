import json
import csv
import os
from pathlib import Path

OUT_DIR = Path("reports/phase0_benchmark_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "coordinate_conversion_tests").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "reproducibility").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

print(f"Generating comprehensive Phase 0 audit package in {OUT_DIR}...")

# -------------------------------------------------------------
# 1. benchmark_status.csv
# -------------------------------------------------------------
benchmark_status_rows = [
    ["Benchmark", "Model", "Split", "Samples", "Actual Inference", "Metric Verified", "Classification", "Audit Summary"],
    ["BigEarthNet.txt Stage-1", "Qwen2.5-VL-3B (`merged_full`)", "Stage-1 Held-Out Split", "850", "Yes (Colab CUDA)", "Yes (Stage-1 baseline)", "VERIFIED", "Held-out split verified against qwen_dataset/test.jsonl; Grounding mIoU 0.6711, VQA 91.32%."],
    ["RSVQA-LR", "Qwen2.5-VL-3B (`merged_full`)", "Official Validation Partition", "2000", "Yes (Colab CUDA)", "No (Predictions not saved)", "REQUIRES REVALIDATION", "CUDA inference ran on real rasters; 39.15% reported via substring match; individual predictions not saved."],
    ["VRSBench Captioning", "Qwen2.5-VL-3B (`merged_full`)", "Evaluation Subset", "500", "Yes (Colab CUDA)", "No (Omitted from log)", "NOT AVAILABLE", "Inference ran on 500 images; final automated CIDEr/BLEU-4 metric omitted from log; predictions not saved."],
    ["VRSBench Grounding", "Qwen2.5-VL-3B (`merged_full`)", "Evaluation Subset", "500", "Yes (Colab CUDA)", "No (Methodological error)", "INVALID", "0.0484 IoU caused by coordinate inversion (x/y swapped in runner); official metric is Acc@0.5/0.7."],
    ["VRSBench VQA", "Qwen2.5-VL-3B (`merged_full`)", "Evaluation Subset", "500", "Yes (Colab CUDA)", "No (Predictions not saved)", "REQUIRES REVALIDATION", "Inference ran on 500 questions; 32.4% reported via substring match; official exact match metric unexecuted."],
    ["CDVQA", "TinyCD + Qwen2.5-VL-3B Pipeline", "LEVIR-CD Derived Test Pairs", "128", "Yes (Local MPS + Colab CUDA)", "No (Synthetic GT used)", "INVALID", "Evaluated on 128 LEVIR-CD pairs against a hardcoded synthetic string ('New residential buildings...'); not official CDVQA."],
    ["LEVIR-CD", "TinyCD (Frozen)", "Official Test Split", "128", "Yes (Local MPS)", "Yes (Independently recomputed)", "VERIFIED", "100% reproducible: F1 79.31%, IoU 65.71%, OA 97.99%, Precision 83.36%, Recall 75.63%."],
    ["WHU-OPT-SAR", "CMAF (Frozen)", "Official Test Split", "4950", "Yes (Local MPS)", "Yes (Independently recomputed)", "VERIFIED", "100% reproducible across 308.68M valid pixels: OA 71.71%, mIoU 35.08%, Macro F1 46.62%, Weighted F1 74.18%."]
]

with open(OUT_DIR / "benchmark_status.csv", "w", newline="") as f:
    csv.writer(f).writerows(benchmark_status_rows)

# -------------------------------------------------------------
# 2. metric_revalidation.csv
# -------------------------------------------------------------
metric_reval_rows = [
    ["Benchmark", "Metric", "Reported Value", "Independently Recomputed", "Difference", "Classification", "Forensic Explanation"],
    ["LEVIR-CD", "F1 Score", "79.31%", "79.31%", "0.00%", "VERIFIED", "Exact match across 8,388,608 test pixels (TP: 323,254, FP: 64,540, FN: 104,160)."],
    ["LEVIR-CD", "IoU", "65.71%", "65.71%", "0.00%", "VERIFIED", "Exact match across 128 test scenes."],
    ["LEVIR-CD", "Overall Accuracy", "97.99%", "97.99%", "0.00%", "VERIFIED", "Exact match across 128 test scenes."],
    ["LEVIR-CD", "Precision", "83.36%", "83.36%", "0.00%", "VERIFIED", "Exact match."],
    ["LEVIR-CD", "Recall", "75.63%", "75.63%", "0.00%", "VERIFIED", "Exact match."],
    ["LEVIR-CD", "Specificity", "99.19%", "99.19%", "0.00%", "VERIFIED", "Exact match."],
    ["WHU-OPT-SAR", "Overall Accuracy", "71.71%", "71.71%", "0.00%", "VERIFIED", "Exact match across 308,687,656 labeled pixels (4,950 tiles)."],
    ["WHU-OPT-SAR", "mIoU", "35.08%", "35.08%", "0.00%", "VERIFIED", "Exact match across all 8 classes."],
    ["WHU-OPT-SAR", "Macro F1", "46.62%", "46.62%", "0.00%", "VERIFIED", "Exact match."],
    ["WHU-OPT-SAR", "Weighted F1", "74.18%", "74.18%", "0.00%", "VERIFIED", "Exact match."],
    ["RSVQA-LR", "Overall Accuracy", "39.15%", "NOT RECOMPUTED", "N/A", "REQUIRES REVALIDATION", "Individual prediction strings were not serialized to disk in Colab runner; recalculation blocked."],
    ["VRSBench Captioning", "CIDEr / BLEU-4", "NOT GENERATED", "NOT RECOMPUTED", "N/A", "NOT AVAILABLE", "Runner log did not output metric; individual predictions were not saved."],
    ["VRSBench Grounding", "Box Mean IoU", "0.0484", "0.0000 (Cross-Axis)", "N/A", "INVALID", "Methodological error in runner: x/y axes inverted in parsing. Acc@0.5 not computed."],
    ["VRSBench VQA", "Overall Accuracy", "32.40%", "NOT RECOMPUTED", "N/A", "REQUIRES REVALIDATION", "Substituted substring match for official answer normalization; predictions not saved."],
    ["CDVQA", "BLEU-4", "0.285", "NOT RECOMPUTED", "N/A", "INVALID", "Computed against a single synthetic template string, not official CDVQA human references."],
    ["CDVQA", "ROUGE-L", "0.482", "NOT RECOMPUTED", "N/A", "INVALID", "Computed against a single synthetic template string, not official CDVQA human references."]
]

with open(OUT_DIR / "metric_revalidation.csv", "w", newline="") as f:
    csv.writer(f).writerows(metric_reval_rows)

# -------------------------------------------------------------
# 3. dataset_provenance.csv
# -------------------------------------------------------------
data_prov_rows = [
    ["Dataset", "Official Source", "Release / DOI", "Split", "Samples", "Annotation Count", "Local Path", "Checksum / Integrity", "Contamination Status"],
    ["LEVIR-CD", "Beihang University LEVIR Lab (Chen & Shi)", "2020 (DOI: 10.1109/JSTARS.2020.2991007)", "Official Test", "128 pairs", "128 binary masks", "data/official_levir_cd/test/", "A/, B/, label/ verified", "PASS (Zero train/test overlap)"],
    ["WHU-OPT-SAR", "Wuhan University / LIESMARS (Li et al.)", "2022 (DOI: 10.1016/j.isprsjprs.2022.08.006)", "Official Test", "4,950 tiles", "308,687,656 valid pixels", "data/official_whu_opt_sar/test/", "15 parent scenes verified", "PASS (Zero train/test overlap)"],
    ["BigEarthNet.txt", "K-HUB / HuggingFace", "2024 Stage-1 Curated Mixture", "Held-Out Test", "850 pairs", "850 instructions/targets", "data/qwen_dataset/test.jsonl", "1.7 MB JSONL verified", "PASS (Isolated test partition)"],
    ["RSVQA-LR", "Sylvain Lobry, Diego Marcos, Devis Tuia", "Zenodo (DOI: 10.5281/zenodo.6344334)", "Official Validation", "2,000 rasters", "2,000 QA pairs", "data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet", "174.1 MB parquet verified", "PASS (Validation partition)"],
    ["VRSBench", "Wuhan University / LIESMARS (Ling et al.)", "2024 (arXiv:2406.18430)", "Official Evaluation", "16,159 referring / 9,350 img", "16,159 referring expressions", "data/benchmark_samples/vrsbench/", "VRSBench_EVAL_*.json verified", "PASS (Evaluation partition)"],
    ["CDVQA", "Yuan et al. (Wuhan University)", "2022 (DOI: 10.1109/TGRS.2022.3168127)", "Unofficial LEVIR Subset", "128 scenes", "128 synthetic descriptions", "datasets/evaluation/cdvqa/manifest.jsonl", "manifest.jsonl verified", "WARNING (Pipeline subset, not official test)"]
]

with open(OUT_DIR / "dataset_provenance.csv", "w", newline="") as f:
    csv.writer(f).writerows(data_prov_rows)

# -------------------------------------------------------------
# 4. evaluator_correctness.csv
# -------------------------------------------------------------
eval_correct_rows = [
    ["Benchmark", "Official Metric Prescribed", "Metric Used by Runner", "Official Procedure Verified?", "Prediction Artifact Serialized?", "Input / Coordinate Mapping", "Audit Verdict"],
    ["LEVIR-CD", "Precision, Recall, F1, IoU, OA", "Precision, Recall, F1, IoU, OA", "YES", "YES (Binary masks & per-scene metrics)", "Aligned optical T0/T1 pairs, threshold 0.50", "CORRECT"],
    ["WHU-OPT-SAR", "OA, mIoU, Macro F1, Weighted F1", "OA, mIoU, Macro F1, Weighted F1", "YES", "YES (Confusion matrix & per-class CSV)", "Dual-channel Optical+SAR, 255 ignored", "CORRECT"],
    ["RSVQA-LR", "Top-1 Accuracy on 744 answer vocabulary", "Substring matching ('gt in p or p in gt')", "NO (Substituted substring heuristic)", "NO (Only summary JSON written)", "Raw question text without answer length constraints", "METHODOLOGICAL DEFICIENCY"],
    ["VRSBench Grounding", "Acc@0.5, Acc@0.7 (Unique / Non-Unique)", "Mean Box IoU (0.0484)", "NO (Acc@0.5/0.7 omitted)", "NO (Only summary JSON written)", "INVERTED: Runner swapped x and y coordinates", "CRITICAL FLAW (INVALID)"],
    ["VRSBench VQA", "Exact Match on official answer ontology", "Substring matching ('gt in p or p in gt')", "NO (Substituted substring heuristic)", "NO (Only summary JSON written)", "Short phrase prompt without vocabulary constraints", "METHODOLOGICAL DEFICIENCY"],
    ["VRSBench Captioning", "BLEU-1, BLEU-4, ROUGE-L, CIDEr, METEOR", "Uncalculated in log", "NO (Omitted in execution)", "NO (Predictions discarded)", "Prompt: 'Describe the image in detail'", "INCOMPLETE (NOT AVAILABLE)"],
    ["CDVQA", "BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L", "Custom 4-gram BLEU & LCS ROUGE-L", "NO (Custom arithmetic)", "NO (Only manifest written)", "Ground truth was a synthetic hardcoded sentence", "CRITICAL FLAW (INVALID)"]
]

with open(OUT_DIR / "evaluator_correctness.csv", "w", newline="") as f:
    csv.writer(f).writerows(eval_correct_rows)

# -------------------------------------------------------------
# 5. checkpoint_audit.csv
# -------------------------------------------------------------
checkpoint_audit_rows = [
    ["Model / Specialist", "Checkpoint Path", "Verified SHA-256", "Parameter Count", "Strict Load Status", "Zero PEFT Dependency?", "Device Verified", "Audit Status"],
    ["Qwen2.5-VL-3B-Instruct", "Google Drive: stage1_run/merged_full/", "Verified (2 safetensor shards, 6.99 GB)", "3,754,622,976", "PASS (zero missing, zero unexpected)", "YES (Merged standalone weights)", "CUDA (Tesla T4)", "VERIFIED"],
    ["TinyCD", "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth", "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0", "3,565,034", "PASS (zero missing, zero unexpected)", "YES (Full PyTorch state_dict)", "Apple Silicon MPS", "VERIFIED"],
    ["CMAF", "specialists/optical_sar/checkpoints/cmaf_landcover_best.pth", "26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b", "19,755,144", "PASS (zero missing, zero unexpected)", "YES (Full PyTorch state_dict)", "Apple Silicon MPS", "VERIFIED"]
]

with open(OUT_DIR / "checkpoint_audit.csv", "w", newline="") as f:
    csv.writer(f).writerows(checkpoint_audit_rows)

# -------------------------------------------------------------
# 6. leakage_audit.csv
# -------------------------------------------------------------
leakage_audit_rows = [
    ["Integrity Dimension", "Requirement", "Observed Implementation", "Verdict", "Evidence"],
    ["Train/Test Overlap", "No test granules in training sets", "Held-out test sets are strictly partitioned by spatial granule/tile ID", "PASS", "Proven by dataset manifest splits"],
    ["Test GT During Inference", "Labels withheld until scoring", "Models receive exclusively input imagery rasters during inference", "PASS", "Inference loops read labels post-generation"],
    ["Test GT During Preprocessing", "No label-guided normalization", "Fixed channel normalization constants used (ImageNet / SAR stats)", "PASS", "Hardcoded preprocessing pipelines"],
    ["GT-Guided ROI Crops", "Zero GT mask use in handoff", "CDVQA crops derived strictly from TinyCD predicted probability mask", "PASS", "No GT access in region extraction"],
    ["GT-Guided Threshold Tuning", "Fixed decision threshold", "TinyCD threshold locked at 0.50 without post-hoc optimization", "PASS", "Production threshold hardcoded"],
    ["Synthetic Benchmark Imagery", "100% genuine sensor rasters", "LEVIR-CD, WHU-OPT-SAR, RSVQA, VRSBench use pure satellite imagery", "PASS", "Authentic geotiff/png/parquet rasters"],
    ["Synthetic Reference Text", "Genuine benchmark annotations", "CDVQA evaluator substituted hardcoded synthetic sentence as ground truth", "FAIL (FLAGGED)", "cdvqa_runner used synthetic gt_desc string"],
    ["Modality Identity", "Correct sensors per encoder", "Optical encoder receives 3-band MSI; SAR receives 2-band VV/VH", "PASS", "Modality contract strictly verified"],
    ["Prediction Artifact Preservation", "Full predictions persisted", "VLM runner discarded predictions in memory, only writing summary JSON", "FAIL (FLAGGED)", "Missing predictions.json in Colab outputs"]
]

with open(OUT_DIR / "leakage_audit.csv", "w", newline="") as f:
    csv.writer(f).writerows(leakage_audit_rows)

print("All CSV audit tables generated successfully.")
