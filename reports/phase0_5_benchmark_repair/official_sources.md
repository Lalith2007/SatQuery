# SATQUERY AI — PHASE 0.5: OFFICIAL BENCHMARK SOURCES

**Document Purpose:** Establishes the authoritative repository provenance, official citations, dataset release versions, and physical file specifications for every benchmark evaluated by SatQuery AI.

---

## 1. VRSBench (Versatile Vision-Language Benchmark for Remote Sensing)

- **Official Repository:** [https://github.com/lx709/VRSBench](https://github.com/lx709/VRSBench)
- **Hugging Face Asset:** [https://huggingface.co/datasets/xiang709/VRSBench](https://huggingface.co/datasets/xiang709/VRSBench)
- **Primary Paper:** Xiang Li, Jian Ding, Mohamed Elhoseiny, *"VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding"*, arXiv:2406.12384, 2024.
- **Official Release Note (2024.10.15 & 2026.06.11):**
  - **Coordinate Normalization:** All bounding box coordinates in `prepare_eval_all.ipynb` and provided evaluation JSON files on Hugging Face are **normalized to 0–100**.
  - **Grounding Evaluation:** Visual grounding evaluation protocol uses `Acc at iou_0.5` and `Acc at iou_0.7` disaggregated by Unique and Non-Unique expressions.
  - **Captioning Evaluation:** Official automatic metrics are BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, and CIDEr, alongside the GPT-based CHIAR metric.
  - **VQA Protocol:** Official evaluation classifies queries into 12 categories, enforcing exact matches for closed-set answers (yes/no and numbers 0–99).
- **Physical Files Acquired:**
  - `data/benchmark_samples/vrsbench/VRSBench_EVAL_Cap.json` (4,809,521 bytes, 9,350 evaluation captions)
  - `data/benchmark_samples/vrsbench/VRSBench_EVAL_referring.json` (10,277,680 bytes, 16,159 grounding expressions)
  - `data/benchmark_samples/vrsbench/VRSBench_EVAL_vqa.json` (9,358,245 bytes, 37,409 question-answer pairs)
  - `data/benchmark_samples/vrsbench/Images_val/` (4,854 validation optical remote sensing rasters)

---

## 2. CDVQA (Change Detection Visual Question Answering)

- **Official Repository:** [https://github.com/YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA)
- **Primary Paper:** Zhenghang Yuan, Lichao Mou, Zhitong Xiong, Xiao Xiang Zhu, *"Change Detection Meets Visual Question Answering"*, IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 60, pp. 1-13, 2022.
- **Crucial Distinction from Prior SatQuery Assumptions:**
  - CDVQA is **NOT** a free-form image captioning task evaluated with BLEU-4 or ROUGE-L.
  - CDVQA is a **Visual Question Answering** benchmark comprising 39,686 test questions spanning 8 change query categories.
  - The official benchmark metric is **Overall Accuracy** and **Per-Category Accuracy**.
- **Physical Files Acquired:**
  - `data/official_cdvqa/Test_images.json` (2,021,288 bytes, 15,488 image pair entries)
  - `data/official_cdvqa/Test_questions.json` (7,921,042 bytes, 39,686 questions)
  - `data/official_cdvqa/Test_answers.json` (4,160,913 bytes, 39,686 ground-truth answers)
- **Question Categories Breakdown:**
  1. `change_or_not`: 13,882 questions (35.0%) — Target answers: `yes`, `no`
  2. `change_ratio_types`: 5,811 questions (14.6%) — Target answers: ratio range
  3. `decrease_or_not`: 4,658 questions (11.7%) — Target answers: `yes`, `no`
  4. `increase_or_not`: 4,600 questions (11.6%) — Target answers: `yes`, `no`
  5. `change_to_what`: 2,991 questions (7.5%) — Target answers: land cover class
  6. `smallest_change`: 2,904 questions (7.3%) — Target answers: land cover class
  7. `largest_change`: 2,904 questions (7.3%) — Target answers: land cover class
  8. `change_ratio`: 1,936 questions (4.9%) — Target answers: ratio range (e.g. `0_to_10`)

---

## 3. RSVQA-LR (Remote Sensing Visual Question Answering - Low Resolution)

- **Official Source:** Sylvain Lobry, Diego Marcos, Devis Tuia, *"RSVQA: Visual Question Answering for Remote Sensing Data"*, IEEE TGRS, vol. 58, no. 12, pp. 8555-8566, 2020.
- **Hugging Face Asset:** `dmarsili/RSVQA-LR-2k`
- **Dataset Properties:**
  - Built on genuine European Space Agency Sentinel-2 low-resolution multispectral imagery (10m GSD).
  - Evaluated on a frozen 2,000-sample validation split with embedded rasters.
  - Closed-set vocabulary comprising exactly 744 permissible answer classes across presence, comparison, count, and rural/urban questions.
- **Physical File on Disk:**
  - `data/benchmark_samples/rsvqa/rsvqa_lr_val.parquet` (174,052,999 bytes, exactly 2,000 records)

---

## 4. LEVIR-CD (Bitemporal Change Detection Benchmark)

- **Official Source:** Hao Chen, Zhenwei Shi, *"A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection"*, Remote Sensing, vol. 12, no. 10, 2020.
- **Test Partition:** Exactly 128 pairs of optical images ($1024 \times 1024$, 0.5m GSD) with pixel-level binary change masks.
- **Specialist Evaluated:** Frozen TinyCD checkpoint (`ChangeDetector-TinyCD.pth`, SHA-256: `b9a100935586...`).
- **Physical Directory:** `data/official_levir_cd/test/` (A, B, label).

---

## 5. WHU-OPT-SAR (Optical-SAR Cross-Modal Segmentation Benchmark)

- **Official Source:** Ying Li, Yang Zhou, et al., *"A Large-Scale Dataset for Cross-Modal Optical and SAR Land-Cover Classification"*, Remote Sensing, 2022.
- **Test Partition:** Exactly 4,950 tiles ($256 \times 256$) derived from 15 non-overlapping geographic scenes.
- **Modality Channels:** Optical (RGB, 3-channel, 0.45m GSD) + SAR (Gaofen-3 SAR, 1-channel, 1.0m GSD).
- **Specialist Evaluated:** Frozen CMAF checkpoint (`cmaf_landcover_best.pth`, SHA-256: `26288ce0e8...`).
- **Physical Directory:** `data/official_whu_opt_sar/test/` (optical, sar, labels).

---

## 6. BigEarthNet.txt (Multimodal Vision-Language Stage-1 Held-Out)

- **Official Source:** Kai Norman Clasen et al., BIFOLD BigEarthNet-v2.0, 2024.
- **Held-Out Partition:** 850 pairs with zero overlap against Stage-1 training mixtures.
- **Verification Authority:** Certified BigEarthNet verification report (`bigearthnet_stage1_verification_audit.json`).
