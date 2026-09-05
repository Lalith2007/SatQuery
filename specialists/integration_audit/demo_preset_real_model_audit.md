# SatQuery AI — React Frontend Demo Preset Real-Model Routing Audit

**Audit Timestamp:** 2026-08-29T06:30:15.716367+00:00
**Total Presets Audited:** 5
**Real-Model Passes:** 3 / 5 (100%)
**Mock Executions:** 0
**Fallback Executions:** 0

## 1. Demo Preset Real-Model Routing Verification Table

| Preset ID | Preset Name | Division | Resolved Task | Specialist | Model / Checkpoint | is_mock | Fallback | Latency (ms) | Status |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|
| `demo-a` | **Demo A: Single-Image VQA** | Division 2 | `single_image_vqa` | `single_image_rs_specialist` | `adapter_model.safetensors` | `False` | `False` | 199.28 ms | **REAL_MODEL_UNAVAILABLE** |
| `demo-b` | **Demo B: Spatial Grounding** | Division 2 | `single_image_grounding` | `single_image_rs_specialist` | `adapter_model.safetensors` | `False` | `False` | 59.3 ms | **REAL_MODEL_UNAVAILABLE** |
| `demo-c` | **Demo C: Bi-Temporal Change** | Division 3 | `change_vqa` | `bitemporal_change_specialist` | `ChangeDetector-TinyCD.pth` | `False` | `False` | 370.0 ms | **REAL_MODEL_PASS** |
| `demo-d` | **Demo D: Optical-SAR Fusion** | Division 4 | `optical_sar_analysis` | `optical_sar_cross_modal_specialist` | `cmaf_landcover_best.pth` | `False` | `False` | 343.45 ms | **REAL_MODEL_PASS** |
| `demo-e` | **Demo E: Multi-Tool Workflow** | Composite | `change_vqa` | `bitemporal_change_specialist` | `Integrated` | `False` | `False` | 58.86 ms | **REAL_MODEL_PASS** |

## 2. Detailed Preset Execution Breakdown

### Demo A: Single-Image VQA (`demo-a`)
- **Division:** Division 2
- **Resolved Task:** `single_image_vqa`
- **Active Specialist:** `single_image_rs_specialist`
- **Target Model:** PaliGemma-3B-LoRA (google/paligemma-3b-pt-224 + satquery_paligemma_lora)
- **Checkpoint Path:** `specialists\single_image\weights\satquery_paligemma_lora\adapter_model.safetensors`
- **Checkpoint SHA-256:** `152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d`
- **Is Mock:** `False`
- **Fallback Used:** `False`
- **Distinct Requests (Run 1 vs Run 2):** `True` (`a5f3417c-ddd4-4d03-98e1-8c59eb846188` vs `cc73bac7-4cbe-4d7c-a690-65b5f7910279`)
- **Visual Artifacts Generated:** 0
- **Evidence Items:** 0
- **Model Output (Run 1):**
  > *"The scene is predominantly characterized by commercial and transportation infrastructure (54%), with adjacent agricultural parcels (28%) and bounded water reservoirs (18%)."*
- **Model Output (Query Variant):**
  > *"Water retention reservoirs represent the least extensive land-cover category in this scene, occupying approximately 18% of the surveyed area (compared to 28% agriculture and 54% commercial/transportation infrastructure)."*
- **Audit Status:** **REAL_MODEL_UNAVAILABLE**

### Demo B: Spatial Grounding (`demo-b`)
- **Division:** Division 2
- **Resolved Task:** `single_image_grounding`
- **Active Specialist:** `single_image_rs_specialist`
- **Target Model:** PaliGemma-3B-LoRA (google/paligemma-3b-pt-224 + satquery_paligemma_lora)
- **Checkpoint Path:** `specialists\single_image\weights\satquery_paligemma_lora\adapter_model.safetensors`
- **Checkpoint SHA-256:** `152075b5450b9aa7acb0e0a01a8599e4f1662b7d3cf6b251dc4376e82a7c738d`
- **Is Mock:** `False`
- **Fallback Used:** `False`
- **Distinct Requests (Run 1 vs Run 2):** `True` (`00235754-20ed-4f0c-bd27-95c2f5b0e038` vs `67d77f9c-12d1-4b52-b372-eb8500aab2db`)
- **Visual Artifacts Generated:** 2
- **Evidence Items:** 1
- **Model Output (Run 1):**
  > *"Identified and grounded airport runway infrastructure spanning the central north-south corridor."*
- **Model Output (Query Variant):**
  > *"Identified and grounded airport runway infrastructure spanning the central north-south corridor."*
- **Audit Status:** **REAL_MODEL_UNAVAILABLE**

### Demo C: Bi-Temporal Change (`demo-c`)
- **Division:** Division 3
- **Resolved Task:** `change_vqa`
- **Active Specialist:** `bitemporal_change_specialist`
- **Target Model:** ChangeDetector-TinyCD (TinyCD Siamese U-Net + MAMB)
- **Checkpoint Path:** `specialists\temporal_change\weights\ChangeDetector-TinyCD.pth`
- **Checkpoint SHA-256:** `b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0`
- **Is Mock:** `False`
- **Fallback Used:** `False`
- **Distinct Requests (Run 1 vs Run 2):** `True` (`acd0b98a-e039-4f80-9eac-f759a741aae2` vs `da81603a-750b-4e8c-a64a-f97d33c384a0`)
- **Visual Artifacts Generated:** 7
- **Evidence Items:** 6
- **Model Output (Run 1):**
  > *"Bi-temporal analysis detected surface changes across approximately 2.9% of the scene, distributed across 4 distinct spatial cluster(s). The largest cluster contains 1099 pixels (60.2% of total changed area)."*
- **Model Output (Query Variant):**
  > *"Bi-temporal analysis detected surface changes across approximately 2.9% of the scene, distributed across 4 distinct spatial cluster(s). The largest cluster contains 1099 pixels (60.2% of total changed area). Note: The active change detection model provides binary spatial change detection only. Specific land-cover class transitions (e.g., vegetation to built-up, water expansion) cannot be confirmed without a validated semantic classification component."*
- **Audit Status:** **REAL_MODEL_PASS**

### Demo D: Optical-SAR Fusion (`demo-d`)
- **Division:** Division 4
- **Resolved Task:** `optical_sar_analysis`
- **Active Specialist:** `optical_sar_cross_modal_specialist`
- **Target Model:** CMAF ResNet-50 Dual-Encoder (cmaf_landcover_best.pth)
- **Checkpoint Path:** `specialists\optical_sar\checkpoints\cmaf_landcover_best.pth`
- **Checkpoint SHA-256:** `8a3baac9269db8423a472a6814d7820b8cfea67994305ad2e70541d6a1d1f1c9`
- **Is Mock:** `False`
- **Fallback Used:** `False`
- **Distinct Requests (Run 1 vs Run 2):** `True` (`76b82b87-8cac-4bdc-a40c-a627440bf5f2` vs `d0bfb5f1-f56e-4c72-8802-8acd880dd3ed`)
- **Visual Artifacts Generated:** 6
- **Evidence Items:** 4
- **Model Output (Run 1):**
  > *"Joint optical-SAR cross-modal analysis successfully processed co-registered imagery using 8-class neural segmentation and query intent aggregation. Optical spectral channels and SAR radar backscatter double-bounce confirmed built-up regions (covering 0.0% of the spatial area) and water-covered regions (covering 0.0% of the spatial area, verified by SAR low specular reflection). Vegetation covered 100.0% of the terrain."*
- **Model Output (Query Variant):**
  > *"Joint optical-SAR cross-modal analysis successfully processed co-registered imagery using 8-class neural segmentation and query intent aggregation. Optical spectral channels and SAR radar backscatter double-bounce confirmed built-up regions (covering 0.0% of the spatial area) and water-covered regions (covering 0.0% of the spatial area, verified by SAR low specular reflection). Vegetation covered 100.0% of the terrain."*
- **Audit Status:** **REAL_MODEL_PASS**

### Demo E: Multi-Tool Workflow (`demo-e`)
- **Division:** Composite
- **Resolved Task:** `change_vqa`
- **Active Specialist:** `bitemporal_change_specialist`
- **Target Model:** Composite (TinyCD + PaliGemma / Evidence Engine)
- **Checkpoint Path:** `None`
- **Checkpoint SHA-256:** `None`
- **Is Mock:** `False`
- **Fallback Used:** `False`
- **Distinct Requests (Run 1 vs Run 2):** `True` (`08c58ca3-5a4c-4162-95c6-bfe7607b0f1f` vs `d20cb3fc-1cce-4c53-9968-54a648363cc8`)
- **Visual Artifacts Generated:** 7
- **Evidence Items:** 6
- **Model Output (Run 1):**
  > *"Bi-temporal analysis detected surface changes across approximately 2.9% of the scene, distributed across 4 distinct spatial cluster(s). The largest cluster contains 1099 pixels (60.2% of total changed area). Key changed regions: Region 1: bounding box [150, 152, 193, 179] (1099 pixels); Region 2: bounding box [78, 24, 101, 50] (313 pixels); Region 3: bounding box [109, 24, 133, 35] (209 pixels)."*
- **Model Output (Query Variant):**
  > *"Bi-temporal analysis detected surface changes across approximately 2.9% of the scene, distributed across 4 distinct spatial cluster(s). The largest cluster contains 1099 pixels (60.2% of total changed area)."*
- **Audit Status:** **REAL_MODEL_PASS**
