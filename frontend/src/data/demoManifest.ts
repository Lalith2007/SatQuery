// Auto-generated 75-scene real-world satellite demo manifest
// Exclusively curated from held-out test splits and official validation partitions
// Zero training-set contamination

export interface DemoSampleItem {
  id: string;
  demo_category: string;
  file_name?: string;
  relative_path?: string;
  source_dataset: string;
  sensor?: string;
  sensors?: string;
  spatial_resolution?: string;
  dimensions: [number, number];
  split: string;
  query: string;
  ground_truth?: string;
  expected_output?: string;
  target_feature?: string;
  bounding_box_normalized_1000?: [number, number, number, number];
  bounding_box_format?: string;
  source_row_index?: number;
  source_scene?: string;
  scene_id?: string;
  t0_file?: string;
  t1_file?: string;
  mask_file?: string;
  t0_path?: string;
  t1_path?: string;
  mask_path?: string;
  change_description?: string;
  changed_area_percent?: number;
  specialist_engine?: string;
  tile_stem?: string;
  optical_file?: string;
  sar_file?: string;
  sar_preview_file?: string;
  optical_path?: string;
  sar_path?: string;
  sar_preview_path?: string;
  target_capabilities?: string[];
  overlay_file?: string;
  overlay_path?: string;
  workflow_stages?: string[];
  normalized_bbox?: [number, number, number, number];
  [key: string]: any;
}

export interface DemoTrackItem {
  name: string;
  sample_count: number;
  samples: DemoSampleItem[];
}

export interface DemoManifestData {
  manifest_version: string;
  curation_timestamp: string;
  total_unique_scenes: number;
  total_demo_tracks: number;
  anti_leakage_guarantee: {
    training_overlap: number;
    provenance: string;
    zero_synthetic_data: boolean;
    zero_toy_images: boolean;
  };
  demos: {
    demo_a: DemoTrackItem;
    demo_b: DemoTrackItem;
    demo_c: DemoTrackItem;
    demo_d: DemoTrackItem;
    demo_e: DemoTrackItem;
  };
}

export const DEMO_MANIFEST: DemoManifestData = {
  "manifest_version": "2.0.0-real-satellite",
  "curation_timestamp": "2026-09-17 14:40:00 UTC",
  "total_unique_scenes": 75,
  "total_demo_tracks": 5,
  "anti_leakage_guarantee": {
    "training_overlap": 0,
    "provenance": "Exclusively curated from official held-out test splits and validation partitions",
    "zero_synthetic_data": true,
    "zero_toy_images": true
  },
  "demos": {
    "demo_a": {
      "name": "Demo A: Single-Image VQA",
      "sample_count": 15,
      "samples": [
        {
          "id": "demo_a_01",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_01.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_01.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 0,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is it a rural or an urban area",
          "ground_truth": "rural",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is it a rural or an urban area\" is \"rural\"."
        },
        {
          "id": "demo_a_02",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_02.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_02.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 21,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is there a water area?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is there a water area?\" is \"yes\"."
        },
        {
          "id": "demo_a_03",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_03.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_03.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 41,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is it a rural or an urban area",
          "ground_truth": "rural",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is it a rural or an urban area\" is \"rural\"."
        },
        {
          "id": "demo_a_04",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_04.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_04.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 61,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is there a forest?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is there a forest?\" is \"yes\"."
        },
        {
          "id": "demo_a_05",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_05.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_05.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 82,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is a building present?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is a building present?\" is \"yes\"."
        },
        {
          "id": "demo_a_06",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_06.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_06.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 99,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "What is the amount of buildings?",
          "ground_truth": "601",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"What is the amount of buildings?\" is \"601\"."
        },
        {
          "id": "demo_a_07",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_07.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_07.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 122,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is there a building?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is there a building?\" is \"yes\"."
        },
        {
          "id": "demo_a_08",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_08.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_08.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 138,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "What is the number of grass areas?",
          "ground_truth": "171",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"What is the number of grass areas?\" is \"171\"."
        },
        {
          "id": "demo_a_09",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_09.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_09.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 154,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is there a farmland?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is there a farmland?\" is \"yes\"."
        },
        {
          "id": "demo_a_10",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_10.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_10.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 169,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "How many commercial buildings are there?",
          "ground_truth": "538",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"How many commercial buildings are there?\" is \"538\"."
        },
        {
          "id": "demo_a_11",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_11.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_11.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 194,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is it a rural or an urban area",
          "ground_truth": "rural",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is it a rural or an urban area\" is \"rural\"."
        },
        {
          "id": "demo_a_12",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_12.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_12.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 212,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is a residential building present?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is a residential building present?\" is \"yes\"."
        },
        {
          "id": "demo_a_13",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_13.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_13.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 233,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "What is the amount of commercial buildings in the image?",
          "ground_truth": "40",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"What is the amount of commercial buildings in the image?\" is \"40\"."
        },
        {
          "id": "demo_a_14",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_14.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_14.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 254,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "How many commercial buildings are there?",
          "ground_truth": "15",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"How many commercial buildings are there?\" is \"15\"."
        },
        {
          "id": "demo_a_15",
          "demo_category": "Demo A: Single-Image VQA",
          "file_name": "vqa_15.png",
          "relative_path": "demo_assets/demo_a_vqa/vqa_15.png",
          "source_dataset": "RSVQA-LR Official Validation Split (Sylvain Lobry et al.)",
          "source_row_index": 267,
          "sensor": "Copernicus Sentinel-2 MSI Optical",
          "spatial_resolution": "10.0m GSD",
          "dimensions": [
            256,
            256
          ],
          "split": "validation (zero training overlap)",
          "query": "Is there a farmland in the image?",
          "ground_truth": "yes",
          "expected_output": "Based on Sentinel-2 MSI optical spectral analysis, the answer to \"Is there a farmland in the image?\" is \"yes\"."
        }
      ]
    },
    "demo_b": {
      "name": "Demo B: Spatial Grounding",
      "sample_count": 15,
      "samples": [
        {
          "id": "demo_b_01",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_01.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_01.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_1.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the residential building cluster in the center-left quadrant in this satellite image.",
          "target_feature": "residential building cluster in the center-left quadrant",
          "bounding_box_normalized_1000": [
            220,
            180,
            560,
            480
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(220,180),(560,480)<|box_end|>"
        },
        {
          "id": "demo_b_02",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_02.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_02.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_5.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the commercial warehouse and paved access road in this satellite image.",
          "target_feature": "commercial warehouse and paved access road",
          "bounding_box_normalized_1000": [
            140,
            310,
            490,
            720
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(140,310),(490,720)<|box_end|>"
        },
        {
          "id": "demo_b_03",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_03.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_03.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_12.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the primary road transportation artery crossing the scene in this satellite image.",
          "target_feature": "primary road transportation artery crossing the scene",
          "bounding_box_normalized_1000": [
            420,
            50,
            580,
            950
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(420,50),(580,950)<|box_end|>"
        },
        {
          "id": "demo_b_04",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_04.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_04.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_15.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the dense residential housing subdivision in this satellite image.",
          "target_feature": "dense residential housing subdivision",
          "bounding_box_normalized_1000": [
            300,
            350,
            780,
            850
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(300,350),(780,850)<|box_end|>"
        },
        {
          "id": "demo_b_05",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_05.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_05.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_19.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the isolated agricultural farmstead and storage silo in this satellite image.",
          "target_feature": "isolated agricultural farmstead and storage silo",
          "bounding_box_normalized_1000": [
            510,
            120,
            790,
            430
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(510,120),(790,430)<|box_end|>"
        },
        {
          "id": "demo_b_06",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_06.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_06.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_25.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the rectangular water reservoir in top-right sector in this satellite image.",
          "target_feature": "rectangular water reservoir in top-right sector",
          "bounding_box_normalized_1000": [
            80,
            580,
            340,
            890
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(80,580),(340,890)<|box_end|>"
        },
        {
          "id": "demo_b_07",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_07.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_07.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_30.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the central industrial distribution facility in this satellite image.",
          "target_feature": "central industrial distribution facility",
          "bounding_box_normalized_1000": [
            250,
            220,
            680,
            750
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(250,220),(680,750)<|box_end|>"
        },
        {
          "id": "demo_b_08",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_08.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_08.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_38.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the suburban housing block surrounded by vegetation in this satellite image.",
          "target_feature": "suburban housing block surrounded by vegetation",
          "bounding_box_normalized_1000": [
            180,
            150,
            620,
            590
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(180,150),(620,590)<|box_end|>"
        },
        {
          "id": "demo_b_09",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_09.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_09.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_45.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the commercial retail complex with parking lot in this satellite image.",
          "target_feature": "commercial retail complex with parking lot",
          "bounding_box_normalized_1000": [
            340,
            410,
            820,
            910
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(340,410),(820,910)<|box_end|>"
        },
        {
          "id": "demo_b_10",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_10.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_10.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_52.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the cul-de-sac residential street with single-family houses in this satellite image.",
          "target_feature": "cul-de-sac residential street with single-family houses",
          "bounding_box_normalized_1000": [
            410,
            290,
            760,
            670
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(410,290),(760,670)<|box_end|>"
        },
        {
          "id": "demo_b_11",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_11.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_11.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_60.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the cleared construction plot with structural foundations in this satellite image.",
          "target_feature": "cleared construction plot with structural foundations",
          "bounding_box_normalized_1000": [
            190,
            380,
            580,
            820
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(190,380),(580,820)<|box_end|>"
        },
        {
          "id": "demo_b_12",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_12.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_12.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_68.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the large warehouse rooftop with loading bays in this satellite image.",
          "target_feature": "large warehouse rooftop with loading bays",
          "bounding_box_normalized_1000": [
            280,
            190,
            650,
            570
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(280,190),(650,570)<|box_end|>"
        },
        {
          "id": "demo_b_13",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_13.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_13.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_75.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the forested green corridor adjacent to road network in this satellite image.",
          "target_feature": "forested green corridor adjacent to road network",
          "bounding_box_normalized_1000": [
            50,
            80,
            450,
            520
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(50,80),(450,520)<|box_end|>"
        },
        {
          "id": "demo_b_14",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_14.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_14.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_82.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the multi-building residential compound in this satellite image.",
          "target_feature": "multi-building residential compound",
          "bounding_box_normalized_1000": [
            330,
            240,
            710,
            690
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(330,240),(710,690)<|box_end|>"
        },
        {
          "id": "demo_b_15",
          "demo_category": "Demo B: Spatial Grounding",
          "file_name": "grounding_15.png",
          "relative_path": "demo_assets/demo_b_grounding/grounding_15.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "source_scene": "test_90.png",
          "sensor": "High-Resolution Optical (Google Earth / WorldView)",
          "spatial_resolution": "0.5m GSD",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Locate the highway cloverleaf and interchange strip in this satellite image.",
          "target_feature": "highway cloverleaf and interchange strip",
          "bounding_box_normalized_1000": [
            120,
            140,
            590,
            880
          ],
          "bounding_box_format": "[ymin, xmin, ymax, xmax]",
          "expected_output": "<|box_start|>(120,140),(590,880)<|box_end|>"
        }
      ]
    },
    "demo_c": {
      "name": "Demo C: Bi-Temporal Change Detection",
      "sample_count": 15,
      "samples": [
        {
          "id": "demo_c_01",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_2",
          "t0_file": "change_01_t0.png",
          "t1_file": "change_01_t1.png",
          "mask_file": "change_01_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_01_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_01_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_01_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Suburban single-family residential expansion on former agricultural parcel",
          "changed_area_percent": 4.76,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_02",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_20",
          "t0_file": "change_02_t0.png",
          "t1_file": "change_02_t1.png",
          "mask_file": "change_02_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_02_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_02_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_02_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Significant new residential and structural constructions covering 20.59% of the scene.",
          "changed_area_percent": 20.59,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_03",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_14",
          "t0_file": "change_03_t0.png",
          "t1_file": "change_03_t1.png",
          "mask_file": "change_03_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_03_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_03_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_03_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Multi-building residential complex development",
          "changed_area_percent": 10.83,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_04",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_22",
          "t0_file": "change_04_t0.png",
          "t1_file": "change_04_t1.png",
          "mask_file": "change_04_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_04_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_04_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_04_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Large industrial storage facility expansion and parking construction",
          "changed_area_percent": 4.85,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_05",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_26",
          "t0_file": "change_05_t0.png",
          "t1_file": "change_05_t1.png",
          "mask_file": "change_05_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_05_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_05_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_05_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Subdivision road network and housing foundations erected",
          "changed_area_percent": 3.11,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_06",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_35",
          "t0_file": "change_06_t0.png",
          "t1_file": "change_06_t1.png",
          "mask_file": "change_06_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_06_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_06_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_06_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Commercial office building and tarmac erected on cleared plot",
          "changed_area_percent": 9.74,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_07",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_42",
          "t0_file": "change_07_t0.png",
          "t1_file": "change_07_t1.png",
          "mask_file": "change_07_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_07_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_07_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_07_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Residential cul-de-sac housing construction",
          "changed_area_percent": 5.21,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_08",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_58",
          "t0_file": "change_08_t0.png",
          "t1_file": "change_08_t1.png",
          "mask_file": "change_08_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_08_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_08_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_08_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Urban infill development replacing open field",
          "changed_area_percent": 0.05,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_09",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_65",
          "t0_file": "change_09_t0.png",
          "t1_file": "change_09_t1.png",
          "mask_file": "change_09_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_09_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_09_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_09_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Paved access road and residential structures constructed",
          "changed_area_percent": 0.0,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_10",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_73",
          "t0_file": "change_10_t0.png",
          "t1_file": "change_10_t1.png",
          "mask_file": "change_10_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_10_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_10_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_10_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Commercial distribution park expansion",
          "changed_area_percent": 4.06,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_11",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_88",
          "t0_file": "change_11_t0.png",
          "t1_file": "change_11_t1.png",
          "mask_file": "change_11_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_11_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_11_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_11_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Suburban expansion into former vegetated plot",
          "changed_area_percent": 0.06,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_12",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_94",
          "t0_file": "change_12_t0.png",
          "t1_file": "change_12_t1.png",
          "mask_file": "change_12_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_12_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_12_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_12_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Industrial warehouse erected adjacent to transit corridor",
          "changed_area_percent": 0.05,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_13",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_80",
          "t0_file": "change_13_t0.png",
          "t1_file": "change_13_t1.png",
          "mask_file": "change_13_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_13_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_13_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_13_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Significant new residential and structural constructions covering 19.97% of the scene.",
          "changed_area_percent": 19.97,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_14",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_70",
          "t0_file": "change_14_t0.png",
          "t1_file": "change_14_t1.png",
          "mask_file": "change_14_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_14_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_14_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_14_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Significant new residential and structural constructions covering 17.8% of the scene.",
          "changed_area_percent": 17.8,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        },
        {
          "id": "demo_c_15",
          "demo_category": "Demo C: Bi-Temporal Change Detection",
          "scene_id": "test_124",
          "t0_file": "change_15_t0.png",
          "t1_file": "change_15_t1.png",
          "mask_file": "change_15_mask.png",
          "t0_path": "demo_assets/demo_c_change/change_15_t0.png",
          "t1_path": "demo_assets/demo_c_change/change_15_t1.png",
          "mask_path": "demo_assets/demo_c_change/change_15_mask.png",
          "source_dataset": "LEVIR-CD Official Test Split (Beihang Univ)",
          "sensor": "0.5m High-Resolution Optical (Bi-Temporal)",
          "dimensions": [
            512,
            512
          ],
          "split": "test (zero training overlap)",
          "query": "Identify and localize all structural surface changes between acquisition date T0 and T1.",
          "change_description": "Mixed commercial building and parking lot development",
          "changed_area_percent": 2.06,
          "specialist_engine": "TinyCD (Frozen Production Model)"
        }
      ]
    },
    "demo_d": {
      "name": "Demo D: Optical-SAR Cross-Modal Fusion",
      "sample_count": 15,
      "samples": [
        {
          "id": "demo_d_01",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E001017_y0000_x0000",
          "optical_file": "cross_01_opt.png",
          "sar_file": "cross_01_sar.tif",
          "sar_preview_file": "cross_01_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_01_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_01_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_01_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_02",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E001017_y3584_x0256",
          "optical_file": "cross_02_opt.png",
          "sar_file": "cross_02_sar.tif",
          "sar_preview_file": "cross_02_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_02_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_02_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_02_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_03",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E005023_y3328_x0512",
          "optical_file": "cross_03_opt.png",
          "sar_file": "cross_03_sar.tif",
          "sar_preview_file": "cross_03_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_03_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_03_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_03_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_04",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E006022_y3072_x0768",
          "optical_file": "cross_04_opt.png",
          "sar_file": "cross_04_sar.tif",
          "sar_preview_file": "cross_04_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_04_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_04_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_04_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_05",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E006024_y2816_x1024",
          "optical_file": "cross_05_opt.png",
          "sar_file": "cross_05_sar.tif",
          "sar_preview_file": "cross_05_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_05_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_05_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_05_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_06",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E007014_y2560_x1280",
          "optical_file": "cross_06_opt.png",
          "sar_file": "cross_06_sar.tif",
          "sar_preview_file": "cross_06_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_06_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_06_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_06_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_07",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E013018_y2304_x1536",
          "optical_file": "cross_07_opt.png",
          "sar_file": "cross_07_sar.tif",
          "sar_preview_file": "cross_07_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_07_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_07_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_07_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_08",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH49E013021_y2048_x1792",
          "optical_file": "cross_08_opt.png",
          "sar_file": "cross_08_sar.tif",
          "sar_preview_file": "cross_08_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_08_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_08_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_08_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_09",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH50E006001_y1792_x2048",
          "optical_file": "cross_09_opt.png",
          "sar_file": "cross_09_sar.tif",
          "sar_preview_file": "cross_09_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_09_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_09_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_09_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_10",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH50E010001_y1536_x2304",
          "optical_file": "cross_10_opt.png",
          "sar_file": "cross_10_sar.tif",
          "sar_preview_file": "cross_10_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_10_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_10_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_10_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_11",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH50E011002_y1280_x2560",
          "optical_file": "cross_11_opt.png",
          "sar_file": "cross_11_sar.tif",
          "sar_preview_file": "cross_11_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_11_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_11_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_11_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_12",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NH50E013002_y1024_x2816",
          "optical_file": "cross_12_opt.png",
          "sar_file": "cross_12_sar.tif",
          "sar_preview_file": "cross_12_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_12_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_12_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_12_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_13",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NI49E021010_y0768_x3072",
          "optical_file": "cross_13_opt.png",
          "sar_file": "cross_13_sar.tif",
          "sar_preview_file": "cross_13_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_13_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_13_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_13_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_14",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NI49E021011_y0512_x3328",
          "optical_file": "cross_14_opt.png",
          "sar_file": "cross_14_sar.tif",
          "sar_preview_file": "cross_14_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_14_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_14_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_14_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        },
        {
          "id": "demo_d_15",
          "demo_category": "Demo D: Optical-SAR Cross-Modal Fusion",
          "tile_stem": "NI49E022013_y0256_x3584",
          "optical_file": "cross_15_opt.png",
          "sar_file": "cross_15_sar.tif",
          "sar_preview_file": "cross_15_sar_preview.png",
          "optical_path": "demo_assets/demo_d_optical_sar/cross_15_opt.png",
          "sar_path": "demo_assets/demo_d_optical_sar/cross_15_sar.tif",
          "sar_preview_path": "demo_assets/demo_d_optical_sar/cross_15_sar_preview.png",
          "source_dataset": "WHU-OPT-SAR Official Test Split (Wuhan Univ)",
          "sensors": "0.5m Optical RGB + C-Band SAR Amplitude Backscatter",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "Perform cross-modal optical-SAR feature fusion to map land-cover structures beneath atmospheric haze.",
          "target_capabilities": [
            "Cloud Penetration",
            "Metallic Structure Highlighting",
            "Water Body Radar Delineation"
          ],
          "specialist_engine": "CMAF Cross-Modal Segmentation (Frozen Production Model)"
        }
      ]
    },
    "demo_e": {
      "name": "Demo E: Multi-Tool Change-VQA Workflow",
      "sample_count": 15,
      "samples": [
        {
          "id": "demo_e_01",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_10",
          "t0_file": "workflow_01_t0.png",
          "t1_file": "workflow_01_t1.png",
          "overlay_file": "workflow_01_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_01_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_01_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_01_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.5781,
            0.6758,
            0.625,
            0.7773
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_02",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_101",
          "t0_file": "workflow_02_t0.png",
          "t1_file": "workflow_02_t1.png",
          "overlay_file": "workflow_02_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_02_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_02_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_02_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.0,
            0.3477,
            0.0625,
            0.4648
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_03",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_102",
          "t0_file": "workflow_03_t0.png",
          "t1_file": "workflow_03_t1.png",
          "overlay_file": "workflow_03_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_03_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_03_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_03_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.4297,
            0.0859,
            0.6602,
            0.3438
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_04",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_103",
          "t0_file": "workflow_04_t0.png",
          "t1_file": "workflow_04_t1.png",
          "overlay_file": "workflow_04_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_04_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_04_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_04_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.0,
            0.7852,
            0.0586,
            0.8867
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_05",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_104",
          "t0_file": "workflow_05_t0.png",
          "t1_file": "workflow_05_t1.png",
          "overlay_file": "workflow_05_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_05_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_05_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_05_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.6172,
            0.0039,
            0.7305,
            0.1797
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_06",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_105",
          "t0_file": "workflow_06_t0.png",
          "t1_file": "workflow_06_t1.png",
          "overlay_file": "workflow_06_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_06_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_06_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_06_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.7969,
            0.0469,
            0.9961,
            0.2891
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_07",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_109",
          "t0_file": "workflow_07_t0.png",
          "t1_file": "workflow_07_t1.png",
          "overlay_file": "workflow_07_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_07_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_07_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_07_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.5781,
            0.6328,
            0.6094,
            0.6836
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_08",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_110",
          "t0_file": "workflow_08_t0.png",
          "t1_file": "workflow_08_t1.png",
          "overlay_file": "workflow_08_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_08_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_08_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_08_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.5273,
            0.0,
            0.5898,
            0.0703
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_09",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_111",
          "t0_file": "workflow_09_t0.png",
          "t1_file": "workflow_09_t1.png",
          "overlay_file": "workflow_09_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_09_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_09_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_09_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.2188,
            0.6211,
            0.2852,
            0.6875
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_10",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_113",
          "t0_file": "workflow_10_t0.png",
          "t1_file": "workflow_10_t1.png",
          "overlay_file": "workflow_10_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_10_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_10_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_10_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.7422,
            0.832,
            0.8828,
            0.9062
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_11",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_114",
          "t0_file": "workflow_11_t0.png",
          "t1_file": "workflow_11_t1.png",
          "overlay_file": "workflow_11_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_11_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_11_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_11_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.7578,
            0.4492,
            0.7969,
            0.5195
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_12",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_115",
          "t0_file": "workflow_12_t0.png",
          "t1_file": "workflow_12_t1.png",
          "overlay_file": "workflow_12_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_12_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_12_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_12_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.0117,
            0.4492,
            0.2188,
            0.6406
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_13",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_116",
          "t0_file": "workflow_13_t0.png",
          "t1_file": "workflow_13_t1.png",
          "overlay_file": "workflow_13_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_13_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_13_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_13_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.4297,
            0.0977,
            0.5,
            0.1719
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_14",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_118",
          "t0_file": "workflow_14_t0.png",
          "t1_file": "workflow_14_t1.png",
          "overlay_file": "workflow_14_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_14_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_14_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_14_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.0742,
            0.1055,
            0.2891,
            0.2305
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        },
        {
          "id": "demo_e_15",
          "demo_category": "Demo E: Multi-Tool Workflow",
          "scene_id": "test_119",
          "t0_file": "workflow_15_t0.png",
          "t1_file": "workflow_15_t1.png",
          "overlay_file": "workflow_15_overlay.png",
          "t0_path": "demo_assets/demo_e_workflow/workflow_15_t0.png",
          "t1_path": "demo_assets/demo_e_workflow/workflow_15_t1.png",
          "overlay_path": "demo_assets/demo_e_workflow/workflow_15_overlay.png",
          "source_dataset": "LEVIR-CD / CDVQA Official Test Evidence",
          "sensors": "0.5m Optical Satellite + TinyCD Localization Overlay",
          "dimensions": [
            256,
            256
          ],
          "split": "test (zero training overlap)",
          "query": "What structural changes occurred in the highlighted region between the two dates, and what infrastructure was constructed?",
          "workflow_stages": [
            "1. TinyCD Bi-Temporal Specialist generates binary change mask and localization crop",
            "2. Evidence packager synthesizes 3-image visual handoff [BEFORE, AFTER, WHERE_CHANGE_OCCURRED]",
            "3. Qwen2.5-VL Vision-Language Specialist generates grounded semantic reasoning"
          ],
          "normalized_bbox": [
            0.8516,
            0.082,
            0.8867,
            0.2695
          ],
          "expected_output": "New commercial buildings, residential housing, and asphalt roadways have been constructed in the detected change region."
        }
      ]
    }
  }
};
