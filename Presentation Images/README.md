# SatQuery AI — Presentation & Demonstration Imagery Catalog

This directory contains real satellite and remote-sensing Earth-observation imagery curated for live system presentations, evaluation demonstrations, and specialist verification.

All images are authentic remote-sensing datasets sourced from **Copernicus Sentinel-2**, **USGS Orthoimagery / NAIP**, **NASA Landsat 8**, **LEVIR-CD Satellite Benchmark**, and **Sentinel-1 SAR**.

---

## Directory Structure

```
Presentation Images/
├── Division_2_Single_Image_Specialist/
│   ├── 01_sentinel2_urban_and_agricultural_fabric.png
│   ├── 02_usgs_aerial_airfield_and_runway_grounding.png
│   ├── 03_landsat8_commercial_logistics_transport_grid.png
│   ├── 04_sentinel2_lake_balkhash_delta_and_wetlands.png
│   ├── 05_sentinel2_lake_powell_reservoir_canyon_hydrology.png
│   └── 06_sentinel2_lake_st_clair_estuary_and_agriculture.png
│
├── Division_3_BiTemporal_Change_Specialist/
│   ├── scene_01_residential_neighborhood_expansion/
│   │   ├── T0_baseline_acquisition.png
│   │   └── T1_developed_acquisition.png
│   ├── scene_02_suburban_residential_construction/
│   │   ├── T0_baseline_acquisition.png
│   │   └── T1_developed_acquisition.png
│   ├── scene_03_commercial_building_addition/
│   │   ├── T0_baseline_acquisition.png
│   │   └── T1_developed_acquisition.png
│   ├── scene_04_dense_housing_and_road_development/
│   │   ├── T0_baseline_acquisition.png
│   │   └── T1_developed_acquisition.png
│   ├── scene_05_industrial_and_logistics_expansion/
│   │   ├── T0_baseline_acquisition.png
│   │   └── T1_developed_acquisition.png
│   └── scene_06_urban_parcel_redevelopment/
│       ├── T0_baseline_acquisition.png
│       └── T1_developed_acquisition.png
│
└── Division_4_Optical_SAR_CrossModal_Specialist/
    ├── scene_01_cloud_covered_agricultural_parcels/
    │   ├── optical_sentinel2_rgb.png
    │   └── sar_sentinel1_cband_amplitude.tif
    ├── scene_02_coastal_harbor_and_vessel_shipping/
    │   ├── optical_sentinel2_rgb.png
    │   └── sar_sentinel1_cband_amplitude.tif
    ├── scene_03_urban_metropolis_metallic_scattering/
    │   ├── optical_sentinel2_rgb.png
    │   └── sar_sentinel1_cband_amplitude.tif
    ├── scene_04_estuary_wetlands_and_waterway/
    │   ├── optical_sentinel2_rgb.png
    │   └── sar_sentinel1_cband_amplitude.tif
    └── scene_05_industrial_logistics_hub_infrastructure/
        ├── optical_sentinel2_rgb.png
        └── sar_sentinel1_cband_amplitude.tif
```

---

## 1. Division 2 — Single-Image Specialist (VQA, Grounding, Captioning)

| File | Sensor / Platform | Spatial Resolution | Scene Content | Suggested Test Queries |
| :--- | :--- | :--- | :--- | :--- |
| `01_sentinel2_urban_and_agricultural_fabric.png` | Copernicus Sentinel-2 MSI | $10\text{ m/px}$ ($512\times 512$) | Mixed urban commercial infrastructure, agricultural fields, river basin | *"What is the dominant land cover in this scene?"*<br>*"What is the least land covered in this image?"* |
| `02_usgs_aerial_airfield_and_runway_grounding.png` | USGS Aerial Orthoimagery (NAIP) | $0.5\text{ m/px}$ ($512\times 512$) | High-resolution international airfield, runways, taxiways, apron | *"Where is the airport runway and apron in this image?"*<br>*"Locate the runway infrastructure."* |
| `03_landsat8_commercial_logistics_transport_grid.png` | NASA / USGS Landsat 8 OLI | $15\text{ m/px}$ ($512\times 512$) | Commercial logistics hubs, freeway intersections, suburban grid | *"Describe the transportation infrastructure and land use."* |
| `04_sentinel2_lake_balkhash_delta_and_wetlands.png` | Copernicus Sentinel-2 MSI | $10\text{ m/px}$ ($512\times 512$) | Water delta, riparian vegetation, silt sedimentation | *"Where is the water body located?"*<br>*"Analyze the wetland vegetation."* |
| `05_sentinel2_lake_powell_reservoir_canyon_hydrology.png` | Copernicus Sentinel-2 MSI | $10\text{ m/px}$ ($512\times 512$) | Canyon landscape, narrow reservoir arms, arid rock formations | *"What geological and hydrological features are present?"* |
| `06_sentinel2_lake_st_clair_estuary_and_agriculture.png` | Copernicus Sentinel-2 MSI | $10\text{ m/px}$ ($512\times 512$) | Coastal water estuary, agricultural parcel grids, shoreline | *"Is there any coastline or water body in this image?"* |

---

## 2. Division 3 — Bi-Temporal Change Specialist (TinyCD Neural Model)

All scenes are authentic $0.5\text{m/pixel}$ bi-temporal satellite pairs from the **LEVIR-CD** benchmark representing genuine real-world building additions, road construction, and parcel development.

| Scene Folder | Baseline ($T_0$) | Follow-Up ($T_1$) | Real Change Event | Suggested Test Queries |
| :--- | :--- | :--- | :--- | :--- |
| `scene_01_residential_neighborhood_expansion/` | Undeveloped / open land | New residential houses & access driveways | Multi-unit residential housing addition | *"What changed between these two acquisition dates?"* |
| `scene_02_suburban_residential_construction/` | Agricultural / bare plot | Single-family housing structures | New building construction | *"Has there been any built-up expansion or construction?"* |
| `scene_03_commercial_building_addition/` | Open parcel | Large commercial / storage facility | Commercial structural expansion | *"Identify newly constructed facilities between T0 and T1."* |
| `scene_04_dense_housing_and_road_development/` | Open fields | Dense housing complex & roads | High-density urban development | *"Where did surface changes occur and what is the changed area?"* |
| `scene_05_industrial_and_logistics_expansion/` | Vacant lot | Industrial warehouse & yard | Industrial development | *"Detect all new infrastructure additions."* |
| `scene_06_urban_parcel_redevelopment/` | Demolished / cleared site | Reconstructed multi-story structures | Urban infill & rebuilding | *"What is the change percentage between these dates?"* |

---

## 3. Division 4 — Optical + SAR Cross-Modal Specialist

Each scene contains a co-registered **Sentinel-2 optical RGB image** and a **Sentinel-1 C-Band SAR amplitude GeoTIFF raster** representing radar cloud penetration and metallic surface scattering.

| Scene Folder | Optical Component | SAR Component | Phenomenon Demonstrated | Suggested Test Queries |
| :--- | :--- | :--- | :--- | :--- |
| `scene_01_cloud_covered_agricultural_parcels/` | Cloud-obscured optical | C-Band SAR Amplitude (`.tif`) | Radar penetration through atmospheric cloud cover | *"Use optical and SAR images together to identify structures beneath clouds."* |
| `scene_02_coastal_harbor_and_vessel_shipping/` | Coastal optical scene | C-Band SAR Amplitude (`.tif`) | Strong metallic dihedral ship scattering in waterways | *"Identify vessels and harbor infrastructure using joint SAR-optical data."* |
| `scene_03_urban_metropolis_metallic_scattering/` | Urban optical view | C-Band SAR Amplitude (`.tif`) | High double-bounce returns from building corners | *"Detect urban density through cloud cover using radar backscatter."* |
| `scene_04_estuary_wetlands_and_waterway/` | Estuary optical view | C-Band SAR Amplitude (`.tif`) | Low specular water backscatter vs rough vegetation | *"Delineate water bodies and flood extent across modalities."* |
| `scene_05_industrial_logistics_hub_infrastructure/` | Industrial complex | C-Band SAR Amplitude (`.tif`) | High radar reflectivity of metallic storage tanks & hangars | *"Characterize industrial assets using cross-modal analysis."* |

---

## How to Test in the SatQuery Web Workspace

1. Open **SatQuery AI** at `http://127.0.0.1:5173`.
2. Switch to the **Upload Rasters** tab in the left panel.
3. Drag and drop any image (or pair of images) from `Presentation Images/`.
4. Enter your custom natural-language query and click **Run Agent Analysis**.
