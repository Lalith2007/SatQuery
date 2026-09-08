# SatQuery AI — Demo Preset Imagery & Attribution Sources

This document describes the provenance, sensors, spatial resolution, and licensing for all demonstration satellite and remote-sensing assets provided in `demo_assets/`.

---

## 1. Demo A — Single-Image Visual Question Answering (VQA)
* **File**: `demo_assets/demo_optical_single.png`
* **Sensor / Platform**: Copernicus Sentinel-2 MSI (MultiSpectral Instrument)
* **Agency / Provider**: European Space Agency (ESA) / European Union Copernicus Programme
* **Modality**: Multi-band optical (True Color RGB — Bands 4, 3, 2)
* **Spatial Resolution**: 10 meters / pixel (512 × 512 pixels)
* **Scene Content**: Dense multi-class Earth observation scene featuring urban commercial and residential fabric, agricultural crop parcels, transportation arteries, and Tiber river / water retention reservoir.
* **License**: Creative Commons Attribution-ShareAlike 3.0 IGO (CC BY-SA 3.0 IGO)
* **Attribution**: Contains modified Copernicus Sentinel data (2020), processed by ESA.

---

## 2. Demo B — Single-Image Spatial Grounding
* **File**: `demo_assets/demo_airport_grounding.png`
* **Sensor / Platform**: USGS High-Resolution Aerial Orthoimagery / NAIP
* **Agency / Provider**: United States Geological Survey (USGS) / National Geospatial Program
* **Modality**: High-resolution optical aerial photography (True Color RGB)
* **Spatial Resolution**: 0.5–1.0 meter / pixel (512 × 512 pixels)
* **Scene Content**: Airport airfield infrastructure showing primary runway strip, threshold markings, taxiways, aircraft parking apron, and tarmac.
* **License**: Public Domain (USGS / United States Government Work)
* **Attribution**: Imagery courtesy of the U.S. Geological Survey.

---

## 3. Demo C & Demo E — Bi-Temporal Change Intelligence & Multi-Tool Workflow
* **Files**: `demo_assets/demo_change_t0.png` (T0 Baseline) and `demo_assets/demo_change_t1.png` (T1 Follow-up)
* **Sensor / Platform**: High-Resolution Satellite Remote Sensing (Google Earth / WorldView)
* **Dataset**: LEVIR-CD (Large-scale Building Change Detection Dataset, Chen et al., 2020)
* **Provider / Authors**: Image Processing Center, Beihang University
* **Modality**: Bi-temporal high-resolution optical (RGB)
* **Spatial Resolution**: 0.5 meter / pixel (256 × 256 pixels)
* **Scene Content**: Real-world residential development and building construction showing baseline bare/vegetated parcel at acquisition T0 and newly constructed commercial/residential structures with access roads at acquisition T1.
* **License**: Academic Research Open Access (LEVIR-CD Dataset)
* **Attribution**: LEVIR-CD dataset created by Hao Chen et al., *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*, Remote Sensing 2020.

---

## 4. Demo D — Optical + SAR Cross-Modal Intelligence (Mock Specialist)
* **Files**: `demo_assets/demo_optical_cross.png` (Optical) and `demo_assets/demo_sar_cross.tif` (SAR Backscatter)
* **Sensor / Platform**: Copernicus Sentinel-2 Optical MSI + Sentinel-1 C-Band SAR (Synthetic Aperture Radar)
* **Agency / Provider**: European Space Agency (ESA) / Copernicus Programme
* **Modality**: Optical (RGB with cloud cover) + Synthetic Aperture Radar C-Band GRD Amplitude (GeoTIFF)
* **Spatial Resolution**: 10–20 meters / pixel (256 × 256 pixels)
* **Scene Content**: Partially clouded agricultural/urban scene with corresponding SAR amplitude backscatter showing radar cloud penetration and metallic structure scattering.
* **Status**: Demonstration / Mock specialist interface (Division 4 awaiting deep integration).
* **License**: CC BY-SA 3.0 IGO (Copernicus Open Access).
