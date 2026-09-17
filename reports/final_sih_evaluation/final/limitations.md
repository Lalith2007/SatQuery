# SatQuery AI — System Limitations & Deployment Boundaries (SIH26167)

## 1. Spaceborne ISRO/SAC Mission Ingestion Boundaries
- **Evaluation Status**: `STATUS = AWAITING OFFICIAL EVALUATION DATA`
- **Integrity Compliance**: SatQuery AI strictly refrains from fabricating synthetic ISRO/SAC evaluation data or substituting public benchmark numbers (such as LEVIR-CD, WHU-OPT-SAR, or BigEarthNet) for official ISRO/SAC mission evaluations.
- **Certified Ingestion**: The certified ingestion interface is implemented at `evaluation/benchmarks/isro_sac.py` (`ISROSACGenericEvaluator`). When official Cartosat-3 or RISAT-1A datasets are provided by the jury, they will be ingested without pipeline modification.

## 2. Cloud and Atmospheric Occlusion in Optical Modality
- Heavy cumulus and cirrus cloud cover exceeding 40% degrades single-optical visual grounding and fine-grained categorization.
- **Mitigation**: The SatQuery Agentic Controller automatically detects optical obstruction and routes the query to the SAR specialist (`cmaf`) or triggers cross-modal fusion to leverage cloud-penetrating Sentinel-1 dual-polarization radar.

## 3. Spatial Resolution and Sub-Pixel Features
- Grounding and referring expression resolution on Sentinel-2 optical imagery (10m GSD) is limited to features spanning at least 20m x 20m (2x2 pixel footprint).
- Fine urban structures below 5m require high-resolution aerial or commercial constellations (e.g., Cartosat panchromatic 0.28m, WorldView, or LEVIR-CD 0.5m).

## 4. Bi-Temporal Alignment and Coregistration
- Change detection via TinyCD assumes orthorectified and coregistered image pairs. Coregistration errors exceeding 1 pixel (0.5m in LEVIR-CD) can induce false positive edge activations, which are mitigated by the deterministic region filter and bounding box aggregation before VLM reasoning.
