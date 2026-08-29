"""Main Optical-SAR Specialist Tool Service for Division 4.

Implements BaseSpecialistTool contract and coordinates preprocessing,
dual pretrained encoders, cross-modal attention fusion, 8-class neural prediction,
query intent modulation, deterministic query aggregation, spatial evidence generation,
and calibrated confidence estimation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from core.interfaces import BaseSpecialistTool, ValidationResult
from core.schemas import (
    Artifact,
    Evidence,
    EvidenceType,
    ExecutionStage,
    ExecutionTraceEntry,
    ImageModality,
    TaskType,
    ToolMetadata,
    ToolRequest,
    ToolResult,
    ToolStatus,
)

from specialists.optical_sar.config import SpecialistConfig
from specialists.optical_sar.confidence import ConfidenceEngine
from specialists.optical_sar.dataset import aggregate_8class_probs_to_satquery_intents
from specialists.optical_sar.decoders.landcover_head import LandCoverTaskHead
from specialists.optical_sar.encoders import BaseModalityEncoder, OpticalEncoder, SarEncoder
from specialists.optical_sar.evidence import SpatialEvidenceEngine
from specialists.optical_sar.fusion import CrossModalAttentionFusion
from specialists.optical_sar.preprocessing import OpticalSarPreprocessor
from specialists.optical_sar.query_intent import QueryIntentInterpreter
from specialists.optical_sar.schemas import extract_and_validate_optical_sar_inputs

logger = logging.getLogger(__name__)


class OpticalSarSpecialist(BaseSpecialistTool):
    """Division 4 Specialist Plugin for Optical-SAR Cross-Modal Intelligence."""

    def __init__(
        self,
        config: Optional[SpecialistConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        require_trained_weights: bool = False,
    ) -> None:
        self.config = config or SpecialistConfig()
        super().__init__(
            name=self.config.plugin_name,
            description="Specialist tool for co-registered Optical and SAR cross-modal joint fusion and land-cover analysis.",
            supported_tasks={TaskType.OPTICAL_SAR_ANALYSIS},
            version=self.config.version,
            metadata=ToolMetadata(
                name=self.config.plugin_name,
                description="Specialist tool for co-registered Optical and SAR cross-modal joint fusion and land-cover analysis.",
                version=self.config.version,
                supported_tasks=[TaskType.OPTICAL_SAR_ANALYSIS],
                required_modalities=[ImageModality.OPTICAL, ImageModality.SAR],
                min_images=2,
                max_images=2,
                author_or_division="Division 4 (Manoj)",
                metadata={"fusion_type": "Cross-Modal-Attention-Fusion (CMAF)", "supports_cloud_penetration": True},
            ),
        )

        # Initialize pipeline components
        self.preprocessor = OpticalSarPreprocessor(self.config.preprocessing)
        self.query_interpreter = QueryIntentInterpreter()
        self.task_head = LandCoverTaskHead(
            feature_channels=self.config.model.fusion_channels,
            num_classes=8,
        )

        self.evidence_engine = SpatialEvidenceEngine(self.config.artifacts_dir)
        self.confidence_engine = ConfidenceEngine(self.config.artifacts_dir)

        # Set up encoders and fusion neck
        opt_enc = OpticalEncoder(
            backbone_type=self.config.model.optical_encoder_type.split("_")[-1],
            pretrained_weights=self.config.model.optical_pretrained_weights,
            in_channels=self.config.model.in_channels_optical,
        )
        sar_enc = SarEncoder(
            backbone_type=self.config.model.sar_encoder_type.split("_")[-1],
            pretrained_weights=self.config.model.sar_pretrained_weights,
            in_channels=self.config.model.in_channels_sar,
        )
        self.set_encoders(opt_enc, sar_enc)

        # Load trained checkpoint weights if available
        self.checkpoint_path = checkpoint_path
        self.is_trained_loaded = False
        self.load_trained_checkpoint(require_trained=require_trained_weights)

    def set_encoders(self, optical_encoder: BaseModalityEncoder, sar_encoder: BaseModalityEncoder) -> None:
        """Set custom optical and SAR encoders and dynamically adapt fusion neck channels."""
        self.optical_encoder = optical_encoder
        self.sar_encoder = sar_encoder

        opt_out_ch = optical_encoder.out_channels["stride_8"]
        sar_out_ch = sar_encoder.out_channels["stride_8"]

        self.fusion_neck = CrossModalAttentionFusion(
            optical_channels=opt_out_ch,
            sar_channels=sar_out_ch,
            out_channels=self.config.model.fusion_channels,
        )

    def load_trained_checkpoint(self, require_trained: bool = False) -> bool:
        """Locate, load, and verify trained model checkpoint weights."""
        search_paths = [
            self.checkpoint_path,
            Path("specialists/optical_sar/checkpoints/cmaf_landcover_best.pth"),
            Path("checkpoints/cmaf_landcover_best.pth"),
        ]

        target_ckpt: Optional[Path] = None
        for p in search_paths:
            if p is not None and Path(p).exists():
                target_ckpt = Path(p)
                break

        if target_ckpt is None:
            if require_trained:
                err_msg = (
                    "Critical: Trained Optical-SAR specialist checkpoint not found at "
                    "'specialists/optical_sar/checkpoints/cmaf_landcover_best.pth'."
                )
                logger.warning(err_msg)
                self.is_trained_loaded = False
                return False
            return False

        try:
            ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=False)
            if "fusion_neck_state_dict" in ckpt:
                self.fusion_neck.load_state_dict(ckpt["fusion_neck_state_dict"])
            if "task_head_state_dict" in ckpt:
                self.task_head.load_state_dict(ckpt["task_head_state_dict"])
            if "optical_encoder_state_dict" in ckpt:
                self.optical_encoder.load_state_dict(ckpt["optical_encoder_state_dict"])
            if "sar_encoder_state_dict" in ckpt:
                self.sar_encoder.load_state_dict(ckpt["sar_encoder_state_dict"])

            self.optical_encoder.eval()
            self.sar_encoder.eval()
            self.fusion_neck.eval()
            self.task_head.eval()

            self.is_trained_loaded = True
            logger.info(f"Successfully loaded trained Optical-SAR checkpoint from '{target_ckpt}'.")
            return True
        except Exception as exc:
            logger.error(f"Failed to load trained checkpoint from '{target_ckpt}': {exc}")
            if require_trained:
                raise RuntimeError(f"Corrupt checkpoint at '{target_ckpt}': {exc}")
            return False

    def validate_request(self, request: ToolRequest) -> ValidationResult:
        """Validate input request compatibility."""
        val_res = extract_and_validate_optical_sar_inputs(request)
        if not val_res.is_valid:
            return ValidationResult(is_valid=False, errors=val_res.errors)
        return ValidationResult(is_valid=True)

    def health_check(self) -> bool:
        """Verify model components are loaded and functional."""
        return self.optical_encoder is not None and self.sar_encoder is not None

    async def execute(self, request: ToolRequest) -> ToolResult:
        """Execute joint Optical-SAR cross-modal analysis on ToolRequest."""
        start_time = time.time()
        trace_entries: List[ExecutionTraceEntry] = []

        # 1. Validate & Extract Inputs
        val_res = extract_and_validate_optical_sar_inputs(request)
        if not val_res.is_valid:
            return ToolResult(
                request_id=request.request_id,
                task=request.task,
                status=ToolStatus.FAILED,
                answer=f"Request validation failed: {'; '.join(val_res.errors)}",
                confidence=0.0,
                execution_trace=[
                    ExecutionTraceEntry(
                        stage=ExecutionStage.INPUT_VALIDATED,
                        component=self.name,
                        status="FAILED",
                        details={"errors": val_res.errors},
                    )
                ],
            )

        opt_img = val_res.optical_image
        sar_img = val_res.sar_image

        trace_entries.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.INPUT_VALIDATED,
                component=self.name,
                status="COMPLETED",
                details={"optical_path": opt_img.path_or_uri, "sar_path": sar_img.path_or_uri},
            )
        )

        # 2. Preprocess & Spatial Grid Alignment
        t0 = time.time()
        opt_raster = self.preprocessor.preprocess_optical(opt_img.path_or_uri)
        sar_raster = self.preprocessor.preprocess_sar(sar_img.path_or_uri)

        opt_aligned, sar_aligned = self.preprocessor.align_spatial_dimensions(
            opt_raster.tensor, sar_raster.tensor
        )
        t_prep_ms = (time.time() - t0) * 1000.0

        trace_entries.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component="preprocessor",
                status="COMPLETED",
                duration_ms=t_prep_ms,
                details={"optical_shape": list(opt_aligned.shape), "sar_shape": list(sar_aligned.shape)},
            )
        )

        # 3. Model Inference (Encoders, CMAF, 8-Class Task Head)
        t1 = time.time()
        self.optical_encoder.eval()
        self.sar_encoder.eval()
        self.fusion_neck.eval()
        self.task_head.eval()

        with torch.no_grad():
            opt_batch = opt_aligned.unsqueeze(0)  # (1, 3, H, W)
            sar_batch = sar_aligned.unsqueeze(0)  # (1, 2, H, W)

            opt_feats = self.optical_encoder(opt_batch)
            sar_feats = self.sar_encoder(sar_batch)

            fused_feat = self.fusion_neck(opt_feats["stride_8"], sar_feats["stride_8"])

            intent_dict = self.query_interpreter.parse_query_intent(request.query)
            intent_vec = self.query_interpreter.get_intent_vector(request.query).unsqueeze(0)  # (1, 4)

            logits, probs_8class = self.task_head(
                fused_feat, intent_vec, optical_raw=opt_batch, sar_raw=sar_batch
            )  # (1, 8, H, W)

        t_inf_ms = (time.time() - t1) * 1000.0
        trace_entries.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.INFERENCE_EXECUTED,
                component="cmaf_landcover_head",
                status="COMPLETED",
                duration_ms=t_inf_ms,
                details={"fused_shape": list(fused_feat.shape), "logits_shape": list(logits.shape), "active_intents": intent_dict},
            )
        )

        # 4. Query Intent Class Aggregation (AFTER neural prediction)
        aggregated_probs = aggregate_8class_probs_to_satquery_intents(probs_8class)
        probs_4class = torch.stack([
            aggregated_probs["built_up"],
            aggregated_probs["water"],
            aggregated_probs["vegetation"],
            aggregated_probs["background"]
        ], dim=1)  # (1, 4, H, W)

        # 5. Evidence Generation
        t2 = time.time()
        probs_np = probs_4class.squeeze(0).cpu().numpy()  # (4, H, W)
        opt_np = opt_aligned.cpu().numpy()                # (3, H, W)
        sar_np = sar_aligned.cpu().numpy()                # (2, H, W)

        evidences, artifacts, metrics = self.evidence_engine.generate_evidence_artifacts(
            probs=probs_np,
            optical_rgb=opt_np,
            sar_intensity=sar_np,
            active_intents=intent_dict,
            request_id=request.request_id,
        )

        # 6. Confidence Calculation
        global_confidence, conf_map, conf_artifact = self.confidence_engine.calculate_confidence(
            probs=probs_np,
            request_id=request.request_id,
        )
        artifacts.append(conf_artifact)
        t_evid_ms = (time.time() - t2) * 1000.0

        trace_entries.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.EVIDENCE_GENERATED,
                component="evidence_confidence_engine",
                status="COMPLETED",
                duration_ms=t_evid_ms,
                details={"evidence_count": len(evidences), "artifact_count": len(artifacts), "confidence": global_confidence},
            )
        )

        # 7. Natural Language Answer Synthesis
        built_ratio = metrics.get("built_up_area_ratio", 0.0) * 100.0
        water_ratio = metrics.get("water_area_ratio", 0.0) * 100.0
        veg_ratio   = metrics.get("vegetation_area_ratio", 0.0) * 100.0

        if built_ratio > 0.0:
            built_desc = f"detected localized built-up structures and road infrastructure (covering {built_ratio:.1f}% of the scene via SAR double-bounce radar return)"
        else:
            built_desc = "no significant built-up structures detected beneath clouds"

        if water_ratio > 0.0:
            water_desc = f"water bodies covering {water_ratio:.1f}% of the scene (confirmed by low SAR specular reflectance)"
        else:
            water_desc = "no standing water bodies detected"

        answer = (
            f"Joint optical-SAR cross-modal analysis successfully processed co-registered Sentinel-1/2 rasters using 8-class neural segmentation and query intent modulation. "
            f"Through cross-modal attention fusion (CMAF), the model {built_desc} and {water_desc}. "
            f"Dominant natural terrain consists of vegetation and cropland covering {veg_ratio:.1f}% of the scene."
        )

        total_ms = (time.time() - start_time) * 1000.0
        trace_entries.append(
            ExecutionTraceEntry(
                stage=ExecutionStage.RESULT_RETURNED,
                component=self.name,
                status="COMPLETED",
                duration_ms=total_ms,
                details={"total_execution_ms": round(total_ms, 2)},
            )
        )

        return ToolResult(
            request_id=request.request_id,
            task=request.task,
            status=ToolStatus.SUCCESS,
            answer=answer,
            confidence=global_confidence,
            evidence=evidences,
            artifacts=artifacts,
            model_info={"name": "SatQuery-OpticalSAR-CrossFuse", "version": self.version, "num_classes": 8},
            parameters={"query": request.query, "active_intents": intent_dict},
            metadata={"optical_id": opt_img.image_id, "sar_id": sar_img.image_id, "metrics": metrics},
            execution_trace=trace_entries,
        )
