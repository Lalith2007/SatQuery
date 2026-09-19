"""Mock and Production ChangeModel adapters for Division 3.

MockChangeModel: Deterministic test backend (no model weights required).
ChangeFormerAdapter: Production adapter wrapping a lightweight Siamese
  difference network for binary change detection.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, List, Optional

import numpy as np

from core.logging import get_logger
from specialists.temporal_change.interfaces import (
    ChangeDetectionOutput,
    ChangedRegion,
    ChangeModel,
)

logger = get_logger("temporal_change.model_adapter")


class MockChangeModel(ChangeModel):
    """Deterministic mock change detection backend for testing.

    Produces a synthetic change map by computing per-pixel absolute
    difference between T0 and T1, then thresholding. This is NOT
    a trained model — it is explicitly labeled as a mock.
    """

    def __init__(self) -> None:
        self._ready = False

    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        self._device = device
        self._ready = True
        logger.info("MockChangeModel initialized (deterministic test backend)")

    def is_ready(self) -> bool:
        return self._ready

    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Generate a deterministic mock change map from pixel differences.

        This is explicitly a test artifact, not a trained model prediction.
        """
        start = time.perf_counter()

        # Ensure float and same shape
        t0_f = t0.astype(np.float32) if t0.dtype != np.float32 else t0
        t1_f = t1.astype(np.float32) if t1.dtype != np.float32 else t1

        # If images are (H, W, C), compute mean absolute difference across channels
        if t0_f.ndim == 3 and t1_f.ndim == 3:
            diff = np.mean(np.abs(t0_f - t1_f), axis=-1)
        elif t0_f.ndim == 2 and t1_f.ndim == 2:
            diff = np.abs(t0_f - t1_f)
        else:
            # Fallback: just use first two dims
            diff = np.abs(t0_f.reshape(t0_f.shape[:2]) - t1_f.reshape(t1_f.shape[:2]))

        # Normalize to [0, 1]
        dmax = diff.max()
        if dmax > 0:
            prob_map = diff / dmax
        else:
            prob_map = diff

        binary_map = (prob_map > 0.3).astype(np.uint8)  # Mock threshold

        elapsed = (time.perf_counter() - start) * 1000.0

        changed_pixels = int(binary_map.sum())
        total_pixels = int(binary_map.shape[0] * binary_map.shape[1])
        ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0

        # Mean confidence over changed pixels
        if changed_pixels > 0:
            change_conf = float(prob_map[binary_map > 0].mean())
        else:
            change_conf = 0.0

        return ChangeDetectionOutput(
            change_probability_map=prob_map.astype(np.float32),
            binary_change_map=binary_map,
            change_confidence=change_conf,
            changed_pixel_ratio=ratio,
            model_name=self.model_name,
            model_version=self.model_version,
            device_used=self._device,
            inference_time_ms=round(elapsed, 2),
            metadata={"is_mock": True, "mock_threshold": 0.3},
        )

    def cleanup(self) -> None:
        self._ready = False

    @property
    def model_name(self) -> str:
        return "MockChangeModel"

    @property
    def model_version(self) -> str:
        return "1.0.0-mock"


class ChangeFormerAdapter(ChangeModel):
    """Production change detector adapter for ChangeFormer.

    Requires a valid trained checkpoint. Untrained fallback models are
    strictly forbidden in production execution.
    """

    def __init__(self, checkpoint_path: str = "", architecture: str = "changeformer") -> None:
        self._checkpoint_path = checkpoint_path
        self._architecture = architecture
        self._model = None
        self._device = "cpu"
        self._ready = False
        self.provenance_status = "PENDING"
        self.checkpoint_sha256 = ""
        self.parameter_count = 0
        self.execution_mode = "production_blocked"
        self.architecture_name = f"ChangeFormer ({architecture})"

    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        self._device = device
        from specialists.temporal_change.errors import ChangeModelLoadError

        if not self._checkpoint_path or not os.path.isfile(self._checkpoint_path):
            self.provenance_status = "CHECKPOINT_INVALID"
            raise ChangeModelLoadError(
                f"ChangeFormer checkpoint path '{self._checkpoint_path}' is empty or file not found. "
                "Production execution strictly forbids silent fallback to untrained models. "
                "STATUS = CHECKPOINT_INVALID"
            )

        try:
            import torch
            actual_device = device
            if device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
            self._device = actual_device

            self._model = self._load_checkpoint(self._checkpoint_path, actual_device)
            logger.info(f"ChangeFormerAdapter loaded checkpoint from {self._checkpoint_path} on {actual_device}")
            self._ready = True
            self.provenance_status = "VERIFIED"
        except Exception as e:
            self.provenance_status = "CHECKPOINT_INVALID"
            logger.error(f"Failed to initialize ChangeFormerAdapter: {e}")
            if isinstance(e, ChangeModelLoadError):
                raise
            raise ChangeModelLoadError(
                f"Failed to load ChangeFormer checkpoint at '{self._checkpoint_path}': {e}. "
                "STATUS = CHECKPOINT_INVALID"
            )

    def _create_feature_diff_model(self, device: str) -> Any:
        """Untrained fallback models are strictly forbidden in production."""
        raise NotImplementedError(
            "Untrained SiameseFeatureDiff fallback is strictly forbidden in production. "
            "STATUS = CHECKPOINT_INVALID"
        )

    def _load_checkpoint(self, path: str, device: str) -> Any:
        """Load a checkpoint file. Fails closed if loading fails."""
        import torch
        from specialists.temporal_change.errors import ChangeModelLoadError
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
            raise ChangeModelLoadError(
                f"ChangeFormer model architecture is not currently supported for production inference. "
                "STATUS = CHECKPOINT_INVALID"
            )
        except Exception as e:
            raise ChangeModelLoadError(
                f"Checkpoint loading failed for ChangeFormer at '{path}': {e}. "
                "Production execution strictly forbids silent fallback to untrained models. "
                "STATUS = CHECKPOINT_INVALID"
            )

    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Run change detection using ChangeFormer."""
        from specialists.temporal_change.errors import ChangeModelLoadError
        if not self.is_ready():
            raise ChangeModelLoadError("ChangeFormerAdapter is not initialized with a valid checkpoint.")
        raise ChangeModelLoadError("ChangeFormer inference is not available in production.")

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def model_name(self) -> str:
        return f"ChangeDetector-{self._architecture}"

    @property
    def model_version(self) -> str:
        return "1.0.0"


class TinyCDAdapter(ChangeModel):
    """Production TinyCD Change Detection Adapter.

    Loads the trained TinyCD (Siamese U-Net + MAMB space-time attention) weights
    and executes neural change detection on bi-temporal image pairs.

    Enforces the Checkpoint Provenance Gate at startup:
    - Verifies checkpoint file exists
    - Computes and verifies SHA-256 matches expected production hash
    - Enforces strict state_dict loading (0 missing keys, 0 unexpected keys)
    - Verifies exact parameter count (3,565,034)
    - If any condition fails, sets provenance_status = 'CHECKPOINT_INVALID'
      and aborts startup.
    """

    EXPECTED_SHA256 = "b9a1009355865c0277d7b3266244a6d9864d0659cd279a1d8735f705ec3345d0"
    EXPECTED_PARAMS = 3565034

    def __init__(
        self,
        checkpoint_path: str = "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth",
        strict: bool = True,
        verify_provenance: bool = True,
        threshold: float = 0.50,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._strict = strict
        self._verify_provenance = verify_provenance
        self._threshold = threshold
        self._model = None
        self._device = "cpu"
        self._ready = False
        self.provenance_status = "PENDING"
        self.checkpoint_sha256 = ""
        self.parameter_count = 0
        self.execution_mode = "production_verified"
        self.architecture = "TinyCD"
        self.architecture_name = "TinyCD (Siamese U-Net + MAMB)"

    @property
    def checkpoint_path(self) -> str:
        return str(self._checkpoint_path)

    @property
    def threshold(self) -> float:
        return self._threshold

    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        self._device = device
        from specialists.temporal_change.errors import ChangeModelLoadError, CheckpointProvenanceError
        import torch
        from specialists.temporal_change.adaptation.models.tinycd import TinyCD

        if not self._checkpoint_path or not os.path.isfile(self._checkpoint_path):
            self.provenance_status = "CHECKPOINT_INVALID"
            raise ChangeModelLoadError(
                f"Trained TinyCD checkpoint not found at '{self._checkpoint_path}'. "
                "Production execution must fail closed; untrained fallback models are strictly forbidden. "
                "STATUS = CHECKPOINT_INVALID"
            )

        # Checkpoint SHA-256 Provenance Check
        hasher = hashlib.sha256()
        with open(self._checkpoint_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        computed_sha256 = hasher.hexdigest()
        self.checkpoint_sha256 = computed_sha256

        if self._verify_provenance and computed_sha256 != self.EXPECTED_SHA256:
            self.provenance_status = "CHECKPOINT_INVALID"
            raise CheckpointProvenanceError(
                f"TinyCD checkpoint SHA-256 mismatch! Found '{computed_sha256}', "
                f"expected authoritative production hash '{self.EXPECTED_SHA256}'. "
                "STATUS = CHECKPOINT_INVALID"
            )

        actual_device = device
        if device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            actual_device = "cpu"
        self._device = actual_device

        try:
            model = TinyCD(in_channels=3, base_features=32)
            state_dict = torch.load(self._checkpoint_path, map_location=actual_device, weights_only=True)
            load_res = model.load_state_dict(state_dict, strict=True)

            if len(load_res.missing_keys) > 0 or len(load_res.unexpected_keys) > 0:
                self.provenance_status = "CHECKPOINT_INVALID"
                raise CheckpointProvenanceError(
                    f"Strict checkpoint loading failed for TinyCD. "
                    f"Missing keys: {load_res.missing_keys}, Unexpected keys: {load_res.unexpected_keys}. "
                    "STATUS = CHECKPOINT_INVALID"
                )

            param_count = sum(p.numel() for p in model.parameters())
            self.parameter_count = param_count

            if self._verify_provenance and param_count != self.EXPECTED_PARAMS:
                self.provenance_status = "CHECKPOINT_INVALID"
                raise CheckpointProvenanceError(
                    f"TinyCD parameter count mismatch! Found {param_count}, "
                    f"expected {self.EXPECTED_PARAMS}. "
                    "STATUS = CHECKPOINT_INVALID"
                )

            model.to(actual_device)
            model.eval()
            self._model = model
            self._ready = True
            self.provenance_status = "VERIFIED"
            logger.info(
                f"TinyCDAdapter verified and loaded production checkpoint: {self._checkpoint_path} "
                f"(SHA256={computed_sha256[:16]}..., params={param_count:,}) on {actual_device}"
            )
        except Exception as e:
            self.provenance_status = "CHECKPOINT_INVALID"
            logger.error(f"Failed to initialize TinyCDAdapter: {e}")
            if isinstance(e, (ChangeModelLoadError, CheckpointProvenanceError)):
                raise
            raise ChangeModelLoadError(f"TinyCD initialization failed: {e}. STATUS = CHECKPOINT_INVALID")

    def is_ready(self) -> bool:
        return self._ready and self._model is not None and self.provenance_status == "VERIFIED"

    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Run change detection inference using the trained TinyCD network."""
        import torch
        from specialists.temporal_change.errors import ChangeModelLoadError

        if not self.is_ready():
            raise ChangeModelLoadError(
                f"TinyCDAdapter is not ready for inference (status={self.provenance_status}). "
                "STATUS = CHECKPOINT_INVALID"
            )

        start = time.perf_counter()

        # Prepare tensors: (H, W, C) -> (1, C, H, W)
        if t0.ndim == 3:
            t0_t = torch.from_numpy(t0).permute(2, 0, 1).unsqueeze(0).float()
            t1_t = torch.from_numpy(t1).permute(2, 0, 1).unsqueeze(0).float()
        else:
            t0_t = torch.from_numpy(t0).unsqueeze(0).unsqueeze(0).float()
            t1_t = torch.from_numpy(t1).unsqueeze(0).unsqueeze(0).float()

        t0_t = t0_t.to(self._device)
        t1_t = t1_t.to(self._device)

        with torch.no_grad():
            prob = self._model(t0_t, t1_t)  # (1, 1, H, W)

        prob_map = prob.squeeze().cpu().numpy().astype(np.float32)
        binary_map = (prob_map >= self._threshold).astype(np.uint8)

        elapsed = (time.perf_counter() - start) * 1000.0

        changed_pixels = int(binary_map.sum())
        total_pixels = int(binary_map.shape[0] * binary_map.shape[1])

        if changed_pixels > 0:
            change_conf = float(prob_map[binary_map > 0].mean())
        else:
            change_conf = 0.0

        return ChangeDetectionOutput(
            change_probability_map=prob_map,
            binary_change_map=binary_map,
            change_confidence=change_conf,
            changed_pixel_ratio=changed_pixels / total_pixels if total_pixels > 0 else 0.0,
            model_name=self.model_name,
            model_version=self.model_version,
            device_used=self._device,
            inference_time_ms=round(elapsed, 2),
            metadata={
                "is_mock": False,
                "architecture": self.architecture_name,
                "checkpoint": str(self._checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "parameter_count": self.parameter_count,
                "execution_mode": self.execution_mode,
                "provenance_status": self.provenance_status,
                "threshold": self._threshold,
            },
        )

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def model_name(self) -> str:
        return "ChangeDetector-TinyCD"

    @property
    def model_version(self) -> str:
        return "1.0.0-trained"

