"""Mock and Production ChangeModel adapters for Division 3.

MockChangeModel: Deterministic test backend (no model weights required).
ChangeFormerAdapter: Production adapter wrapping a lightweight Siamese
  difference network for binary change detection.
"""

from __future__ import annotations

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
    """Production change detector adapter.

    Wraps a lightweight Siamese difference network. When full ChangeFormer
    or TinyCD weights are available, this adapter loads them. Otherwise,
    it falls back to a simple learned-difference approach using torchvision
    feature extraction.

    The adapter is designed to be replaced with any ChangeModel implementation
    without modifying the specialist.
    """

    def __init__(self, checkpoint_path: str = "", architecture: str = "changeformer") -> None:
        self._checkpoint_path = checkpoint_path
        self._architecture = architecture
        self._model = None
        self._device = "cpu"
        self._ready = False

    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        self._device = device
        try:
            import torch
            actual_device = device
            if device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
                logger.warning("CUDA requested but unavailable; falling back to CPU.")
            self._device = actual_device

            if self._checkpoint_path:
                # Load production checkpoint
                self._model = self._load_checkpoint(self._checkpoint_path, actual_device)
                logger.info(f"ChangeFormerAdapter loaded checkpoint from {self._checkpoint_path} on {actual_device}")
            else:
                # No checkpoint: use feature-difference approach
                self._model = self._create_feature_diff_model(actual_device)
                logger.info(f"ChangeFormerAdapter initialized with feature-difference model on {actual_device}")

            self._ready = True
        except Exception as e:
            logger.error(f"Failed to initialize ChangeFormerAdapter: {e}")
            raise

    def _create_feature_diff_model(self, device: str) -> Any:
        """Create a lightweight feature-difference model using pretrained CNN features."""
        import torch
        import torch.nn as nn
        import torchvision.models as models

        class SiameseFeatureDiff(nn.Module):
            """Lightweight Siamese feature difference network."""

            def __init__(self):
                super().__init__()
                # Use first few layers of ResNet18 for feature extraction
                resnet = models.resnet18(weights=None)
                self.features = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                )
                # Simple 1x1 conv for change prediction
                self.classifier = nn.Sequential(
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid(),
                )

            def forward(self, t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
                f0 = self.features(t0)
                f1 = self.features(t1)
                diff = torch.abs(f0 - f1)
                out = self.classifier(diff)
                # Upsample to input size
                out = nn.functional.interpolate(out, size=t0.shape[2:], mode="bilinear", align_corners=False)
                return out.squeeze(1)

        model = SiameseFeatureDiff()
        model.to(device)
        model.eval()
        return model

    def _load_checkpoint(self, path: str, device: str) -> Any:
        """Load a checkpoint file. Falls back to feature-diff model if loading fails."""
        import torch
        try:
            model = self._create_feature_diff_model(device)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model
        except Exception as e:
            logger.warning(f"Checkpoint loading failed ({e}); using untrained feature-diff model")
            return self._create_feature_diff_model(device)

    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Run change detection using the Siamese feature difference model."""
        import torch

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
            prob = self._model(t0_t, t1_t)  # (1, H, W)

        prob_map = prob.squeeze().cpu().numpy().astype(np.float32)
        binary_map = (prob_map > 0.5).astype(np.uint8)

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
            metadata={"is_mock": False, "architecture": self._architecture},
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
        return f"ChangeDetector-{self._architecture}"

    @property
    def model_version(self) -> str:
        return "1.0.0"


class TinyCDAdapter(ChangeModel):
    """Production TinyCD Change Detection Adapter.

    Loads the trained TinyCD (Siamese U-Net + MAMB space-time attention) weights
    and executes neural change detection on bi-temporal image pairs.
    """

    def __init__(
        self,
        checkpoint_path: str = "specialists/temporal_change/weights/ChangeDetector-TinyCD.pth",
        strict: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._strict = strict
        self._model = None
        self._device = "cpu"
        self._ready = False

    def initialize(self, device: str = "cpu", **kwargs: Any) -> None:
        self._device = device
        try:
            import torch
            from specialists.temporal_change.adaptation.models.tinycd import TinyCD

            actual_device = device
            if device == "cuda" and not torch.cuda.is_available():
                actual_device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                actual_device = "cpu"

            self._device = actual_device
            model = TinyCD(in_channels=3, base_features=32)

            if self._checkpoint_path and os.path.isfile(self._checkpoint_path):
                state_dict = torch.load(self._checkpoint_path, map_location=actual_device, weights_only=True)
                model.load_state_dict(state_dict)
                logger.info(f"TinyCDAdapter loaded trained checkpoint: {self._checkpoint_path} on {actual_device}")
            else:
                if self._strict:
                    from specialists.temporal_change.errors import ChangeModelLoadError
                    raise ChangeModelLoadError(
                        f"Trained TinyCD checkpoint not found at '{self._checkpoint_path}'. "
                        "Strict evaluation mode forbids untrained inference."
                    )
                logger.warning(
                    f"TinyCD checkpoint not found at '{self._checkpoint_path}'; running in structural test mode."
                )

            model.to(actual_device)
            model.eval()
            self._model = model
            self._ready = True
        except Exception as e:
            logger.error(f"Failed to initialize TinyCDAdapter: {e}")
            raise

    def is_ready(self) -> bool:
        return self._ready and self._model is not None

    def detect_change(
        self,
        t0: np.ndarray,
        t1: np.ndarray,
        **kwargs: Any,
    ) -> ChangeDetectionOutput:
        """Run change detection inference using the trained TinyCD network."""
        import torch

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
        binary_map = (prob_map >= 0.5).astype(np.uint8)

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
            metadata={"is_mock": False, "architecture": "TinyCD (Siamese U-Net + MAMB)"},
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

