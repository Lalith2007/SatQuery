"""Confidence and Uncertainty Engine for Division 4 Optical-SAR Specialist."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # Non-GUI headless backend for background tasks and testing
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.schemas import Artifact

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """Calculates pixel-level uncertainty, calibrated scalar confidence, and confidence heatmaps."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_confidence(
        self,
        probs: np.ndarray,  # Shape (K, H, W)
        request_id: str,
    ) -> Tuple[float, np.ndarray, Artifact]:
        """Compute pixel prediction margin, normalized entropy map, global scalar confidence score, and save heatmap artifact.
        
        Returns:
            Tuple of (global_confidence_score, confidence_map, artifact_obj)
        """
        k, h, w = probs.shape

        # 1. Top-1 vs Top-2 Probability Margin
        sorted_probs = np.sort(probs, axis=0)  # Ascending order
        margin_map = sorted_probs[-1] - sorted_probs[-2]  # (H, W) in [0.0, 1.0]

        # 2. Normalized Pixel Entropy Map
        eps = 1e-7
        probs_clamped = np.clip(probs, eps, 1.0 - eps)
        entropy_map = -np.sum(probs_clamped * np.log(probs_clamped), axis=0) / np.log(float(k))
        uncertainty_map = np.clip(entropy_map, 0.0, 1.0)  # 0 = high certainty, 1 = high uncertainty

        # 3. Combined Confidence Map
        confidence_map = margin_map * (1.0 - uncertainty_map)  # Range [0.0, 1.0]

        # 4. Global Scalar Confidence
        global_confidence = float(np.mean(confidence_map))
        global_confidence = round(max(0.1, min(0.99, global_confidence)), 4)

        # 5. Render & Save Confidence Heatmap Artifact
        heatmap_filename = f"{request_id}_confidence_heatmap.png"
        heatmap_path = self.output_dir / heatmap_filename

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(confidence_map, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Confidence (Margin x (1 - Entropy))")
        plt.title("Spatial Confidence Map", fontsize=10)
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        artifact = Artifact(
            name=heatmap_filename,
            type="confidence_heatmap",
            uri_or_path=str(heatmap_path),
            description=f"Pixel-level confidence heatmap (Global Confidence: {global_confidence:.3f})",
            mime_type="image/png",
        )

        return global_confidence, confidence_map, artifact
