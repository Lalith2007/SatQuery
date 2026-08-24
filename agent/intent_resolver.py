"""Natural-language query interpretation and TaskIntent resolution.

Decouples query semantic understanding from tool routing.
Provides a fast, deterministic, offline-capable rule engine and an extensible
LLM-adapter interface without exposing private chain-of-thought.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.logging import get_logger
from core.schemas import ImageInput, ImageModality, TaskIntent, TaskType

logger = get_logger("intent_resolver")


class IntentResolver:
    """Resolves natural language queries and image configurations into structured TaskIntent."""

    # Regex / semantic patterns for deterministic intent resolution
    _CROSS_MODAL_PATTERNS = [
        r"\b(optical\s+and\s+sar|sar\s+and\s+optical|cross[- ]modal|radar\s+and\s+optical|optical\s+and\s+radar)\b",
        r"\b(fuse|fusion|joint\s+analysis|penetrate\s+clouds?)\b",
    ]

    _CHANGE_PATTERNS = [
        r"\b(change|changed|difference|differences|diff)\b",
        r"\b(between\s+(these\s+)?(dates|images|acquisitions|t0\s+and\s+t1|periods))\b",
        r"\b(increased|decreased|expanded|expansion|shrinkage|growth|loss|deforestation)\b",
        r"\b(before\s+and\s+after|temporal\s+analysis)\b",
    ]

    _GROUNDING_PATTERNS = [
        r"\b(where\s+is|where\s+are|locate|localization|highlight|segment|segmentation|find\s+the|bounding\s*box)\b",
        r"\b(ground\s+the|pinpoint|detect\s+location)\b",
    ]

    _CAPTION_PATTERNS = [
        r"\b(describe(\s+this)?|caption|generate\s+a\s+caption|scene\s+summary|summarize\s+the\s+scene|overview\s+of\s+this)\b",
        r"\b(what\s+does\s+this\s+scene\s+depict)\b",
    ]

    @classmethod
    def resolve_intent(
        cls,
        query: str,
        images: Optional[List[ImageInput]] = None,
        task_hint: Optional[TaskType] = None,
    ) -> TaskIntent:
        """Resolve query into structured TaskIntent."""
        # 1. Manual Task Hint Override
        if task_hint:
            logger.info(f"Using explicit task hint: {task_hint.value}")
            return TaskIntent(
                task=task_hint,
                confidence=1.0,
                intent_explanation=f"Explicit task hint '{task_hint.value}' provided by caller.",
                target_features=cls._extract_target_features(query),
            )

        query_cleaned = query.strip().lower()
        num_images = len(images) if images else 1
        modalities = [img.modality for img in images] if images else []

        # 2. Check for Optical-SAR Cross-Modal intent
        has_optical_and_sar_images = (
            ImageModality.SAR in modalities and
            any(m in {ImageModality.OPTICAL, ImageModality.MULTISPECTRAL} for m in modalities)
        )
        is_cross_modal_query = any(re.search(p, query_cleaned) for p in cls._CROSS_MODAL_PATTERNS)

        if is_cross_modal_query or (has_optical_and_sar_images and num_images == 2):
            return TaskIntent(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                confidence=0.96 if is_cross_modal_query else 0.88,
                intent_explanation="Query or image pair specifies joint optical-SAR cross-modal analysis.",
                target_features=cls._extract_target_features(query),
            )

        # 3. Check for Bi-Temporal Change intent
        is_change_query = any(re.search(p, query_cleaned) for p in cls._CHANGE_PATTERNS)
        if is_change_query or (num_images == 2 and not has_optical_and_sar_images):
            # Differentiate between general change analysis vs specific change-VQA question
            is_question = query_cleaned.endswith("?") or any(
                query_cleaned.startswith(w) for w in ["what", "how", "has", "did", "is", "can", "why"]
            )
            resolved_task = TaskType.CHANGE_VQA if is_question else TaskType.CHANGE_ANALYSIS
            return TaskIntent(
                task=resolved_task,
                confidence=0.95 if is_change_query else 0.85,
                intent_explanation="Query requests temporal change detection / comparison between acquisitions.",
                target_features=cls._extract_target_features(query),
            )

        # 4. Check for Single Image Grounding / Localization
        is_grounding_query = any(re.search(p, query_cleaned) for p in cls._GROUNDING_PATTERNS)
        if is_grounding_query:
            return TaskIntent(
                task=TaskType.SINGLE_IMAGE_GROUNDING,
                confidence=0.94,
                intent_explanation="Query requests spatial localization/grounding of specific remote sensing features.",
                target_features=cls._extract_target_features(query),
            )

        # 5. Check for Single Image Captioning
        is_caption_query = any(re.search(p, query_cleaned) for p in cls._CAPTION_PATTERNS)
        if is_caption_query:
            return TaskIntent(
                task=TaskType.SINGLE_IMAGE_CAPTION,
                confidence=0.93,
                intent_explanation="Query requests comprehensive scene description or image captioning.",
                target_features=cls._extract_target_features(query),
            )

        # 6. Default to Single Image VQA for all standard questions / queries
        return TaskIntent(
            task=TaskType.SINGLE_IMAGE_VQA,
            confidence=0.90,
            intent_explanation="Query resolved as visual question answering on single remote sensing image.",
            target_features=cls._extract_target_features(query),
        )

    @classmethod
    def _extract_target_features(cls, query: str) -> List[str]:
        """Extract candidate target entities or keywords from query."""
        keywords = [
            "runway", "airport", "aircraft", "plane", "water", "river", "lake", "reservoir",
            "building", "urban", "construction", "road", "vegetation", "forest", "tree",
            "cloud", "ship", "vessel", "solar panel", "agricultural", "field", "storage tank"
        ]
        found = []
        q_lower = query.lower()
        for kw in keywords:
            if kw in q_lower:
                found.append(kw)
        return found
