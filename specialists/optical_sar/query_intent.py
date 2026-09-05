"""Controlled query intent interpreter and FiLM modulation generator for Division 4."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class QueryIntentInterpreter:
    """Parses natural text queries into normalized target intent weights for 8 official WHU-OPT-SAR classes."""

    TARGET_CLASSES = [
        "background", # 0
        "farmland",   # 1
        "city",       # 2
        "village",    # 3
        "water",      # 4
        "forest",     # 5
        "road",       # 6
        "others",     # 7
    ]

    BACKGROUND_KEYWORDS = ["background"]
    FARMLAND_KEYWORDS = ["farmland", "farm", "crop", "crops", "agriculture", "field", "fields"]
    CITY_KEYWORDS = ["city", "urban", "building", "buildings", "structure", "structures", "footprint"]
    VILLAGE_KEYWORDS = ["village", "rural", "settlement", "house", "houses", "town"]
    BUILT_UP_GENERAL_KEYWORDS = ["built-up", "built up", "construction"]
    WATER_KEYWORDS = ["water", "river", "lake", "reservoir", "stream", "flood", "flooding", "sea", "ocean", "pond"]
    FOREST_KEYWORDS = ["forest", "tree", "trees", "woodland", "green", "vegetation", "canopy"]
    ROAD_KEYWORDS = ["road", "roads", "highway", "path", "street", "transportation"]
    OTHERS_KEYWORDS = ["others", "bare", "other"]

    def parse_query_intent(self, query: str) -> Dict[str, float]:
        """Parse query string and return class activation weights in range [0.0, 1.0]."""
        q_lower = query.lower()
        weights = {cls: 0.1 for cls in self.TARGET_CLASSES}

        matched_city = any(kw in q_lower for kw in self.CITY_KEYWORDS)
        matched_village = any(kw in q_lower for kw in self.VILLAGE_KEYWORDS)
        matched_built_general = any(kw in q_lower for kw in self.BUILT_UP_GENERAL_KEYWORDS)
        matched_water = any(kw in q_lower for kw in self.WATER_KEYWORDS)
        matched_farm = any(kw in q_lower for kw in self.FARMLAND_KEYWORDS)
        matched_forest = any(kw in q_lower for kw in self.FOREST_KEYWORDS)
        matched_road = any(kw in q_lower for kw in self.ROAD_KEYWORDS)

        if matched_city:
            weights["city"] = 1.0
        if matched_village:
            weights["village"] = 1.0
        if matched_built_general and not (matched_city or matched_village or matched_road):
            weights["city"] = 1.0
            weights["village"] = 0.8
            weights["road"] = 0.5
        if matched_water:
            weights["water"] = 1.0
        if matched_farm:
            weights["farmland"] = 1.0
        if matched_forest:
            weights["forest"] = 1.0
        if matched_road:
            weights["road"] = 1.0

        # General land-cover fallback if no specific target keyword matched
        if not any([matched_city, matched_village, matched_built_general, matched_water, matched_farm, matched_forest, matched_road]):
            weights["city"] = 1.0
            weights["village"] = 0.8
            weights["water"] = 1.0
            weights["farmland"] = 0.5
            weights["forest"] = 0.5
            weights["road"] = 0.5

        # Convenience aggregated keys for downstream NLG synthesis
        weights["built_up"] = max(weights["city"], weights["village"], weights["road"])
        weights["vegetation"] = max(weights["farmland"], weights["forest"])

        return weights

    def get_intent_vector(self, query: str) -> torch.Tensor:
        """Return float tensor of intent weights shape (8,) matching 8 official WHU-OPT-SAR classes."""
        dict_w = self.parse_query_intent(query)
        return torch.tensor([dict_w[c] for c in self.TARGET_CLASSES], dtype=torch.float32)


class FiLMQueryModulator(nn.Module):
    """Feature-wise Linear Modulation (FiLM) driven by 8-class query intent vectors."""

    def __init__(self, feature_channels: int, num_classes: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fc_gamma = nn.Linear(num_classes, feature_channels)
        self.fc_beta  = nn.Linear(num_classes, feature_channels)

        # Initialize to identity modulation (gamma=1, beta=0)
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, feature_map: torch.Tensor, intent_vec: torch.Tensor) -> torch.Tensor:
        """Modulate feature map using gamma * feature + beta.
        
        Args:
            feature_map: Tensor shape (B, C, H, W)
            intent_vec: Tensor shape (B, 8) or (8,)
        """
        if intent_vec.dim() == 1:
            intent_vec = intent_vec.unsqueeze(0)

        gamma = self.fc_gamma(intent_vec).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = self.fc_beta(intent_vec).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        return gamma * feature_map + beta
