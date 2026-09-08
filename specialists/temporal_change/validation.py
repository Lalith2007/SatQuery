"""6-Point Pair Validation for Division 3: Bi-Temporal Change Intelligence.

Validation stages (each returns structured errors):
  1. Image count — exactly N=2
  2. File readability — both paths exist and can be opened
  3. Dimension/grid — width, height, channels, resolution
  4. CRS — coordinate reference system compatibility
  5. Spatial compatibility — bounds overlap, transform, registration
  6. Timestamp — distinct temporal observations
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger
from core.schemas import ImageInput, ToolRequest

from specialists.temporal_change.errors import (
    GeospatialMismatchError,
    ImageReadError,
    IncompatibleDimensionsError,
    TemporalValidationError,
)

logger = get_logger("temporal_change.validation")


@dataclass
class ValidationStageResult:
    """Result of a single validation stage."""
    stage: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairValidationResult:
    """Aggregate result of all 6 validation stages."""
    is_valid: bool
    stages: List[ValidationStageResult] = field(default_factory=list)
    t0_info: Dict[str, Any] = field(default_factory=dict)
    t1_info: Dict[str, Any] = field(default_factory=dict)
    alignment_status: str = "unknown"  # "aligned", "alignable", "incompatible"

    @property
    def errors(self) -> List[str]:
        return [s.message for s in self.stages if not s.passed]


def validate_image_count(request: ToolRequest) -> ValidationStageResult:
    """Stage 1: Verify exactly 2 images are provided."""
    count = len(request.images)
    if count != 2:
        return ValidationStageResult(
            stage="image_count",
            passed=False,
            message=f"Bi-temporal analysis requires exactly 2 images, received {count}.",
            details={"received": count, "required": 2},
        )
    return ValidationStageResult(stage="image_count", passed=True)


def validate_file_readability(images: List[ImageInput]) -> ValidationStageResult:
    """Stage 2: Verify both image files exist and are readable."""
    errors = []
    for i, img in enumerate(images):
        path = img.path_or_uri
        if not os.path.isfile(path):
            errors.append(f"Image {i} (T{i}) path does not exist: {path}")
        elif not os.access(path, os.R_OK):
            errors.append(f"Image {i} (T{i}) is not readable: {path}")
        else:
            # Attempt to read the first few bytes
            try:
                with open(path, "rb") as f:
                    header = f.read(16)
                    if len(header) < 8:
                        errors.append(f"Image {i} (T{i}) file is too small to be a valid raster: {path}")
            except OSError as e:
                errors.append(f"Image {i} (T{i}) read error: {e}")

    if errors:
        return ValidationStageResult(
            stage="file_readability",
            passed=False,
            message="; ".join(errors),
            details={"errors": errors},
        )
    return ValidationStageResult(stage="file_readability", passed=True)


def _extract_image_info(img: ImageInput) -> Dict[str, Any]:
    """Extract dimension/metadata info from an ImageInput, using metadata fields."""
    info: Dict[str, Any] = {
        "image_id": img.image_id,
        "width": img.width,
        "height": img.height,
        "channels": img.channel_count,
        "dtype": img.dtype,
        "format": img.format.value if img.format else None,
        "modality": img.modality.value if img.modality else None,
    }
    if img.geospatial:
        geo = img.geospatial
        info["crs"] = geo.crs
        info["geo_bounds"] = geo.geo_bounds
        info["resolution"] = geo.resolution
        info["acquisition_timestamp"] = (
            geo.acquisition_timestamp.isoformat() if geo.acquisition_timestamp else None
        )
        info["sensor"] = geo.sensor
    return info


def validate_dimensions(t0: ImageInput, t1: ImageInput) -> ValidationStageResult:
    """Stage 3: Inspect width, height, channels, resolution, grid compatibility.

    Dimension mismatch alone does NOT trigger an error — it records the mismatch
    for later alignment decision. Only channel count incompatibility is a hard failure.
    """
    details: Dict[str, Any] = {}
    issues: List[str] = []

    # Channel count must be compatible
    if t0.channel_count and t1.channel_count and t0.channel_count != t1.channel_count:
        issues.append(
            f"Channel count mismatch: T0 has {t0.channel_count}, T1 has {t1.channel_count}."
        )
        details["channel_mismatch"] = True

    # Record dimension info (mismatch is informational, not a failure by itself)
    dims_match = (t0.width == t1.width) and (t0.height == t1.height)
    details["dimensions_match"] = dims_match
    if not dims_match:
        details["t0_dims"] = f"{t0.width}x{t0.height}"
        details["t1_dims"] = f"{t1.width}x{t1.height}"

    # Resolution info if available
    if t0.geospatial and t0.geospatial.resolution and t1.geospatial and t1.geospatial.resolution:
        details["t0_resolution"] = t0.geospatial.resolution
        details["t1_resolution"] = t1.geospatial.resolution

    if issues:
        return ValidationStageResult(
            stage="dimension_grid",
            passed=False,
            message="; ".join(issues),
            details=details,
        )
    return ValidationStageResult(stage="dimension_grid", passed=True, details=details)


def validate_crs(t0: ImageInput, t1: ImageInput) -> ValidationStageResult:
    """Stage 4: Compare CRS if available."""
    t0_crs = t0.geospatial.crs if t0.geospatial else None
    t1_crs = t1.geospatial.crs if t1.geospatial else None
    details: Dict[str, Any] = {"t0_crs": t0_crs, "t1_crs": t1_crs}

    if t0_crs is None and t1_crs is None:
        return ValidationStageResult(
            stage="crs",
            passed=True,
            message="No CRS metadata available for either image; skipping CRS check.",
            details=details,
        )

    if t0_crs is None or t1_crs is None:
        return ValidationStageResult(
            stage="crs",
            passed=True,
            message=f"CRS available for only one image (T0={t0_crs}, T1={t1_crs}); cannot validate.",
            details=details,
        )

    if t0_crs != t1_crs:
        return ValidationStageResult(
            stage="crs",
            passed=False,
            message=f"CRS mismatch: T0={t0_crs}, T1={t1_crs}. Reprojection may be required.",
            details=details,
        )

    return ValidationStageResult(stage="crs", passed=True, details=details)


def validate_spatial_compatibility(
    t0: ImageInput, t1: ImageInput, allow_reprojection: bool = False
) -> Tuple[ValidationStageResult, str]:
    """Stage 5: Inspect bounds overlap, transform, and spatial registration.

    Returns:
        Tuple of (validation result, alignment_status) where alignment_status
        is one of: "aligned", "alignable", "incompatible".
    """
    t0_geo = t0.geospatial
    t1_geo = t1.geospatial

    # No geospatial metadata at all
    if not t0_geo and not t1_geo:
        dims_match = (t0.width == t1.width) and (t0.height == t1.height)
        if dims_match:
            return (
                ValidationStageResult(
                    stage="spatial_compatibility",
                    passed=True,
                    message="No geospatial metadata; dimensions match, assuming co-registered.",
                    details={"assumption": "co-registered_by_dimension_match"},
                ),
                "aligned",
            )
        else:
            return (
                ValidationStageResult(
                    stage="spatial_compatibility",
                    passed=False,
                    message=(
                        f"Dimension mismatch (T0={t0.width}x{t0.height}, "
                        f"T1={t1.width}x{t1.height}) without geospatial metadata. "
                        "Cannot determine spatial relationship."
                    ),
                    details={"t0_dims": f"{t0.width}x{t0.height}", "t1_dims": f"{t1.width}x{t1.height}"},
                ),
                "incompatible",
            )

    # Bounds overlap check
    t0_bounds = t0_geo.geo_bounds if t0_geo else None
    t1_bounds = t1_geo.geo_bounds if t1_geo else None

    if t0_bounds and t1_bounds and len(t0_bounds) == 4 and len(t1_bounds) == 4:
        # [minx, miny, maxx, maxy]
        overlap_x = max(t0_bounds[0], t1_bounds[0]) < min(t0_bounds[2], t1_bounds[2])
        overlap_y = max(t0_bounds[1], t1_bounds[1]) < min(t0_bounds[3], t1_bounds[3])
        if not (overlap_x and overlap_y):
            return (
                ValidationStageResult(
                    stage="spatial_compatibility",
                    passed=False,
                    message="No spatial overlap between T0 and T1 bounding boxes.",
                    details={"t0_bounds": t0_bounds, "t1_bounds": t1_bounds},
                ),
                "incompatible",
            )

        # Check if bounds are identical (strongly co-registered)
        bounds_match = t0_bounds == t1_bounds
        dims_match = (t0.width == t1.width) and (t0.height == t1.height)
        crs_match = (t0_geo and t1_geo and t0_geo.crs == t1_geo.crs) if (t0_geo and t1_geo) else False

        if bounds_match and dims_match and crs_match:
            return (
                ValidationStageResult(
                    stage="spatial_compatibility",
                    passed=True,
                    message="Images are spatially co-registered (matching CRS, bounds, and dimensions).",
                    details={"bounds_match": True, "dims_match": True, "crs_match": True},
                ),
                "aligned",
            )

        if crs_match and bounds_match and not dims_match:
            if allow_reprojection:
                return (
                    ValidationStageResult(
                        stage="spatial_compatibility",
                        passed=True,
                        message="Same CRS and bounds but different grid. Controlled resampling authorized.",
                        details={"alignment_required": True, "type": "grid_resampling"},
                    ),
                    "alignable",
                )
            else:
                return (
                    ValidationStageResult(
                        stage="spatial_compatibility",
                        passed=False,
                        message=(
                            "Same CRS and bounds but different grid dimensions. "
                            "Set allow_geospatial_reprojection=True to enable controlled resampling."
                        ),
                    ),
                    "incompatible",
                )

        # Partial overlap with same CRS
        if crs_match:
            if allow_reprojection:
                return (
                    ValidationStageResult(
                        stage="spatial_compatibility",
                        passed=True,
                        message="Overlapping bounds with same CRS. Controlled alignment authorized.",
                        details={"alignment_required": True, "type": "bounds_alignment"},
                    ),
                    "alignable",
                )

    # Dimensions match, reasonable assumption of co-registration
    if (t0.width == t1.width) and (t0.height == t1.height):
        return (
            ValidationStageResult(
                stage="spatial_compatibility",
                passed=True,
                message="Dimensions match; assuming spatially co-registered.",
                details={"assumption": "co-registered_by_dimension_match"},
            ),
            "aligned",
        )

    # Fallback: incompatible
    return (
        ValidationStageResult(
            stage="spatial_compatibility",
            passed=False,
            message="Cannot establish spatial compatibility between T0 and T1.",
        ),
        "incompatible",
    )


def validate_timestamps(t0: ImageInput, t1: ImageInput) -> ValidationStageResult:
    """Stage 6: Verify acquisition timestamps represent distinct temporal observations."""
    t0_ts = t0.geospatial.acquisition_timestamp if t0.geospatial else None
    t1_ts = t1.geospatial.acquisition_timestamp if t1.geospatial else None

    if t0_ts is None and t1_ts is None:
        return ValidationStageResult(
            stage="timestamp",
            passed=True,
            message="No acquisition timestamps available; skipping temporal validation.",
        )

    if t0_ts and t1_ts:
        if t0_ts == t1_ts:
            return ValidationStageResult(
                stage="timestamp",
                passed=False,
                message="T0 and T1 have identical acquisition timestamps; expected different dates.",
                details={"t0_timestamp": t0_ts.isoformat(), "t1_timestamp": t1_ts.isoformat()},
            )
        return ValidationStageResult(
            stage="timestamp",
            passed=True,
            details={
                "t0_timestamp": t0_ts.isoformat(),
                "t1_timestamp": t1_ts.isoformat(),
                "temporal_gap_days": abs((t1_ts - t0_ts).days),
            },
        )

    return ValidationStageResult(
        stage="timestamp",
        passed=True,
        message="Timestamp available for only one image; cannot fully validate temporal relationship.",
    )


def validate_pair(
    request: ToolRequest,
    allow_reprojection: bool = False,
) -> PairValidationResult:
    """Run all 6 validation stages on a bi-temporal image pair request.

    Returns a PairValidationResult with detailed stage-by-stage results.
    """
    stages: List[ValidationStageResult] = []

    # Stage 1: Image count
    count_result = validate_image_count(request)
    stages.append(count_result)
    if not count_result.passed:
        return PairValidationResult(is_valid=False, stages=stages)

    t0, t1 = request.images[0], request.images[1]

    # Stage 2: File readability
    read_result = validate_file_readability([t0, t1])
    stages.append(read_result)
    if not read_result.passed:
        return PairValidationResult(is_valid=False, stages=stages)

    # Stage 3: Dimension/grid
    dim_result = validate_dimensions(t0, t1)
    stages.append(dim_result)
    # Channel mismatch is a hard failure
    if not dim_result.passed:
        return PairValidationResult(is_valid=False, stages=stages)

    # Stage 4: CRS
    crs_result = validate_crs(t0, t1)
    stages.append(crs_result)
    # CRS mismatch without reprojection enabled is blocking
    if not crs_result.passed and not allow_reprojection:
        return PairValidationResult(is_valid=False, stages=stages)

    # Stage 5: Spatial compatibility
    spatial_result, alignment_status = validate_spatial_compatibility(t0, t1, allow_reprojection)
    stages.append(spatial_result)
    if not spatial_result.passed:
        return PairValidationResult(
            is_valid=False,
            stages=stages,
            alignment_status=alignment_status,
        )

    # Stage 6: Timestamp
    ts_result = validate_timestamps(t0, t1)
    stages.append(ts_result)
    # Identical timestamps is a warning, not a hard block
    # (could be legitimate in some datasets)

    # Extract image info for downstream use
    t0_info = _extract_image_info(t0)
    t1_info = _extract_image_info(t1)

    all_passed = all(s.passed for s in stages)

    return PairValidationResult(
        is_valid=all_passed,
        stages=stages,
        t0_info=t0_info,
        t1_info=t1_info,
        alignment_status=alignment_status,
    )
