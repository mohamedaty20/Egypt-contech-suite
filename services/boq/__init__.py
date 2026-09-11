"""
services.boq — Architectural BOQ engine.

Batch 1: taxonomy + geometry primitives.
No AI, no UI, no DXF parsing — those come in later batches.
"""

from .taxonomy import (
    LAYER_TAXONOMY,
    classify_layer,
    normalize_layer_name,
    levenshtein,
)
from .geometry import (
    shoelace_area,
    bbox,
    polyline_length,
    line_angle_deg,
    line_midpoint,
    perpendicular_distance,
    parallel_lines,
    point_in_polygon,
    point_to_segment_distance,
    is_axis_aligned,
)

__all__ = [
    "LAYER_TAXONOMY",
    "classify_layer",
    "normalize_layer_name",
    "levenshtein",
    "shoelace_area",
    "bbox",
    "polyline_length",
    "line_angle_deg",
    "line_midpoint",
    "perpendicular_distance",
    "parallel_lines",
    "point_in_polygon",
    "point_to_segment_distance",
    "is_axis_aligned",
]
