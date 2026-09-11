"""
services.boq — Architectural BOQ engine.

Batch 1: taxonomy + geometry primitives.
Batch 2: DXF extractor + wall pair processor.
"""

from .taxonomy import (
    LAYER_TAXONOMY, CATEGORY_GROUPS,
    classify_layer, normalize_layer_name, levenshtein,
    in_group, is_primary_architectural, is_annotation,
)
from .geometry import (
    shoelace_area, bbox, polyline_length, is_closed,
    line_angle_deg, line_midpoint, line_length, is_axis_aligned,
    perpendicular_distance, parallel_lines,
    point_in_polygon, point_to_segment_distance,
    polygon_inside_polygon, polygon_centroid,
    angle_bucket,
)
from .extractor_dxf import extract_elements
from .wall_processor import process_walls

__all__ = [
    # taxonomy
    "LAYER_TAXONOMY", "CATEGORY_GROUPS",
    "classify_layer", "normalize_layer_name", "levenshtein",
    "in_group", "is_primary_architectural", "is_annotation",
    # geometry
    "shoelace_area", "bbox", "polyline_length", "is_closed",
    "line_angle_deg", "line_midpoint", "line_length", "is_axis_aligned",
    "perpendicular_distance", "parallel_lines",
    "point_in_polygon", "point_to_segment_distance",
    "polygon_inside_polygon", "polygon_centroid",
    "angle_bucket",
    # pipeline
    "extract_elements", "process_walls",
]
