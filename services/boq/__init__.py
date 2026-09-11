"""
services/boq — Architectural BOQ engine.

Batch 1: taxonomy + geometry primitives.
Batch 2: DXF extractor + wall pair processor.
Batch 3: room classifier + BOQ engine (quantities only).
Batch 4: AI vision extractor + AI layer classifier.
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
from .extractor_dxf import extract_elements, collect_unknown_layers
from .wall_processor import process_walls
from .room_processor import process_rooms
from .engine import compute_boq, PARAM_DEFAULTS
from .classifier import classify_unknown_layers
from .extractor_image import extract_elements_from_image

__all__ = [
    "LAYER_TAXONOMY", "CATEGORY_GROUPS",
    "classify_layer", "normalize_layer_name", "levenshtein",
    "in_group", "is_primary_architectural", "is_annotation",
    "shoelace_area", "bbox", "polyline_length", "is_closed",
    "line_angle_deg", "line_midpoint", "line_length", "is_axis_aligned",
    "perpendicular_distance", "parallel_lines",
    "point_in_polygon", "point_to_segment_distance",
    "polygon_inside_polygon", "polygon_centroid",
    "angle_bucket",
    "extract_elements", "collect_unknown_layers",
    "process_walls", "process_rooms",
    "compute_boq", "PARAM_DEFAULTS",
    "classify_unknown_layers",
    "extract_elements_from_image",
]
