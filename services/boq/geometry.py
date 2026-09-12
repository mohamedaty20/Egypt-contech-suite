"""
services/boq/geometry.py

Pure geometry primitives for the BOQ engine.

Everything operates in the drawing's native units (typically mm).
No unit conversion happens here — that is handled at extraction time.

Key functions:
  - shoelace_area         : area of a closed polygon
  - bbox                  : axis-aligned bounding box
  - polyline_length       : total edge length of an open polyline
  - line_angle_deg        : angle of a segment in [0, 180)
  - line_midpoint         : midpoint of a segment
  - perpendicular_distance: signed offset between two parallel lines
  - parallel_lines        : detect if two segments are parallel within tolerance
  - point_in_polygon      : ray-casting test
  - point_to_segment_distance
  - is_axis_aligned       : horizontal or vertical within tolerance
"""

import math
print("[BOQ] geometry.py VERSION 4")
print("[BOQ] geometry.py VERSION 3 — tol=0.5, shorter-line rule")


# =====================================================================
# BASIC
# =====================================================================
def shoelace_area(points):
    """
    Area of a polygon in the same squared units as `points`.

    points: list of (x, y) tuples, either open or closed (first != last
    is fine — this function closes implicitly).

    Returns a positive float. If the polygon has < 3 points → 0.0.
    """
    if not points or len(points) < 3:
        return 0.0
    n = len(points)
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def bbox(points):
    """
    Axis-aligned bounding box.

    Returns (min_x, min_y, max_x, max_y).
    Empty input → (0, 0, 0, 0).
    """
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def polyline_length(points, closed=False):
    """
    Total length along a polyline.

    If closed=True, adds the closing edge.
    """
    if not points or len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    if closed:
        x1, y1 = points[-1]
        x2, y2 = points[0]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def is_closed(points, tolerance=1.0):
    """
    True if the polyline's first and last vertices are within tolerance.
    Tolerance in the same units as the points (default 1 mm).
    """
    if not points or len(points) < 3:
        return False
    x0, y0 = points[0]
    x1, y1 = points[-1]
    return math.hypot(x1 - x0, y1 - y0) <= tolerance


# =====================================================================
# LINE-LEVEL
# =====================================================================
def line_angle_deg(p1, p2):
    """
    Angle of a segment in degrees, normalized to [0, 180).
    Horizontal → 0. Vertical → 90. Reversed direction gives same angle.
    """
    x1, y1 = p1
    x2, y2 = p2
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return ang


def line_midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def line_length(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def is_axis_aligned(p1, p2, tolerance_deg=1.0):
    """
    True if the segment is horizontal or vertical within tolerance.
    """
    a = line_angle_deg(p1, p2)
    return (a <= tolerance_deg) or (abs(a - 90.0) <= tolerance_deg) or \
           (abs(a - 180.0) <= tolerance_deg)


# =====================================================================
# PARALLEL / PERPENDICULAR
# =====================================================================
def perpendicular_distance(line_a, line_b):
    """
    Perpendicular distance between two *parallel* lines.

    line_a, line_b: ((x1,y1), (x2,y2))
    Returns the offset distance, or None if lines are not parallel.
    Uses the midpoint of B projected onto A's normal.
    """
    (ax1, ay1), (ax2, ay2) = line_a
    (bx1, by1), (bx2, by2) = line_b

    dx, dy = ax2 - ax1, ay2 - ay1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    nx, ny = -dy / L, dx / L

    bmx, bmy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    # signed distance from B's midpoint to line A along A's normal
    return (bmx - ax1) * nx + (bmy - ay1) * ny


def parallel_lines(line_a, line_b,
                   angle_tol_deg=2.0,
                   min_offset=80.0,
                   max_offset=500.0,
                   length_tol_ratio=0.5):
         
    """
    Decide whether two segments form a *wall pair*.

    Conditions:
      - Angle difference <= angle_tol_deg
      - Perpendicular offset between min_offset and max_offset
      - Length difference ratio <= length_tol_ratio
      - Projection overlap >= 60% of the shorter segment

    Returns dict with:
      {
        "is_pair": bool,
        "offset":  float | None,   # wall thickness
        "length":  float | None,   # wall length (mean of the two)
      }
    """
    result = {"is_pair": False, "offset": None, "length": None}

    (ax1, ay1), (ax2, ay2) = line_a
    (bx1, by1), (bx2, by2) = line_b

    len_a = math.hypot(ax2 - ax1, ay2 - ay1)
    len_b = math.hypot(bx2 - bx1, by2 - by1)
    if len_a < 1 or len_b < 1:
        return result

    ang_a = line_angle_deg(line_a[0], line_a[1])
    ang_b = line_angle_deg(line_b[0], line_b[1])
    d_ang = abs(ang_a - ang_b)
    if d_ang > 90:
        d_ang = 180 - d_ang
    if d_ang > angle_tol_deg:
        return result

    if abs(len_a - len_b) / max(len_a, len_b) > length_tol_ratio:
        return result

    off = perpendicular_distance(line_a, line_b)
    if off is None:
        return result
    off_abs = abs(off)
    if off_abs < min_offset or off_abs > max_offset:
        return result

    # Projection overlap check
    dx, dy = ax2 - ax1, ay2 - ay1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    # B's endpoints projected onto A's axis
    tb1 = (bx1 - ax1) * ux + (by1 - ay1) * uy
    tb2 = (bx2 - ax1) * ux + (by2 - ay1) * uy
    b_lo, b_hi = min(tb1, tb2), max(tb1, tb2)
    a_lo, a_hi = 0.0, L
    overlap = max(0.0, min(b_hi, a_hi) - max(b_lo, a_lo))
    shorter = min(len_a, len_b)
    if shorter > 0 and (overlap / shorter) < 0.60:
        return result

    result["is_pair"] = True
    result["offset"] = off_abs
    result["length"] = (len_a + len_b) / 2.0
    return result


# =====================================================================
# POINT-IN-POLYGON
# =====================================================================
def point_in_polygon(pt, polygon):
    """
    Ray-casting. Returns True if `pt` is strictly inside `polygon`.
    Boundary counts as inside (uses >=).
    """
    if not polygon or len(polygon) < 3:
        return False
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # check if pt is on the segment
        if _on_segment((xj, yj), (xi, yi), (x, y)):
            return True
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi):
            inside = not inside
        j = i
    return inside


def _on_segment(a, b, p, tol=1e-6):
    """True if p is collinear with a-b and between them (with tolerance)."""
    (ax, ay), (bx, by), (px, py) = a, b, p
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > tol:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -tol:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    if dot - sq_len > tol:
        return False
    return True


def point_to_segment_distance(pt, a, b):
    """Shortest distance from point `pt` to segment a-b."""
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# =====================================================================
# POLYGON CONTAINMENT
# =====================================================================
def polygon_inside_polygon(inner, outer):
    """
    True if every vertex of `inner` is inside `outer`.
    Does NOT handle touching edges as "inside" — use with care.
    """
    if not inner or not outer:
        return False
    return all(point_in_polygon(p, outer) for p in inner)


def polygon_centroid(points):
    """
    Vertex-average centroid. Good enough for room-label association.
    Returns (cx, cy) or None.
    """
    if not points:
        return None
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return (cx, cy)


# =====================================================================
# ANGLE BUCKETING (used by wall-pair detection)
# =====================================================================
def angle_bucket(angle_deg, bucket_size=5.0):
    """
    Round an angle to the nearest bucket. Used to group lines that are
    'roughly the same direction' before pairwise comparison.
    Returns an int (the bucket index).
    """
    return int(round(angle_deg / bucket_size)) % int(180.0 / bucket_size)
