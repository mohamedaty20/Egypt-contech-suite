"""
services/boq/wall_processor.py

Takes raw element records (from extractor_dxf) and produces the FINAL
wall list, with parallel-line pairs merged, thickness measured, and
each wall classified as external or internal.

Rule (hardcoded, cannot be skipped):
  Every wall drawn as TWO parallel lines must be merged into ONE wall.
  Never count the two sides as two separate walls.

Input:
  records     — list of dicts from extractor_dxf.extract_elements()
  ext_thick   — external wall thickness in mm (user input, default 250)
  int_thick   — internal wall thickness in mm (user input, default 120)

Output:
  {
    "walls": [ {wall record}, ... ],
    "envelope": { "min_x", "min_y", "max_x", "max_y", "w", "h", "area_mm2" },
    "stats": { "pairs_merged", "singles", "polylines", "skipped", "external_len_m", "internal_len_m" }
  }
"""

from .geometry import (
    bbox, line_angle_deg, line_midpoint, line_length,
    parallel_lines, angle_bucket, bbox as _bbox,
)


# =====================================================================
# UTILITIES
# =====================================================================
def _is_wall_line(rec):
    return (rec["category"] == "wall"
            and rec["geometry"]["type"] == "line"
            and len(rec["geometry"]["points"]) == 2)


def _is_wall_polyline(rec):
    return (rec["category"] == "wall"
            and rec["geometry"]["type"] == "polyline"
            and len(rec["geometry"]["points"]) >= 3
            and rec["geometry"].get("closed"))


def _wall_polyline_as_wall(rec, ext_thick, int_thick):
    """
    A closed wall polyline is either:
      (a) a wall drawn as a long thin rectangle — length = max(bbox), width = min(bbox)
      (b) an entire building outline polyline — treat as ONE wall run with
          thickness = user input (ext or int) and length = total perimeter.

    Rule: aspect ratio (long/short) >= 3  → case (a)
          otherwise                       → case (b)
    """
    pts = rec["geometry"]["points"]
    minx, miny, maxx, maxy = _bbox(pts)
    bw = maxx - minx
    bh = maxy - miny
    if min(bw, bh) < 1:
        return None
    aspect = max(bw, bh) / min(bw, bh)

    if aspect >= 3.0:
        # Long thin rectangle = wall segment
        length = max(bw, bh)
        thickness = min(bw, bh)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        if bw >= bh:
            p1 = (minx, cy); p2 = (maxx, cy)
        else:
            p1 = (cx, miny); p2 = (cx, maxy)
        return {
            "length_mm": length,
            "thickness_mm": thickness,
            "centerline": (p1, p2),
            "method": "polyline_rect",
        }
    else:
        # Building outline / room outline. Perimeter-based.
        # We treat it as ONE long run with user-supplied ext thickness.
        perim = rec["geometry"]["length_mm"] or 0.0
        return {
            "length_mm": perim,
            "thickness_mm": ext_thick,
            "centerline": (pts[0], pts[-1]) if pts else ((0, 0), (0, 0)),
            "method": "polyline_outline",
        }


def _envelope_from_points(points):
    minx, miny, maxx, maxy = _bbox(points)
    return {
        "min_x": minx, "min_y": miny,
        "max_x": maxx, "max_y": maxy,
        "w": maxx - minx, "h": maxy - miny,
        "area_mm2": (maxx - minx) * (maxy - miny),
    }


def _distance_to_envelope_edge(midpoint, env):
    """Min distance from a point to any of the 4 envelope edges."""
    x, y = midpoint
    d_left   = abs(x - env["min_x"])
    d_right  = abs(x - env["max_x"])
    d_bottom = abs(y - env["min_y"])
    d_top    = abs(y - env["max_y"])
    return min(d_left, d_right, d_bottom, d_top)


# =====================================================================
# PAIR MERGING (the core algorithm)
# =====================================================================
def _merge_pairs(wall_lines, ext_thick):
    """
    Given wall LINE records, find parallel pairs.

    Returns:
      merged_walls: list of { length_mm, thickness_mm, centerline, method }
      leftover_lines: list of wall LINE records that had no pair
    """
    merged = []
    used = set()

    # Bucket by angle
    buckets = {}
    for i, rec in enumerate(wall_lines):
        p1, p2 = rec["geometry"]["points"]
        a = line_angle_deg(p1, p2)
        b = angle_bucket(a, bucket_size=5.0)
        buckets.setdefault(b, []).append(i)

    # For each bucket, do O(n^2) pair search
    for bucket, idxs in buckets.items():
        n = len(idxs)
        for a_pos in range(n):
            i = idxs[a_pos]
            if i in used:
                continue
            best_j = None
            best_pair = None
            best_offset = None
            for b_pos in range(a_pos + 1, n):
                j = idxs[b_pos]
                if j in used:
                    continue
                la = tuple(wall_lines[i]["geometry"]["points"])
                lb = tuple(wall_lines[j]["geometry"]["points"])
                r = parallel_lines(la, lb)
                if not r["is_pair"]:
                    continue
                # Prefer the closest pair (smallest offset within range)
                if best_offset is None or r["offset"] < best_offset:
                    best_offset = r["offset"]
                    best_j = j
                    best_pair = r
            if best_j is not None:
                la = tuple(wall_lines[i]["geometry"]["points"])
                lb = tuple(wall_lines[best_j]["geometry"]["points"])
                mid_a = line_midpoint(*la)
                mid_b = line_midpoint(*lb)
                center_mid = ((mid_a[0] + mid_b[0]) / 2.0,
                              (mid_a[1] + mid_b[1]) / 2.0)
                # centerline = midpoint between the two lines, same direction as la
                ux = (la[1][0] - la[0][0])
                uy = (la[1][1] - la[0][1])
                L = (ux * ux + uy * uy) ** 0.5 or 1.0
                ux /= L; uy /= L
                half = best_pair["length"] / 2.0
                p1 = (center_mid[0] - ux * half, center_mid[1] - uy * half)
                p2 = (center_mid[0] + ux * half, center_mid[1] + uy * half)
                merged.append({
                    "length_mm": best_pair["length"],
                    "thickness_mm": best_pair["offset"],
                    "centerline": (p1, p2),
                    "method": "pair",
                })
                used.add(i); used.add(best_j)

    leftover = [wall_lines[i] for i in range(len(wall_lines)) if i not in used]
    return merged, leftover


# =====================================================================
# MAIN
# =====================================================================
def process_walls(records, ext_thick=250.0, int_thick=120.0):
    """
    Produce the final wall list.
    """
    ext_thick = float(ext_thick or 250.0)
    int_thick = float(int_thick or 120.0)

    wall_lines = [r for r in records if _is_wall_line(r)]
    wall_polys = [r for r in records if _is_wall_polyline(r)]

    # Compute envelope from ALL wall geometry (lines + polylines)
    all_points = []
    for r in wall_lines:
        all_points.extend(r["geometry"]["points"])
    for r in wall_polys:
        all_points.extend(r["geometry"]["points"])

    if not all_points:
        return {
            "walls": [],
            "envelope": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0,
                         "w": 0, "h": 0, "area_mm2": 0},
            "stats": {"pairs_merged": 0, "singles": 0, "polylines": 0,
                      "skipped": 0, "external_len_m": 0.0, "internal_len_m": 0.0},
        }

    env = _envelope_from_points(all_points)

    # Merge pairs
    merged_from_lines, leftover_lines = _merge_pairs(wall_lines, ext_thick)

    # Process polylines as walls
    merged_from_polys = []
    for rec in wall_polys:
        w = _wall_polyline_as_wall(rec, ext_thick, int_thick)
        if w:
            merged_from_polys.append(w)

    # Leftover single lines: thickness = user-supplied (ext by default)
    merged_from_singles = []
    for rec in leftover_lines:
        p1, p2 = rec["geometry"]["points"]
        length = rec["geometry"]["length_mm"] or line_length(p1, p2)
        merged_from_singles.append({
            "length_mm": length,
            "thickness_mm": ext_thick,
            "centerline": (p1, p2),
            "method": "single_line",
        })

    all_walls = merged_from_lines + merged_from_polys + merged_from_singles

    # Classify ext vs int by proximity to envelope edge
    # Threshold: within 1.5 × ext_thick of any envelope edge → external
    prox_threshold = ext_thick * 1.5
    wall_records = []
    ext_len = 0.0
    int_len = 0.0

    for idx, w in enumerate(all_walls):
        p1, p2 = w["centerline"]
        mid = line_midpoint(p1, p2)
        d = _distance_to_envelope_edge(mid, env)
        subtype = "external" if d <= prox_threshold else "internal"
        if subtype == "external":
            ext_len += w["length_mm"]
        else:
            int_len += w["length_mm"]

        wall_records.append({
            "id":        f"wall_{idx+1:04d}",
            "category":  "wall",
            "subtype":   subtype,
            "layer":     None,           # lost in merge, can be re-attached if needed
            "source":    "dxf",
            "units":     "mm",
            "geometry": {
                "type":       "wall",
                "points":     [p1, p2],
                "length_mm":  w["length_mm"],
                "width_mm":   w["thickness_mm"],
                "area_mm2":   w["length_mm"] * w["thickness_mm"],
                "count":      1,
            },
            "text":       None,
            "confidence": "high" if w["method"] == "pair" else "medium",
            "meta": {
                "method":          w["method"],
                "distance_to_env": d,
            },
        })

    stats = {
        "pairs_merged":    len(merged_from_lines),
        "singles":         len(merged_from_singles),
        "polylines":       len(merged_from_polys),
        "skipped":         0,
        "external_len_m":  round(ext_len / 1000.0, 3),
        "internal_len_m":  round(int_len / 1000.0, 3),
        "total_walls":     len(wall_records),
    }

    print(f"[wall] envelope: {env['w']:.0f} x {env['h']:.0f} mm  "
          f"area={env['area_mm2']/1e6:.2f} m2")
    print(f"[wall] pairs merged: {stats['pairs_merged']}  "
          f"singles: {stats['singles']}  polylines: {stats['polylines']}")
    print(f"[wall] ext_len = {stats['external_len_m']} m   "
          f"int_len = {stats['internal_len_m']} m")

    return {"walls": wall_records, "envelope": env, "stats": stats}
