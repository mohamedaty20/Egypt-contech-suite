"""
services/boq/wall_processor.py

Takes raw element records (from extractor_dxf) and produces the FINAL
wall list, with parallel-line pairs merged, thickness measured, and
each wall classified as external or internal.

Handles EVERY geometry type on a wall layer:
  - LINE               (paired or single)
  - LWPOLYLINE closed  (long thin = wall segment; perimeter = building run)
  - LWPOLYLINE open    (each segment = one wall run)
  - POLYLINE           (same as LWPOLYLINE)
  - CIRCLE             (circumference = wall run, diameter = thickness)
  - ARC                (arc length = wall run)

Rule (hardcoded, cannot be skipped):
  Every wall drawn as TWO parallel lines must be merged into ONE wall.
  Never count the two sides as two separate walls.
"""

import math
print("[BOQ] wall_processor.py VERSION 4")

from .geometry import (
    bbox, line_angle_deg, line_midpoint, line_length,
    parallel_lines, angle_bucket, bbox as _bbox,
)


# =====================================================================
# CLASSIFICATION HELPERS
# =====================================================================
def _cat(rec):
    return (rec.get("category") or "").lower()


def _is_wall_rec(rec):
    return _cat(rec) == "wall"


def _is_line(rec):
    return (rec.get("geometry") or {}).get("type") == "line"


def _is_polyline(rec):
    return (rec.get("geometry") or {}).get("type") == "polyline"


def _is_circle(rec):
    return (rec.get("geometry") or {}).get("type") == "circle"


def _is_arc(rec):
    return (rec.get("geometry") or {}).get("type") == "arc"


def _closed(rec):
    return bool((rec.get("geometry") or {}).get("closed"))


# =====================================================================
# AREA / LENGTH HELPERS
# =====================================================================
def _polyline_segments(pts, closed=False):
    """Yield (p1, p2) for each segment of a polyline."""
    n = len(pts)
    if n < 2:
        return
    for i in range(n - 1):
        yield pts[i], pts[i + 1]
    if closed and n >= 3:
        yield pts[-1], pts[0]


def _polyline_sum_length(pts, closed=False):
    total = 0.0
    for p1, p2 in _polyline_segments(pts, closed=closed):
        total += line_length(p1, p2)
    return total


# =====================================================================
# WALL EXTRACTION FROM ANY ENTITY TYPE
# =====================================================================
def _wall_from_line(rec):
    """LINE on wall layer → single wall candidate."""
    pts = rec["geometry"].get("points") or []
    if len(pts) != 2:
        return None
    p1, p2 = pts
    length = rec["geometry"].get("length_mm") or line_length(p1, p2)
    return {
        "length_mm":   length,
        "centerline":  (p1, p2),
        "kind":        "line",
        "source_rec":  rec,
    }


def _wall_from_closed_polyline(rec, ext_thick, int_thick):
    """
    Closed polyline on a wall layer:
      - aspect >= 3 → long thin rectangle = wall segment (thickness = short side)
      - otherwise   → building / room outline. Perimeter becomes one long
                      wall run with user-supplied thickness.
    """
    pts = rec["geometry"].get("points") or []
    if len(pts) < 3:
        return None
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
            "length_mm":   length,
            "thickness_mm": thickness,
            "centerline":  (p1, p2),
            "kind":        "polyline_rect",
            "source_rec":  rec,
        }

    # Building outline → total perimeter as one long wall
    perim = _polyline_sum_length(pts, closed=True)
    if perim < 100:
        return None
    return {
        "length_mm":   perim,
        "thickness_mm": ext_thick,
        "centerline":  (pts[0], pts[-1]),
        "kind":        "polyline_outline",
        "source_rec":  rec,
        "bbox":        (minx, miny, maxx, maxy),
    }


def _walls_from_open_polyline(rec, thickness_mm):
    """
    Open polyline on a wall layer → each segment is one wall run.
    Thickness = user-supplied.
    """
    pts = rec["geometry"].get("points") or []
    if len(pts) < 2:
        return []
    out = []
    for p1, p2 in _polyline_segments(pts, closed=False):
        L = line_length(p1, p2)
        if L < 50:   # ignore degenerate sub-mm edges
            continue
        out.append({
            "length_mm":   L,
            "thickness_mm": thickness_mm,
            "centerline":  (p1, p2),
            "kind":        "polyline_segment",
            "source_rec":  rec,
        })
    return out


def _wall_from_circle(rec):
    """CIRCLE on wall layer → circumference = wall length, 2r = thickness."""
    g = rec["geometry"] or {}
    r = (g.get("width_mm") or 0) / 2.0
    if r <= 0:
        meta_r = (rec.get("meta") or {}).get("radius_mm")
        if meta_r:
            r = float(meta_r)
    if r <= 0:
        return None
    circumference = 2.0 * math.pi * r
    center = (g.get("points") or [(0, 0)])[0]
    return {
        "length_mm":   circumference,
        "thickness_mm": 2.0 * r,
        "centerline":  (center, center),
        "kind":        "circle",
        "source_rec":  rec,
    }


def _wall_from_arc(rec):
    """ARC on wall layer → arc length = wall run."""
    g = rec["geometry"] or {}
    L = g.get("length_mm")
    if not L:
        return None
    center = (g.get("points") or [(0, 0)])[0]
    meta = rec.get("meta") or {}
    r = float(meta.get("radius_mm") or (g.get("width_mm") or 0) / 2.0 or 0)
    return {
        "length_mm":   L,
        "thickness_mm": max(2.0 * r * 0.1, 120.0),  # heuristic
        "centerline":  (center, center),
        "kind":        "arc",
        "source_rec":  rec,
    }


# =====================================================================
# PAIR MERGING (works only on LINE candidates)
# =====================================================================
def _merge_line_pairs(line_candidates):
    """
    Given a list of {'length_mm', 'centerline', ...} from LINE entities,
    find parallel pairs. Returns (merged, leftover).
    """
    merged = []
    used = set()

    buckets = {}
    for i, c in enumerate(line_candidates):
        p1, p2 = c["centerline"]
        a = line_angle_deg(p1, p2)
        b = angle_bucket(a, bucket_size=5.0)
        buckets.setdefault(b, []).append(i)

    for _, idxs in buckets.items():
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
                la = tuple(line_candidates[i]["centerline"])
                lb = tuple(line_candidates[j]["centerline"])
                r = parallel_lines(la, lb)
                if not r["is_pair"]:
                    continue
                if best_offset is None or r["offset"] < best_offset:
                    best_offset = r["offset"]
                    best_j = j
                    best_pair = r
            if best_j is not None:
                c_i = line_candidates[i]
                c_j = line_candidates[best_j]
                # Use the SHORTER of the two parallel lines as the wall
                # reference face. For concentric building outlines this
                # is the void-facing (inner) line, which is what the
                # user wants measured. For ordinary double-line walls
                # the two lines are near-equal so either works.
                if c_i["length_mm"] <= c_j["length_mm"]:
                    ref_line = c_i["centerline"]
                    ref_len = c_i["length_mm"]
                else:
                    ref_line = c_j["centerline"]
                    ref_len = c_j["length_mm"]

                merged.append({
                    "length_mm":    ref_len,
                    "thickness_mm": best_pair["offset"],
                    "centerline":   ref_line,
                    "kind":         "pair",
                    "source_rec":   c_i["source_rec"],
                })
                used.add(i)
                used.add(best_j)

    leftover = [line_candidates[i] for i in range(len(line_candidates))
                if i not in used]
    return merged, leftover


def _bbox_nested(inner_bbox, outer_bbox, max_offset=600.0):
    """
    True if inner_bbox is nested inside outer_bbox with edge gap
    no greater than max_offset on every side.
    """
    ix0, iy0, ix1, iy1 = inner_bbox
    ox0, oy0, ox1, oy1 = outer_bbox
    # inner must be strictly inside outer
    if not (ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1):
        return False
    # must be a tight nesting — small gap
    gap = min(abs(ix0 - ox0), abs(iy0 - oy0), abs(ix1 - ox1), abs(iy1 - oy1))
    return gap <= max_offset


def _dedup_nested_polylines(polyline_walls, max_offset=600.0):
    """
    For concentric building outlines (outer + inner face of the same wall),
    drop the OUTER and keep the INNER (void-facing). This matches the
    rule: wall length is measured on the face nearest the void.

    For non-nested polylines, both are kept.
    """
    kept = []
    dropped = set()
    n = len(polyline_walls)
    for i in range(n):
        if i in dropped:
            continue
        a_bbox = polyline_walls[i].get("bbox")
        if a_bbox is None:
            kept.append(polyline_walls[i])
            continue
        for j in range(n):
            if i == j or j in dropped:
                continue
            b_bbox = polyline_walls[j].get("bbox")
            if b_bbox is None:
                continue
            if _bbox_nested(a_bbox, b_bbox, max_offset=max_offset):
                # a is inside b → drop b (outer), keep a (inner)
                dropped.add(j)
            elif _bbox_nested(b_bbox, a_bbox, max_offset=max_offset):
                # b is inside a → drop a (outer), keep b (inner)
                dropped.add(i)
                break
        if i not in dropped:
            kept.append(polyline_walls[i])
    return kept

# =====================================================================
# ENVELOPE
# =====================================================================
def _envelope_from_points(points):
    minx, miny, maxx, maxy = _bbox(points)
    return {
        "min_x": minx, "min_y": miny,
        "max_x": maxx, "max_y": maxy,
        "w": maxx - minx, "h": maxy - miny,
        "area_mm2": (maxx - minx) * (maxy - miny),
    }


def _distance_to_envelope_edge(midpoint, env):
    x, y = midpoint
    return min(abs(x - env["min_x"]), abs(x - env["max_x"]),
               abs(y - env["min_y"]), abs(y - env["max_y"]))


# =====================================================================
# MAIN
# =====================================================================
def process_walls(records, ext_thick=250.0, int_thick=120.0):
    """
    Produce the final wall list from ANY combination of entity types.
    """
    ext_thick = float(ext_thick or 250.0)
    int_thick = float(int_thick or 120.0)

    wall_recs = [r for r in records if _is_wall_rec(r)]

    # --- Gather every wall candidate regardless of geometry type ---
    line_candidates = []
    other_candidates = []
    all_points = []

    for rec in wall_recs:
        if _is_line(rec):
            c = _wall_from_line(rec)
            if c:
                line_candidates.append(c)
                all_points.extend(c["centerline"])
        elif _is_polyline(rec):
            pts = (rec["geometry"] or {}).get("points") or []
            all_points.extend(pts)
            if _closed(rec):
                c = _wall_from_closed_polyline(rec, ext_thick, int_thick)
                if c:
                    other_candidates.append(c)
            else:
                for c in _walls_from_open_polyline(rec, ext_thick):
                    other_candidates.append(c)
        elif _is_circle(rec):
            c = _wall_from_circle(rec)
            if c:
                other_candidates.append(c)
                all_points.extend((rec["geometry"] or {}).get("points") or [])
        elif _is_arc(rec):
            c = _wall_from_arc(rec)
            if c:
                other_candidates.append(c)
                all_points.extend((rec["geometry"] or {}).get("points") or [])

    if not all_points:
        return {
            "walls": [],
            "envelope": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0,
                         "w": 0, "h": 0, "area_mm2": 0},
            "stats": {"pairs_merged": 0, "singles": 0, "polylines": 0,
                      "circles": 0, "arcs": 0, "skipped": 0,
                      "total_wall_len_m": 0.0,
                      "external_len_m": 0.0, "internal_len_m": 0.0},
        }

    env = _envelope_from_points(all_points)

    # --- Merge LINE pairs ---
    merged_from_lines, leftover_lines = _merge_line_pairs(line_candidates)

    # --- Deduplicate nested polylines: keep the void-facing one ---
    outlines = [c for c in other_candidates
                if c.get("kind") == "polyline_outline"]
    non_outlines = [c for c in other_candidates
                    if c.get("kind") != "polyline_outline"]
    if outlines:
        before = len(outlines)
        outlines = _dedup_nested_polylines(outlines, max_offset=600.0)
        dropped = before - len(outlines)
        if dropped:
            print(f"[wall] dropped {dropped} nested outline(s) — "
                  f"kept innermost (void-facing)")
    other_candidates = non_outlines + outlines

    # --- Leftover single lines: wall with user-supplied ext thickness ---
    singles = []
    for c in leftover_lines:
        singles.append({
            "length_mm":   c["length_mm"],
            "thickness_mm": ext_thick,
            "centerline":  c["centerline"],
            "kind":        "single_line",
            "source_rec":  c["source_rec"],
        })

    # --- Everything combines ---
    all_walls = merged_from_lines + other_candidates + singles

    # --- Classify ext vs int by proximity to envelope ---
    prox_threshold = ext_thick * 1.5
    wall_records = []
    ext_len = 0.0
    int_len = 0.0

    for idx, w in enumerate(all_walls):
        p1, p2 = w["centerline"]
        if p1 == p2:
            # circle / arc — center only; treat as internal for classification
            d = prox_threshold + 1.0
        else:
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
            "layer":     (w["source_rec"] or {}).get("layer"),
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
            "confidence": "high" if w["kind"] == "pair" else "medium",
            "meta": {
                "method":          w["kind"],
                "distance_to_env": d,
            },
        })

    n_polylines = sum(1 for w in other_candidates if w["kind"].startswith("polyline"))
    n_circles   = sum(1 for w in other_candidates if w["kind"] == "circle")
    n_arcs      = sum(1 for w in other_candidates if w["kind"] == "arc")

    stats = {
        "pairs_merged":       len(merged_from_lines),
        "singles":            len(singles),
        "polylines":          n_polylines,
        "circles":            n_circles,
        "arcs":               n_arcs,
        "skipped":            0,
        "total_wall_len_m":   round((ext_len + int_len) / 1000.0, 3),
        "external_len_m":     round(ext_len / 1000.0, 3),
        "internal_len_m":     round(int_len / 1000.0, 3),
        "total_walls":        len(wall_records),
    }

    print(f"[wall] envelope: {env['w']:.0f} x {env['h']:.0f} mm  "
          f"area={env['area_mm2']/1e6:.2f} m2")
    print(f"[wall] pairs={stats['pairs_merged']}  singles={stats['singles']}  "
          f"polylines={stats['polylines']}  circles={stats['circles']}  "
          f"arcs={stats['arcs']}")
    print(f"[wall] TOTAL wall length = {stats['total_wall_len_m']} m "
          f"(ext={stats['external_len_m']}, int={stats['internal_len_m']})")

    return {"walls": wall_records, "envelope": env, "stats": stats}
