"""
services/boq/extractor_dxf.py

Reads an opened ezdxf document and produces a flat list of element records.

Does NOT do wall pairing — that is wall_processor.py's job.
Does NOT do AI classification — only the taxonomy matcher.
All coordinates are converted to MILLIMETRES at extraction time.
"""

import math

from .taxonomy import classify_layer
from .geometry import (
    shoelace_area, bbox, polyline_length, is_closed,
    line_length, line_midpoint, line_angle_deg,
)


# =====================================================================
# UNIT SCALES → mm
# =====================================================================
_UNIT_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0, "millimetre": 1.0, "millimeters": 1.0, "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0, "centimetre": 10.0, "centimeters": 10.0, "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0, "metre": 1000.0, "meters": 1000.0, "metres": 1000.0,
}


def _scale_for(units):
    key = str(units or "mm").strip().lower()
    return _UNIT_TO_MM.get(key, 1.0)


# =====================================================================
# HELPERS
# =====================================================================
def _safe_layer(entity):
    try:
        return entity.dxf.layer or "0"
    except Exception:
        return "0"


def _record_id(category, idx):
    return f"{category or 'unknown'}_{idx:05d}"


def _mk_record(category, subtype, layer, geom_type, points_mm,
               length_mm=None, width_mm=None, area_mm2=None,
               count=1, text=None, confidence="high",
               taxonomy_info=None, extra_meta=None):
    rec = {
        "id":     None,   # filled by caller with a running idx
        "category": category,
        "subtype":  subtype,
        "layer":    layer,
        "source":   "dxf",
        "units":    "mm",
        "geometry": {
            "type":       geom_type,
            "points":     points_mm,
            "length_mm":  length_mm,
            "width_mm":   width_mm,
            "area_mm2":   area_mm2,
            "count":      count,
        },
        "text":       text,
        "confidence": confidence,
        "meta":       extra_meta or {},
    }
    if taxonomy_info:
        rec["meta"]["taxonomy"] = {
            "matched":    taxonomy_info.get("matched"),
            "confidence": taxonomy_info.get("confidence"),
            "normalized": taxonomy_info.get("normalized"),
        }
    return rec


# =====================================================================
# ENTITY HANDLERS
# =====================================================================
def _read_line(entity, scale):
    try:
        s = entity.dxf.start
        e = entity.dxf.end
    except Exception:
        return None
    p1 = (float(s.x) * scale, float(s.y) * scale)
    p2 = (float(e.x) * scale, float(e.y) * scale)
    return {"type": "line", "points": [p1, p2]}


def _read_lwpolyline(entity, scale):
    try:
        pts_raw = list(entity.get_points("xy"))
    except Exception:
        return None
    if not pts_raw:
        return None
    pts = [(float(p[0]) * scale, float(p[1]) * scale) for p in pts_raw]
    try:
        closed_flag = bool(entity.closed)
    except Exception:
        closed_flag = False
    closed = closed_flag or is_closed(pts, tolerance=1.0)
    return {"type": "polyline", "points": pts, "closed": closed}


def _read_polyline(entity, scale):
    try:
        verts = list(entity.vertices)
    except Exception:
        return None
    pts = []
    for v in verts:
        try:
            loc = v.dxf.location
            pts.append((float(loc.x) * scale, float(loc.y) * scale))
        except Exception:
            continue
    if not pts:
        return None
    try:
        closed_flag = bool(entity.is_closed)
    except Exception:
        closed_flag = False
    closed = closed_flag or is_closed(pts, tolerance=1.0)
    return {"type": "polyline", "points": pts, "closed": closed}


def _read_circle(entity, scale):
    try:
        c = entity.dxf.center
        r = float(entity.dxf.radius) * scale
        cx = float(c.x) * scale
        cy = float(c.y) * scale
    except Exception:
        return None
    return {"type": "circle", "center": (cx, cy), "radius": r}


def _read_arc(entity, scale):
    try:
        c = entity.dxf.center
        r = float(entity.dxf.radius) * scale
        a0 = float(entity.dxf.start_angle)
        a1 = float(entity.dxf.end_angle)
        cx = float(c.x) * scale
        cy = float(c.y) * scale
    except Exception:
        return None
    return {"type": "arc", "center": (cx, cy), "radius": r,
            "start_angle": a0, "end_angle": a1}


def _read_hatch(entity, scale):
    """Return a list of boundary paths (each a list of (x,y) in mm)."""
    paths = []
    try:
        for p in entity.paths:
            verts = getattr(p, "vertices", None)
            if not verts:
                continue
            pts = [(float(v[0]) * scale, float(v[1]) * scale) for v in verts]
            if len(pts) >= 3:
                paths.append(pts)
    except Exception:
        pass
    return paths


def _read_text(entity, scale):
    try:
        ins = entity.dxf.insert
        x = float(ins.x) * scale
        y = float(ins.y) * scale
    except Exception:
        return None
    typ = entity.dxftype()
    try:
        if typ == "TEXT":
            content = entity.dxf.text or ""
        elif typ == "MTEXT":
            content = getattr(entity, "text", "") or entity.dxf.get("text", "")
        else:
            content = ""
    except Exception:
        content = ""
    return {"type": "text", "pos": (x, y), "content": str(content).strip()}


def _read_insert(entity, scale):
    try:
        ins = entity.dxf.insert
        x = float(ins.x) * scale
        y = float(ins.y) * scale
        name = entity.dxf.name or ""
    except Exception:
        return None
    return {"type": "insert", "pos": (x, y), "block_name": str(name)}


# =====================================================================
# MAIN ENTRY
# =====================================================================
def collect_unknown_layers(doc):
    """
    Fast pass: return the set of layer names in the DXF that the taxonomy
    could NOT classify. Used to ask Gemini to classify them before the
    full extraction pass.
    """
    seen = set()
    unknown = set()
    msp = doc.modelspace()
    for entity in msp:
        try:
            layer = entity.dxf.layer or "0"
        except Exception:
            continue
        if layer in seen:
            continue
        seen.add(layer)
        r = classify_layer(layer)
        if not r["category"]:
            unknown.add(layer)
    return sorted(unknown)


def extract_elements(doc, units="mm", max_block_depth=1, layer_overrides=None):
    """
    Extract every meaningful entity from doc.modelspace() into a flat
    list of records. Returns (records, stats).

    records: list of dicts, see _mk_record for schema.
    stats:   dict with counts per category and per dxftype.
    """
    scale = _scale_for(units)
    msp = doc.modelspace()
    records = []
    stats = {"by_category": {}, "by_dxftype": {}, "skipped": 0,
             "ai_overrides_used": 0}
    idx = 0
    layer_overrides = layer_overrides or {}

    for entity in msp:
        dxftype = entity.dxftype()
        stats["by_dxftype"][dxftype] = stats["by_dxftype"].get(dxftype, 0) + 1

        layer = _safe_layer(entity)
        tx = classify_layer(layer)
        category = tx["category"]
        subtype = tx["subtype"]

        # If taxonomy failed and an AI override exists for this layer, use it.
        if not category and layer in layer_overrides:
            ov = layer_overrides[layer]
            category = ov.get("category")
            subtype = ov.get("subtype")
            if category:
                tx = {
                    "category": category, "subtype": subtype,
                    "confidence": ov.get("confidence", "medium"),
                    "matched": f"ai:{layer}", "raw": layer,
                    "normalized": layer.lower(),
                }
                stats["ai_overrides_used"] += 1

        # -------- LINE --------
        if dxftype == "LINE":
            geom = _read_line(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            p1, p2 = geom["points"]
            rec = _mk_record(
                category, subtype, layer, "line", geom["points"],
                length_mm=line_length(p1, p2),
                width_mm=None, area_mm2=None,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"angle_deg": line_angle_deg(p1, p2),
                            "midpoint": line_midpoint(p1, p2)},
            )
            idx += 1
            rec["id"] = _record_id(category, idx)
            records.append(rec)

        # -------- LWPOLYLINE --------
        elif dxftype == "LWPOLYLINE":
            geom = _read_lwpolyline(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            pts = geom["points"]
            closed = geom["closed"]
            length = polyline_length(pts, closed=closed)
            area = shoelace_area(pts) if closed else None
            bb = bbox(pts)
            w_mm = bb[2] - bb[0]
            h_mm = bb[3] - bb[1]
            rec = _mk_record(
                category, subtype, layer, "polyline", pts,
                length_mm=length,
                width_mm=min(w_mm, h_mm),
                area_mm2=area,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"closed": closed,
                            "bbox_w": w_mm, "bbox_h": h_mm,
                            "aspect": (max(w_mm, h_mm) / min(w_mm, h_mm)) if min(w_mm, h_mm) > 1 else None},
            )
            idx += 1
            rec["id"] = _record_id(category, idx)
            records.append(rec)

        # -------- POLYLINE (old-style) --------
        elif dxftype == "POLYLINE":
            geom = _read_polyline(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            pts = geom["points"]
            closed = geom["closed"]
            length = polyline_length(pts, closed=closed)
            area = shoelace_area(pts) if closed else None
            bb = bbox(pts)
            w_mm = bb[2] - bb[0]
            h_mm = bb[3] - bb[1]
            rec = _mk_record(
                category, subtype, layer, "polyline", pts,
                length_mm=length,
                width_mm=min(w_mm, h_mm),
                area_mm2=area,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"closed": closed,
                            "bbox_w": w_mm, "bbox_h": h_mm,
                            "aspect": (max(w_mm, h_mm) / min(w_mm, h_mm)) if min(w_mm, h_mm) > 1 else None},
            )
            idx += 1
            rec["id"] = _record_id(category, idx)
            records.append(rec)

        # -------- CIRCLE --------
        elif dxftype == "CIRCLE":
            geom = _read_circle(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            r = geom["radius"]
            area = math.pi * r * r
            rec = _mk_record(
                category, subtype, layer, "circle", [geom["center"]],
                length_mm=2 * math.pi * r,
                width_mm=2 * r,
                area_mm2=area,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"radius_mm": r},
            )
            idx += 1
            rec["id"] = _record_id(category, idx)
            records.append(rec)

        # -------- ARC --------
        elif dxftype == "ARC":
            geom = _read_arc(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            # approximate arc length
            a0 = math.radians(geom["start_angle"])
            a1 = math.radians(geom["end_angle"])
            sweep = (a1 - a0) % (2 * math.pi)
            arc_len = geom["radius"] * sweep
            rec = _mk_record(
                category, subtype, layer, "arc", [geom["center"]],
                length_mm=arc_len,
                width_mm=2 * geom["radius"],
                area_mm2=None,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"radius_mm": geom["radius"],
                            "start_angle": geom["start_angle"],
                            "end_angle": geom["end_angle"]},
            )
            idx += 1
            rec["id"] = _record_id(category, idx)
            records.append(rec)

        # -------- HATCH --------
        elif dxftype == "HATCH":
            paths = _read_hatch(entity, scale)
            for path_pts in paths:
                area = shoelace_area(path_pts)
                if area < 1.0:  # ignore degenerate
                    continue
                rec = _mk_record(
                    category or "hatch", subtype, layer, "polyline", path_pts,
                    length_mm=polyline_length(path_pts, closed=True),
                    width_mm=None, area_mm2=area,
                    confidence=tx["confidence"] or "low",
                    taxonomy_info=tx,
                    extra_meta={"from_hatch": True},
                )
                idx += 1
                rec["id"] = _record_id(category or "hatch", idx)
                records.append(rec)

        # -------- TEXT / MTEXT --------
        elif dxftype in ("TEXT", "MTEXT"):
            geom = _read_text(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            rec = _mk_record(
                category or "text", None, layer, "text", [geom["pos"]],
                length_mm=None, width_mm=None, area_mm2=None,
                text=geom["content"],
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"text_position": geom["pos"]},
            )
            idx += 1
            rec["id"] = _record_id(category or "text", idx)
            records.append(rec)

        # -------- INSERT (block reference) --------
        elif dxftype == "INSERT":
            geom = _read_insert(entity, scale)
            if not geom:
                stats["skipped"] += 1; continue
            rec = _mk_record(
                category, subtype, layer, "insert", [geom["pos"]],
                length_mm=None, width_mm=None, area_mm2=None,
                confidence=tx["confidence"] or "low",
                taxonomy_info=tx,
                extra_meta={"block_name": geom["block_name"]},
            )
            idx += 1
            rec["id"] = _record_id(category or "insert", idx)
            records.append(rec)

        # -------- DIMENSION (skip geometry, but log) --------
        elif dxftype == "DIMENSION":
            rec = _mk_record(
                "dim", None, layer, "text", [],
                confidence="high",
                taxonomy_info=tx,
            )
            idx += 1
            rec["id"] = _record_id("dim", idx)
            records.append(rec)

        # -------- Other entity types: skip --------
        else:
            stats["skipped"] += 1
            continue

        if category:
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

    print(f"[extractor] units={units} scale={scale}")
    print(f"[extractor] dxftype counts: {stats['by_dxftype']}")
    print(f"[extractor] category counts: {stats['by_category']}")
    print(f"[extractor] total records: {len(records)}  skipped: {stats['skipped']}")
    return records, stats
