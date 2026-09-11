"""
services/boq/room_processor.py

Extracts rooms (closed polygons) from element records and classifies each
as WET or DRY using three signals:

  1. Text labels inside the room polygon (bath, kitchen, WC, مطبخ, حمام …)
  2. Furniture block names inside the room (toilet, sink, bathtub …)
  3. Layer name match (bathroom, kitchen, wet-area …)

No AI. Pure deterministic matching against curated keyword lists.

Rule (from user spec):
  - Wet rooms get ceramic tiles on their walls.
  - Dry rooms get plaster + paint on their walls.
"""

from .geometry import (
    bbox, shoelace_area, polyline_length, polygon_centroid,
    point_in_polygon,
)


# =====================================================================
# KEYWORD LISTS  (English + Egyptian Arabic transliterations)
# =====================================================================
WET_TEXT_KEYWORDS = [
    "bath", "bathroom", "bath room", "wc", "toilet", "restroom",
    "powder", "powder room", "washroom", "wash room",
    "kitchen", "kitchenette", "pantry", "laundry", "utility",
    "حمام", "دورة", "مطبخ", "تويلت", "مرحاض",
]

WET_FURNITURE_KEYWORDS = [
    "toilet", "wc", "sink", "basin", "bathtub", "tub",
    "shower", "bidet", "urinal",
    "kitchen", "counter", "stove", "cooktop", "oven",
    "dishwasher", "washer", "washing",
    "مرحاض", "حوض", "بانيو", "دش", "مطبخ", "فرن",
]

WET_LAYER_KEYWORDS = [
    "bath", "bathroom", "wc", "toilet", "kitchen", "wet",
    "حمام", "مطبخ", "مبلل",
]


# =====================================================================
# FILTERING
# =====================================================================
_EXCLUDED_CATEGORIES = {
    "wall", "column", "beam", "slab", "footing",
    "door", "window", "stair", "balcony", "void",
    "hatch", "dim", "text", "furniture", "grid",
    "section", "level", "boundary", "title_block",
}


def _is_room_candidate(rec):
    """A closed polygon that is not an excluded element type."""
    geom = rec.get("geometry") or {}
    if geom.get("type") != "polyline":
        return False
    if not geom.get("closed"):
        return False
    if rec.get("category") in _EXCLUDED_CATEGORIES:
        return False

    pts = geom.get("points") or []
    if len(pts) < 3:
        return False

    minx, miny, maxx, maxy = bbox(pts)
    bw = maxx - minx
    bh = maxy - miny
    if min(bw, bh) < 500:      # < 0.5 m in one dimension → wall, not room
        return False
    aspect = max(bw, bh) / min(bw, bh)
    if aspect > 4.0:           # extremely elongated → likely a wall run
        return False

    area = geom.get("area_mm2") or shoelace_area(pts)
    if area < 1_000_000:       # < 1 m²  → too small to be a room
        return False
    return True


# =====================================================================
# SIGNAL COLLECTION
# =====================================================================
def _collect_signals(rec, texts, inserts):
    """Return labels, furniture names, and layer-name signals for one room."""
    pts = rec["geometry"]["points"]
    centroid = polygon_centroid(pts) or (0.0, 0.0)

    inside_texts = []
    for t in texts:
        pos = t.get("position")
        if not pos:
            continue
        if point_in_polygon(pos, pts):
            inside_texts.append(t.get("content", ""))

    inside_furniture = []
    for f in inserts:
        pos = f.get("position")
        if not pos:
            continue
        if point_in_polygon(pos, pts):
            inside_furniture.append(f.get("block_name", ""))

    layer_signal = (rec.get("layer") or "").lower()
    return {
        "centroid": centroid,
        "labels": inside_texts,
        "furniture": inside_furniture,
        "layer": layer_signal,
    }


def _classify_wet(signals):
    """
    Apply the 3 signals and return (subtype, confidence, reasons).
    subtype: 'wet' | 'dry'
    confidence: 'high' | 'medium' | 'low'
    """
    reasons = []

    # Signal 1: text labels
    for label in signals["labels"]:
        low = str(label).lower()
        for kw in WET_TEXT_KEYWORDS:
            if kw in low:
                reasons.append(f"text:{kw}")
                break

    # Signal 2: furniture block names
    for blk in signals["furniture"]:
        low = str(blk).lower()
        for kw in WET_FURNITURE_KEYWORDS:
            if kw in low:
                reasons.append(f"furniture:{kw}")
                break

    # Signal 3: layer name
    layer = signals["layer"]
    for kw in WET_LAYER_KEYWORDS:
        if kw in layer:
            reasons.append(f"layer:{kw}")
            break

    if not reasons:
        return ("dry", "medium", [])

    has_text = any(r.startswith("text:") for r in reasons)
    has_furn = any(r.startswith("furniture:") for r in reasons)

    if has_text and has_furn:
        return ("wet", "high", reasons)
    if has_text:
        return ("wet", "high", reasons)
    if has_furn:
        return ("wet", "medium", reasons)
    return ("wet", "low", reasons)  # layer-only match


# =====================================================================
# MAIN
# =====================================================================
def process_rooms(records):
    """
    Extract rooms from raw records and classify each as wet/dry.

    Returns:
      {
        "rooms": [ {room record}, ... ],
        "stats": {
            "total": N, "wet": N, "dry": N,
            "wet_area_m2": F, "dry_area_m2": F,
        }
      }
    """
    # Collect text and INSERT positions from all records
    texts = []
    inserts = []
    for r in records:
        cat = r.get("category")
        if cat == "text":
            pos = None
            meta = r.get("meta") or {}
            if "text_position" in meta:
                pos = meta["text_position"]
            elif r.get("geometry", {}).get("points"):
                pos = r["geometry"]["points"][0]
            content = r.get("text") or ""
            if pos and content:
                texts.append({"position": pos, "content": content})
        elif r.get("geometry", {}).get("type") == "insert":
            pts = r["geometry"].get("points") or []
            if pts:
                inserts.append({
                    "position": pts[0],
                    "block_name": (r.get("meta") or {}).get("block_name", ""),
                })

    rooms = []
    wet_area = 0.0
    dry_area = 0.0
    n_wet = 0
    n_dry = 0

    for rec in records:
        if not _is_room_candidate(rec):
            continue

        pts = rec["geometry"]["points"]
        area = rec["geometry"].get("area_mm2") or shoelace_area(pts)
        perim = rec["geometry"].get("length_mm") or polyline_length(pts, closed=True)
        signals = _collect_signals(rec, texts, inserts)
        subtype, confidence, reasons = _classify_wet(signals)

        rooms.append({
            "id":        f"room_{len(rooms)+1:04d}",
            "category":  "room",
            "subtype":   subtype,
            "layer":     rec.get("layer"),
            "source":    "dxf",
            "units":     "mm",
            "geometry": {
                "type":       "polygon",
                "points":     pts,
                "length_mm":  perim,
                "width_mm":   None,
                "area_mm2":   area,
                "count":      1,
            },
            "text":       " | ".join(signals["labels"]) if signals["labels"] else None,
            "confidence": confidence,
            "meta": {
                "centroid":  signals["centroid"],
                "labels":    signals["labels"],
                "furniture": signals["furniture"],
                "reasons":   reasons,
            },
        })

        if subtype == "wet":
            wet_area += area / 1e6
            n_wet += 1
        else:
            dry_area += area / 1e6
            n_dry += 1

    stats = {
        "total":       len(rooms),
        "wet":         n_wet,
        "dry":         n_dry,
        "wet_area_m2": round(wet_area, 3),
        "dry_area_m2": round(dry_area, 3),
    }

    print(f"[rooms] total={stats['total']}  wet={stats['wet']}  dry={stats['dry']}")
    print(f"[rooms] wet_area={stats['wet_area_m2']} m2  dry_area={stats['dry_area_m2']} m2")

    return {"rooms": rooms, "stats": stats}
