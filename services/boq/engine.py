"""
services/boq/engine.py

The BOQ engine. Applies every architectural formula from the master spec
and produces a QUANTITY-ONLY BOQ (no prices, no rates).

Inputs:
  walls_result    — from wall_processor.process_walls()
  rooms_result    — from room_processor.process_rooms()
  records         — raw records from extractor_dxf.extract_elements()
  params          — user-supplied parameters (see PARAM_DEFAULTS)

Output:
  {
    "summary": { ...derived quantities... },
    "boq":     [ {item, unit, qty, category, notes}, ... ],
    "totals":  { ...grouped quantities by category... },
  }
"""


# =====================================================================
# USER PARAMETERS (with defaults)
# =====================================================================
PARAM_DEFAULTS = {
    "ext_wall_thickness_m":   0.250,
    "int_wall_thickness_m":   0.120,
    "floor_height_m":         3.00,
    "door_height_m":          2.10,
    "window_height_m":        1.20,
    "wet_tile_height_m":      2.10,
    "mortar_bed_thickness_m": 0.020,
    "tile_size_m":            0.60,
    "grout_kg_per_m2":        0.50,
    "skirting_deduction":     0.10,
}


def _p(params, key):
    if not params:
        params = {}
    v = params.get(key)
    if v is None:
        return PARAM_DEFAULTS[key]
    return float(v)


# =====================================================================
# DOOR / WINDOW SUMMARY
# =====================================================================
def _summarize_openings(records, door_h_m, win_h_m):
    door_count = 0
    door_area_m2 = 0.0
    win_count = 0
    win_area_m2 = 0.0

    for r in records:
        cat = r.get("category")
        geom = r.get("geometry") or {}
        gt = geom.get("type")

        if cat == "door":
            width_mm = None
            if gt == "polyline":
                pts = geom.get("points") or []
                if len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    width_mm = max(max(xs) - min(xs), max(ys) - min(ys))
            elif gt == "arc":
                radius = (r.get("meta") or {}).get("radius_mm")
                if radius:
                    width_mm = 2 * radius
            elif gt == "line":
                width_mm = geom.get("length_mm")

            if width_mm and width_mm > 300:
                door_count += 1
                door_area_m2 += (width_mm / 1000.0) * door_h_m

        elif cat == "window":
            width_mm = None
            if gt == "polyline":
                pts = geom.get("points") or []
                if len(pts) >= 3:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    width_mm = max(max(xs) - min(xs), max(ys) - min(ys))
            elif gt == "line":
                width_mm = geom.get("length_mm")

            if width_mm and width_mm > 300:
                win_count += 1
                win_area_m2 += (width_mm / 1000.0) * win_h_m

    return door_count, door_area_m2, win_count, win_area_m2


# =====================================================================
# MAIN
# =====================================================================
def compute_boq(walls_result, rooms_result, records, params=None):
    """
    Produce the full architectural BOQ — quantities only, no prices.
    """
    ext_t_m     = _p(params, "ext_wall_thickness_m")
    int_t_m     = _p(params, "int_wall_thickness_m")
    floor_h_m   = _p(params, "floor_height_m")
    door_h_m    = _p(params, "door_height_m")
    win_h_m     = _p(params, "window_height_m")
    wet_tile_m  = _p(params, "wet_tile_height_m")
    mortar_m    = _p(params, "mortar_bed_thickness_m")
    tile_m      = _p(params, "tile_size_m")
    grout_kg    = _p(params, "grout_kg_per_m2")
    skirt_ded   = _p(params, "skirting_deduction")

    walls = walls_result.get("walls", [])
    ext_len_m = sum(w["geometry"]["length_mm"] for w in walls
                    if w["subtype"] == "external") / 1000.0
    int_len_m = sum(w["geometry"]["length_mm"] for w in walls
                    if w["subtype"] == "internal"
                    and (w.get("meta") or {}).get("method") != "polyline_outline") / 1000.0

    env = walls_result.get("envelope", {})
    env_area_m2 = (env.get("area_mm2") or 0.0) / 1e6

    ext_footprint_m2 = ext_len_m * ext_t_m
    int_footprint_m2 = int_len_m * int_t_m

    void_area_m2 = 0.0
    stair_area_m2 = 0.0
    for r in records:
        cat = r.get("category")
        area = (r.get("geometry") or {}).get("area_mm2") or 0.0
        if cat == "void":
            void_area_m2 += area / 1e6
        elif cat == "stair":
            stair_area_m2 += area / 1e6

    net_floor_m2 = env_area_m2 - int_footprint_m2 - stair_area_m2 - void_area_m2
    if net_floor_m2 < 0:
        net_floor_m2 = 0.0

    rstats = rooms_result.get("stats", {})
    wet_area_m2 = rstats.get("wet_area_m2", 0.0)
    dry_area_m2 = max(0.0, net_floor_m2 - wet_area_m2)

    ext_elev_m2 = ext_len_m * floor_h_m
    int_elev_m2 = int_len_m * floor_h_m * 2.0
    gross_wall_m2 = ext_elev_m2 + int_elev_m2

    door_count, door_area_m2, win_count, win_area_m2 = _summarize_openings(
        records, door_h_m, win_h_m
    )
    net_wall_m2 = max(0.0, gross_wall_m2 - door_area_m2 - win_area_m2)

    wet_perim_m = 0.0
    for r in rooms_result.get("rooms", []):
        if r["subtype"] == "wet":
            wet_perim_m += (r["geometry"].get("length_mm") or 0.0) / 1000.0
    wet_ceramic_wall_m2 = max(0.0, wet_perim_m * wet_tile_m - door_area_m2)

    dry_wall_m2 = max(0.0, net_wall_m2 - wet_ceramic_wall_m2)

    ceramic_floor_m2 = net_floor_m2
    mortar_m3 = ceramic_floor_m2 * mortar_m
    tile_count = ceramic_floor_m2 / (tile_m * tile_m) if tile_m > 0 else 0.0
    grout_total_kg = ceramic_floor_m2 * grout_kg

    ceiling_m2 = net_floor_m2

    total_perim_m = 0.0
    for r in rooms_result.get("rooms", []):
        total_perim_m += (r["geometry"].get("length_mm") or 0.0) / 1000.0
    skirting_len_m = total_perim_m * (1.0 - skirt_ded)

    wall_masonry_m2 = ext_elev_m2 + int_elev_m2

    summary = {
        "envelope_area_m2":         round(env_area_m2, 3),
        "total_wall_len_m":         round(ext_len_m + int_len_m, 3),
        "ext_wall_len_m":           round(ext_len_m, 3),   # internal use only
        "int_wall_len_m":           round(int_len_m, 3),   # internal use only
        "ext_wall_footprint_m2":    round(ext_footprint_m2, 3),
        "int_wall_footprint_m2":    round(int_footprint_m2, 3),
        "stair_area_m2":            round(stair_area_m2, 3),
        "void_area_m2":             round(void_area_m2, 3),
        "net_floor_area_m2":        round(net_floor_m2, 3),
        "wet_area_m2":              round(wet_area_m2, 3),
        "dry_area_m2":              round(dry_area_m2, 3),
        "gross_wall_elevation_m2":  round(gross_wall_m2, 3),
        "net_wall_finish_m2":       round(net_wall_m2, 3),
        "wet_ceramic_wall_m2":      round(wet_ceramic_wall_m2, 3),
        "dry_wall_m2":              round(dry_wall_m2, 3),
        "door_count":               door_count,
        "door_area_m2":             round(door_area_m2, 3),
        "window_count":             win_count,
        "window_area_m2":           round(win_area_m2, 3),
        "ceramic_floor_m2":         round(ceramic_floor_m2, 3),
        "ceramic_mortar_m3":        round(mortar_m3, 3),
        "ceramic_tile_count":       int(round(tile_count)),
        "ceramic_grout_kg":         round(grout_total_kg, 2),
        "ceiling_m2":               round(ceiling_m2, 3),
        "skirting_len_m":           round(skirting_len_m, 3),
        "wall_masonry_m2":          round(wall_masonry_m2, 3),
    }

    def _item(name, unit, qty, category, notes=""):
        return {
            "item":     name,
            "unit":     unit,
            "qty":      round(float(qty), 3),
            "category": category,
            "notes":    notes,
        }

    boq = []

    boq.append(_item("Wall masonry (brick / block)",
                     "m²", wall_masonry_m2, "masonry",
                     f"{ext_len_m:.1f} m ext + {int_len_m:.1f} m int × {floor_h_m:.2f} m"))
    boq.append(_item("Wall plastering — dry areas",
                     "m²", dry_wall_m2, "finishing",
                     "Dry walls after wet areas + openings"))
    boq.append(_item("Wall painting — dry areas",
                     "m²", dry_wall_m2, "finishing",
                     "2 coats emulsion"))

    if wet_ceramic_wall_m2 > 0:
        boq.append(_item("Wall ceramic tiles — wet areas",
                         "m²", wet_ceramic_wall_m2, "finishing",
                         f"Bathrooms + kitchen up to {wet_tile_m:.2f} m"))

    boq.append(_item("Floor ceramic tiles (60×60)",
                     "m²", ceramic_floor_m2, "finishing"))
    boq.append(_item("Floor mortar bed (20 mm)",
                     "m³", mortar_m3, "finishing"))
    boq.append(_item("Tile grout",
                     "kg", grout_total_kg, "finishing"))
    boq.append(_item("Ceiling plastering",
                     "m²", ceiling_m2, "finishing"))
    boq.append(_item("Ceiling painting",
                     "m²", ceiling_m2, "finishing"))
    boq.append(_item("Skirting board",
                     "m", skirting_len_m, "finishing",
                     f"{100*skirt_ded:.0f}% deducted for door openings"))

    if door_count > 0:
        boq.append(_item("Doors (supply + install)",
                         "unit", door_count, "openings",
                         f"avg area/unit: {(door_area_m2/max(1,door_count)):.2f} m²"))
    if win_area_m2 > 0:
        boq.append(_item("Windows (supply + install)",
                         "m²", win_area_m2, "openings"))

    totals = {}
    for b in boq:
        cat = b["category"]
        totals.setdefault(cat, []).append({
            "item": b["item"], "unit": b["unit"], "qty": b["qty"],
        })

    print(f"[engine] net_floor_area = {net_floor_m2:.2f} m²")
    print(f"[engine] wall elevation (gross) = {gross_wall_m2:.2f} m²  "
          f"net = {net_wall_m2:.2f} m²")
    print(f"[engine] BOQ items: {len(boq)}")

    return {
        "summary": summary,
        "boq":     boq,
        "totals":  totals,
    }
